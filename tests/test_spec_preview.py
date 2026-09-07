"""Golden and safety tests for the interpreted OptimizationSpec preview."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from modelsurgeon.conversation import (
    IntentField,
    IntentOutcome,
    IntentProvenance,
    IntentRecord,
    SourceSpan,
)
from modelsurgeon.search import (
    SpecPreviewError,
    build_spec_preview,
)
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    ContractMetric,
    HardConstraint,
    MetricUnit,
    ObjectiveContract,
    ObjectiveDirection,
    ObjectiveNormalization,
    SoftObjective,
)

REQUEST = "Keep quality above 0.95 and minimize latency."


def _spec(threshold: float = 0.95) -> dict[str, object]:
    return ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                ConstraintDirection.MINIMUM,
                threshold,
                MetricUnit.RATIO,
            ),
        ),
        objectives=(
            SoftObjective(
                ContractMetric.LATENCY,
                ObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                normalization=ObjectiveNormalization.IDENTITY,
            ),
        ),
    ).to_record()


def _intent(
    *,
    threshold: float = 0.95,
    budget: bool = False,
    outcome: IntentOutcome = IntentOutcome.EXECUTABLE,
) -> IntentRecord:
    span = SourceSpan("request", 0, len(REQUEST), REQUEST)
    fields = [
        IntentField(
            "constraint.quality",
            {
                "kind": "hard_constraint",
                "metric": "quality",
                "direction": "minimum",
                "threshold": threshold,
                "unit": "ratio",
            },
            "ratio",
            1.0,
            ("request",),
            True,
        ),
        IntentField(
            "objective.latency",
            {
                "kind": "preference",
                "metric": "latency",
                "direction": "minimize",
                "unit": "milliseconds",
                "normalization": "identity",
            },
            "milliseconds",
            1.0,
            ("request",),
        ),
    ]
    if budget:
        fields.append(
            IntentField(
                "budget.evaluations",
                {"kind": "budget", "value": 4, "unit": "count"},
                "count",
                1.0,
                ("request",),
            )
        )
    fields.sort(key=lambda item: item.field_id)
    effective_outcome = IntentOutcome.UNSUPPORTED if budget else outcome
    return IntentRecord(
        REQUEST,
        (span,),
        tuple(fields),
        (),
        (),
        IntentProvenance(
            "request-v1",
            "fixture-provider",
            "provider-v1",
            "tool-v1",
            ("evidence:request",),
        ),
        effective_outcome,
        ("typed interpretation complete",),
        _spec(threshold) if effective_outcome is IntentOutcome.EXECUTABLE else None,
    )


def test_preview_golden_rendering_and_exact_submission_spec() -> None:
    preview = build_spec_preview(_intent())
    expected = json.loads(
        (Path(__file__).parent / "golden" / "spec_preview_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert preview.to_record() == expected
    assert preview.canonical_json() == preview.canonical_json()
    assert json.loads(preview.render()) == expected
    assert preview.spec == expected["spec"]
    assert preview.spec_digest == expected["spec_digest"]

    with pytest.raises(SpecPreviewError, match="approval"):
        preview.confirm()
    submission = preview.confirm(approval_id="approval.fixture")
    assert submission.spec == preview.spec
    assert submission.spec_digest == preview.spec_digest

    calls: list[Mapping[str, object]] = []
    assert calls == []
    submission.submit(lambda spec: calls.append(spec) or "accepted")
    assert calls == [preview.spec]


def test_unresolved_and_unsupported_fields_are_visible_and_blocked() -> None:
    preview = build_spec_preview(_intent(budget=True))

    assert preview.outcome is IntentOutcome.UNSUPPORTED
    assert preview.spec is None
    assert preview.executable is False
    assert preview.budgets[0]["field_id"] == "budget.evaluations"
    assert "budget.evaluations" in preview.unresolved_fields
    with pytest.raises(SpecPreviewError, match="blocked"):
        preview.confirm(approval_id="approval.fixture")


def test_edit_recompile_creates_material_identity_and_visible_diff() -> None:
    original = build_spec_preview(_intent())
    edited = original.recompile(_intent(threshold=0.97))

    assert edited.spec_identity != original.spec_identity
    assert edited.spec_digest != original.spec_digest
    assert edited.diff is not None
    assert edited.diff.material
    assert "spec.constraints[0].threshold" in edited.diff.changed_paths
    assert "prior confirmation" in edited.diff.reason


def test_rejected_intent_has_no_spec_or_execution_authority() -> None:
    rejected = build_spec_preview(_intent(outcome=IntentOutcome.CLARIFICATION_REQUIRED))

    assert rejected.outcome is IntentOutcome.CLARIFICATION_REQUIRED
    assert rejected.spec is None
    assert rejected.unresolved_fields == ("intent",)
    assert rejected.approval_required is False

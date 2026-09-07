"""Acceptance tests for bounded measurable-target elicitation."""

from __future__ import annotations

from modelsurgeon.conversation import (
    IntentField,
    IntentOutcome,
    IntentProvenance,
    IntentRecord,
    InterpretationStep,
    MeasurableTargetStatus,
    SourceSpan,
    assess_measurable_targets,
)
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    HardConstraint,
    MetricUnit,
    ObjectiveContract,
    ObjectiveDirection,
    ObjectiveNormalization,
    SoftObjective,
)

REQUEST = "make it faster and fit on my GPU"
PROVENANCE = IntentProvenance(
    "target-test",
    "fixture-provider",
    "fixture-v1",
    "target-test-v1",
    ("evidence:request",),
)


def _intent(
    fields: tuple[IntentField, ...],
    *,
    outcome: IntentOutcome = IntentOutcome.CLARIFICATION_REQUIRED,
    spec: dict[str, object] | None = None,
) -> IntentRecord:
    fields = tuple(sorted(fields, key=lambda item: item.field_id))
    span = SourceSpan("request", 0, len(REQUEST), REQUEST)
    return IntentRecord(
        REQUEST,
        (span,),
        fields,
        (),
        (
            InterpretationStep(
                "target-step",
                "normalize",
                (span.span_id,),
                tuple(sorted(item.field_id for item in fields)),
                0.99,
                PROVENANCE.evidence_refs,
            ),
        ),
        PROVENANCE,
        outcome,
        ("target fixture",),
        spec,
    )


def _objective(metric: str = "latency") -> IntentField:
    return IntentField(
        f"objective.{metric}",
        {
            "kind": "soft_objective",
            "metric": metric,
            "direction": "minimize",
            "unit": "milliseconds",
            "normalization": "baseline_ratio",
        },
        "milliseconds",
        0.99,
        ("request",),
    )


def _constraint() -> IntentField:
    return IntentField(
        "constraint.quality",
        {
            "kind": "hard_constraint",
            "metric": "quality",
            "direction": "minimum",
            "threshold": 0.95,
            "unit": "ratio",
        },
        "ratio",
        0.99,
        ("request",),
        True,
    )


def _spec() -> dict[str, object]:
    return ObjectiveContract(
        (
            HardConstraint(
                "quality",
                ConstraintDirection.MINIMUM,
                0.95,
                MetricUnit.RATIO,
            ),
        ),
        (
            SoftObjective(
                "latency",
                ObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                normalization=ObjectiveNormalization.BASELINE_RATIO,
            ),
        ),
    ).to_record()


def test_vague_speed_and_gpu_request_gets_targeted_question() -> None:
    result = assess_measurable_targets(_intent((_objective(),)))

    assert result.status is MeasurableTargetStatus.NEEDS_CLARIFICATION
    assert len(result.questions) == 1
    assert result.questions[0].category == "missing-latency-target"
    assert "unit" in result.questions[0].prompt
    assert "default" in result.questions[0].prompt


def test_complete_objective_and_quality_constraint_need_no_question() -> None:
    intent = _intent(
        (_objective(), _constraint()),
        outcome=IntentOutcome.EXECUTABLE,
        spec=_spec(),
    )
    result = assess_measurable_targets(intent)

    assert result.status is MeasurableTargetStatus.COMPLETE
    assert result.questions == ()


def test_unsupported_metric_is_explicit_not_a_soft_claim() -> None:
    result = assess_measurable_targets(_intent((_objective("magic_score"),)))

    assert result.status is MeasurableTargetStatus.UNSUPPORTED
    assert result.questions == ()
    assert "magic_score" in result.diagnostics[0]

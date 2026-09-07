"""Acceptance tests for the closed v2.5 constraint-negotiation boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.audit_v25_constraint_negotiation_release import (
    ConstraintNegotiationReleaseAuditError,
    audit_release,
)

from modelsurgeon.conversation import (
    NegotiationPolicy,
    load_and_run_negotiation_study,
    load_negotiation_study,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "v2.5-constraint-negotiation-release-v1.json"
STUDY = ROOT / "tests" / "fixtures" / "negotiation_decision_quality_v1.json"


def test_v25_release_boundary_replays_and_audits() -> None:
    audit_release(ROOT)


def test_end_to_end_replay_retains_safe_and_negative_outcomes() -> None:
    corpus = load_negotiation_study(STUDY)
    run = load_and_run_negotiation_study(STUDY)

    assert run.passed
    assert len(run.results) == 36
    assert {item.status.value for scenario in corpus.scenarios for item in scenario.candidates} == {
        "failed",
        "inconclusive",
        "measured",
        "predicted",
        "unknown",
        "unsupported",
    }
    assert {item.explanation_outcome.value for item in run.results} >= {
        "infeasible",
        "predicted_only",
        "unsupported",
    }
    assert all(item.negative for item in run.results)
    assert any(item.inconclusive for item in run.results)

    measured = next(
        item for item in run.metrics if item.policy is NegotiationPolicy.MEASURED_PARETO
    )
    assert measured.constraint_preservation_rate == 1.0
    assert measured.measured_alternative_grounding_rate == 1.0
    assert measured.amendment_traceability_rate == 1.0
    assert measured.amendment_accuracy_rate == 1.0
    assert measured.misleading_claim_rate == 0.0


def test_direct_api_amendment_applications_preserve_original_history() -> None:
    run = load_and_run_negotiation_study(STUDY)
    cells = [
        item
        for item in run.results
        if item.policy is NegotiationPolicy.MEASURED_PARETO and item.amendment_attempted
    ]

    assert len(cells) == 6
    for cell in cells:
        assert cell.amendment_id is not None
        assert cell.amendment_diff_id is not None
        assert cell.amendment_application is not None
        application = cell.amendment_application
        assert application["original_spec_identity"] == cell.original_contract_id
        assert application["effective_spec_identity"] == cell.effective_contract_id
        assert application["original_campaign_id"] == next(
            scenario.campaign_id
            for scenario in load_negotiation_study(STUDY).scenarios
            if scenario.scenario_id == cell.scenario_id
        )
        assert application["downstream_campaign_id"] != application["original_campaign_id"]
        assert application["preserved_evidence_ids"] == list(cell.retained_evidence_ids)


def test_prediction_only_control_is_retained_but_never_shippable() -> None:
    run = load_and_run_negotiation_study(STUDY)
    prediction = [item for item in run.results if item.policy is NegotiationPolicy.PREDICTION_ONLY]

    assert prediction
    assert all(not item.shippable for item in prediction)
    assert all(not item.amendment_attempted for item in prediction)
    assert any(item.presented_alternative_statuses == ("predicted",) for item in prediction)


def test_release_audit_rejects_automatic_relaxation_claim(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["release_boundary"]["automatic_constraint_relaxation"] = "supported"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(
        ConstraintNegotiationReleaseAuditError, match="automatic_constraint_relaxation"
    ):
        audit_release(ROOT, manifest=path)


def test_release_audit_rejects_prediction_promotion(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["decision_quality_evidence"]["prediction_only_control"]["shippable"] = True
    path = tmp_path / "release.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ConstraintNegotiationReleaseAuditError, match="prediction-only"):
        audit_release(ROOT, manifest=path)

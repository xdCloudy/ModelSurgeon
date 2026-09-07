"""Reproducible v2.5 negotiation decision-quality study tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tools.audit_v25_negotiation_quality import audit_release

from modelsurgeon.conversation import (
    NegotiationPolicy,
    NegotiationStudyError,
    load_and_run_negotiation_study,
    load_negotiation_study,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "tests" / "fixtures" / "negotiation_decision_quality_v1.json"


def test_fixture_protocol_is_bounded_and_retains_negative_cells() -> None:
    study = load_negotiation_study(STUDY)

    assert study.protocol_id == "v25-negotiation-decision-quality-v1"
    assert len(study.scenarios) == 4
    assert {item.split.value for item in study.scenarios} == {
        "fixture",
        "held_out",
        "adversarial",
    }
    assert all(len(item.seeds) == 3 for item in study.scenarios)
    assert all(item.fixed_evaluation_count == 4 for item in study.scenarios)
    assert all(item.negative for item in study.scenarios)
    assert any(item.inconclusive for item in study.scenarios)


def test_measured_pareto_is_grounded_and_traces_explicit_amendments() -> None:
    run = load_and_run_negotiation_study(STUDY)
    metrics = next(item for item in run.metrics if item.policy is NegotiationPolicy.MEASURED_PARETO)

    assert run.passed
    assert run.policy_passed(NegotiationPolicy.MEASURED_PARETO)
    assert metrics.constraint_preservation_rate == 1.0
    assert metrics.measured_alternative_grounding_rate == 1.0
    assert metrics.amendment_traceability_rate == 1.0
    assert metrics.amendment_accuracy_rate == 1.0
    assert metrics.feasible_target_recovery_rate == 1.0
    assert metrics.misleading_claim_rate == 0.0
    assert metrics.silent_constraint_changes == 0
    assert metrics.errors == 0

    cells = [item for item in run.results if item.policy is NegotiationPolicy.MEASURED_PARETO]
    assert all(item.constraint_preserved and item.shippable for item in cells)
    assert all(item.amendment_traceable for item in cells if item.amendment_attempted)
    assert all(item.feasible_target_recovered for item in cells if item.recovery_target_expected)


def test_prediction_only_is_retained_as_a_non_shippable_negative_control() -> None:
    run = load_and_run_negotiation_study(STUDY)
    metrics = next(item for item in run.metrics if item.policy is NegotiationPolicy.PREDICTION_ONLY)
    cells = [item for item in run.results if item.policy is NegotiationPolicy.PREDICTION_ONLY]

    assert not run.policy_passed(NegotiationPolicy.PREDICTION_ONLY)
    assert metrics.misleading_claim_rate == 1.0
    prediction_cells = [item for item in cells if item.presented_alternative_ids]
    assert prediction_cells
    assert all(not item.shippable for item in prediction_cells)
    assert all(item.presented_alternative_statuses == ("predicted",) for item in prediction_cells)
    assert all(not item.amendment_attempted for item in cells)


def test_replay_ids_and_canonical_output_are_byte_stable() -> None:
    first = load_and_run_negotiation_study(STUDY)
    second = load_and_run_negotiation_study(STUDY)

    assert first.run_id == second.run_id
    assert first.canonical_json() == second.canonical_json()
    assert len({item.cell_id for item in first.results}) == len(first.results)


def test_study_rejects_more_than_three_seeds(tmp_path: Path) -> None:
    payload = STUDY.read_text(encoding="utf-8").replace(
        '"seeds": [\n        0,\n        1,\n        2\n      ]',
        '"seeds": [\n        0,\n        1,\n        2,\n        3\n      ]',
        1,
    )
    path = tmp_path / "study.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(NegotiationStudyError, match="seed bound"):
        load_negotiation_study(path)


def test_study_corpus_rejects_a_changed_evaluation_bound() -> None:
    study = load_negotiation_study(STUDY)
    with pytest.raises(NegotiationStudyError, match="exactly its fixed evaluations"):
        replace(study.scenarios[0], fixed_evaluation_count=3)


def test_checked_in_v25_audit_replays_the_protocol() -> None:
    audit_release(ROOT)

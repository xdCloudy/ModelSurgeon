"""Reproducible v2.8 evidence-grounding coverage and factuality tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from tools.audit_evidence_factuality_study import audit_release

from modelsurgeon.evaluation.evidence_factuality_study import (
    EvidenceFactualityMethod,
    EvidenceFactualityStudyError,
    load_and_run_evidence_factuality_study,
    load_evidence_factuality_study,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "tests" / "fixtures" / "evidence_factuality_study_v1.json"


def test_protocol_has_fixed_splits_bounded_seeds_and_retained_negative_cells() -> None:
    study = load_evidence_factuality_study(STUDY)

    assert study.protocol_id == "v28-evidence-factuality-v1"
    assert len(study.cases) == 6
    assert {item.split.value for item in study.cases} == {"fixture", "held_out"}
    assert study.limits.seeds == (0, 1, 2)
    assert all(item.retained for item in study.cases)
    assert any(
        claim.status.value in {"unsupported", "inconclusive"}
        for case in study.cases
        for claim in case.gold_claims
    )


def test_grounded_renderer_meets_release_thresholds_and_retains_snapshot_identity() -> None:
    run = load_and_run_evidence_factuality_study(STUDY)
    metrics = next(
        item for item in run.metrics if item.method is EvidenceFactualityMethod.GROUNDED_RENDERER
    )
    cells = [
        item for item in run.results if item.method is EvidenceFactualityMethod.GROUNDED_RENDERER
    ]

    assert run.passed
    assert run.policy_passed(EvidenceFactualityMethod.GROUNDED_RENDERER)
    assert metrics.claim_source_precision == 1.0
    assert metrics.claim_source_recall == 1.0
    assert metrics.source_id_correctness == 1.0
    assert metrics.measured_predicted_confusion_rate == 0.0
    assert metrics.negative_evidence_coverage == 1.0
    assert metrics.uncertainty_disclosure == 1.0
    assert metrics.uncertainty_calibration == 1.0
    assert metrics.unsupported_claim_rate == 0.0
    assert metrics.deterministic_ids == 1.0
    assert metrics.critical_escapes == 0
    assert all(item.snapshot_id and item.snapshot_digest for item in cells)
    assert all(item.response_digest and item.explanation_digest for item in cells)
    assert all(item.retained for item in run.results)


def test_unconstrained_text_is_a_retained_never_shippable_negative_control() -> None:
    run = load_and_run_evidence_factuality_study(STUDY)
    metrics = next(
        item for item in run.metrics if item.method is EvidenceFactualityMethod.UNCONSTRAINED_TEXT
    )

    assert not run.policy_passed(EvidenceFactualityMethod.UNCONSTRAINED_TEXT)
    assert metrics.claim_source_precision == 0.0
    assert metrics.claim_source_recall == 0.0
    assert metrics.measured_predicted_confusion_rate > 0.0
    assert metrics.negative_evidence_coverage == 0.0
    assert metrics.unsupported_claim_rate == 1.0
    assert metrics.critical_escapes == 18
    assert metrics.failures == 18
    assert all(item.retained for item in run.results)


def test_replay_ids_and_canonical_output_are_byte_stable() -> None:
    first = load_and_run_evidence_factuality_study(STUDY)
    second = load_and_run_evidence_factuality_study(STUDY)

    assert first.run_id == second.run_id
    assert first.canonical_json() == second.canonical_json()
    assert len({item.result_id for item in first.results}) == len(first.results)


def test_study_rejects_more_than_three_seeds() -> None:
    study = load_evidence_factuality_study(STUDY)
    with pytest.raises(EvidenceFactualityStudyError, match="at most three"):
        replace(study.limits, seeds=(0, 1, 2, 3))


def test_checked_in_audit_replays_the_protocol() -> None:
    audit_release(ROOT)

from __future__ import annotations

import pytest

from modelsurgeon.evaluation import (
    CONSUMER_REPAIR_STUDY_SCHEMA_VERSION,
    DEFAULT_CONSUMER_REPAIR_STUDY,
    ConsumerRepairArm,
    ConsumerRepairCell,
    ConsumerRepairEvidence,
    ConsumerRepairInterval,
    ConsumerRepairKey,
    ConsumerRepairMetric,
    ConsumerRepairOutcome,
    ConsumerRepairStudyConfig,
    ConsumerRepairStudyError,
    RepairRecommendation,
    build_consumer_repair_study,
)

SOURCE = "a" * 64
ARTIFACT = "b" * 64
TEACHER = "c" * 64


def _key(arm: ConsumerRepairArm) -> ConsumerRepairKey:
    return ConsumerRepairKey("llama", "small", "low_vram_12gb", "low", "short", arm, 0)


def _evidence(
    key: ConsumerRepairKey,
    quality: float,
    deployment_cost: float,
    outcome: ConsumerRepairOutcome = ConsumerRepairOutcome.MEASURED,
) -> ConsumerRepairEvidence:
    values = {
        "artifact_bytes": 1_000.0,
        "deployment_cost": deployment_cost,
        "energy_joules": 10.0,
        "failure_rate": 0.0,
        "heldout_quality": quality,
        "overfit_rate": 0.0,
        "quality_gain_per_joule": 0.01,
        "quality_gain_per_second": 0.05,
        "quality_gain_per_token": 0.001,
        "repair_seconds": 2.0,
        "rollback_rate": 0.0,
        "tokens_per_second": 20.0,
    }
    return ConsumerRepairEvidence(
        key,
        outcome,
        SOURCE,
        TEACHER,
        ARTIFACT,
        "data-v1",
        "hardware-v1",
        tuple(ConsumerRepairMetric(name, values[name]) for name in sorted(values)),
        (
            ConsumerRepairInterval(
                "deployment_cost", deployment_cost - 1, deployment_cost + 1, 0.95
            ),
            ConsumerRepairInterval("heldout_quality", quality - 0.01, quality + 0.01, 0.95),
        ),
        3,
        ("evaluator-v1", "tool-v1"),
        None if outcome is ConsumerRepairOutcome.MEASURED else "negative held-out result",
    )


def _study(*cells: ConsumerRepairCell):
    config = ConsumerRepairStudyConfig(require_complete_matrix=False)
    return build_consumer_repair_study(cells, config)


def test_default_protocol_is_complete_and_does_not_claim_unmeasured_success() -> None:
    study = DEFAULT_CONSUMER_REPAIR_STUDY

    assert study.schema_version == CONSUMER_REPAIR_STUDY_SCHEMA_VERSION
    assert len(study.cells) == 864
    assert study.summary()["claim"] == "unsupported"
    assert all(
        cell.evidence.outcome is ConsumerRepairOutcome.UNSUPPORTED for cell in study.cells
    )


def test_matched_heldout_evidence_classifies_beneficial_and_is_deterministic() -> None:
    baseline = ConsumerRepairCell(
        _key(ConsumerRepairArm.NO_REPAIR),
        _evidence(_key(ConsumerRepairArm.NO_REPAIR), 0.80, 100.0),
    )
    repair = ConsumerRepairCell(
        _key(ConsumerRepairArm.FIXED_REPAIR),
        _evidence(_key(ConsumerRepairArm.FIXED_REPAIR), 0.84, 110.0),
    )

    first = _study(baseline, repair)
    second = _study(repair, baseline)

    recommendation = first.recommendations()[0]
    assert recommendation.recommendation is RepairRecommendation.BENEFICIAL
    assert recommendation.selected_arm is ConsumerRepairArm.FIXED_REPAIR
    assert first.study_id == second.study_id


def test_failed_repair_is_infeasible_and_partial_terminal_evidence_is_rejected() -> None:
    baseline = ConsumerRepairCell(
        _key(ConsumerRepairArm.NO_REPAIR),
        _evidence(_key(ConsumerRepairArm.NO_REPAIR), 0.80, 100.0),
    )
    failed = ConsumerRepairCell(
        _key(ConsumerRepairArm.FIXED_REPAIR),
        ConsumerRepairEvidence(
            _key(ConsumerRepairArm.FIXED_REPAIR),
            ConsumerRepairOutcome.FAILED,
            None,
            None,
            None,
            None,
            None,
            (),
            (),
            None,
            (),
            "12 GiB profile exceeded during repair",
        ),
    )
    recommendation = _study(baseline, failed).recommendations()[0]
    assert recommendation.recommendation is RepairRecommendation.INFEASIBLE

    with pytest.raises(ConsumerRepairStudyError, match="partial"):
        ConsumerRepairEvidence(
            _key(ConsumerRepairArm.FIXED_REPAIR),
            ConsumerRepairOutcome.UNSUPPORTED,
            SOURCE,
            None,
            None,
            None,
            None,
            (),
            (),
            None,
            (),
            "unsupported",
        )

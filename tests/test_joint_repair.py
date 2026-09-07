from __future__ import annotations

import pytest

from modelsurgeon.search import (
    JointRepairAction,
    JointRepairBudget,
    JointRepairCandidate,
    JointRepairDecisionStatus,
    JointRepairEvidence,
    JointRepairEvidenceStatus,
    JointRepairOutcome,
    JointRepairPrediction,
    JointRepairSearchConfig,
    build_joint_repair_study,
    joint_candidate_id,
    record_joint_repair_evidence,
)
from modelsurgeon.surgery import RepairCost, RepairMethod

SOURCE = "a" * 64
PARENT = "b" * 64
FINAL = "c" * 64
BUDGET = JointRepairBudget(8, 100, 10_000, 60.0, 30.0, 10_000)
ACTION = JointRepairAction(RepairMethod.LORA, "short", "q4_k", "repair_then_quantize")


def _candidate(damage: float, quality_low: float = 0.8) -> JointRepairCandidate:
    candidate_id = joint_candidate_id("state_parent", SOURCE, "llama", damage, ACTION)
    return JointRepairCandidate(
        candidate_id,
        "state_parent",
        SOURCE,
        "llama",
        damage,
        0.7,
        100.0,
        ACTION,
        BUDGET,
        JointRepairPrediction(
            JointRepairEvidenceStatus.PREDICTED,
            quality_low,
            quality_low + 0.02,
            quality_low + 0.04,
            0.1,
            0.15,
            0.2,
            110.0,
            0.9,
            ("cost-predictor-v1", "recoverability-v1"),
        ),
    )


def _evidence(candidate: JointRepairCandidate, quality: float) -> JointRepairEvidence:
    return JointRepairEvidence(
        candidate.candidate_id,
        JointRepairOutcome.MEASURED,
        quality,
        110.0,
        RepairCost(3, 100, 2.0, 1.0, 3.0, 100),
        1.0,
        100,
        PARENT,
        SOURCE,
        FINAL,
        ("heldout-0",),
        False,
        True,
        ("quant-tool-v1", "repair-tool-v1"),
    )


def test_prediction_can_acquire_larger_damage_but_cannot_promote_it() -> None:
    candidate = _candidate(0.8, 0.9)
    study = build_joint_repair_study((candidate,), JointRepairSearchConfig())

    assert study.predicted_frontier() == (candidate,)
    decision = study.decide()
    assert decision.status is JointRepairDecisionStatus.NO_MEASURED_ARTIFACT
    assert decision.selected is None
    assert decision.acquisition_candidates == (candidate.candidate_id,)


def test_measured_frontier_selects_final_quality_and_retains_negative_cells() -> None:
    lower_damage = _candidate(0.2, 0.8)
    higher_damage = _candidate(0.8, 0.9)
    lower_damage = record_joint_repair_evidence(lower_damage, _evidence(lower_damage, 0.86))
    higher_damage = record_joint_repair_evidence(higher_damage, _evidence(higher_damage, 0.91))
    study = build_joint_repair_study((higher_damage, lower_damage))

    decision = study.decide()

    assert decision.status is JointRepairDecisionStatus.SELECTED
    assert decision.selected is higher_damage
    assert study.measured_frontier() == (higher_damage,)

    negative = JointRepairEvidence(
        lower_damage.candidate_id,
        JointRepairOutcome.NEGATIVE_RESULT,
        0.75,
        115.0,
        RepairCost(3, 100, 2.0, 1.0, 3.0, 100),
        1.0,
        100,
        PARENT,
        SOURCE,
        FINAL,
        ("heldout-0",),
        True,
        False,
        ("quant-tool-v1", "repair-tool-v1"),
        "held-out quality did not clear the promotion threshold",
    )
    assert record_joint_repair_evidence(lower_damage, negative).evidence is negative


def test_terminal_and_incompatible_predictions_are_not_promoted() -> None:
    incompatible = JointRepairAction(
        RepairMethod.LORA,
        "short",
        "q4_k",
        "quantize_then_repair",
        compatible=False,
        compatibility_reason="runtime cannot edit quantized tensors",
    )
    candidate = _candidate(0.5)
    candidate = JointRepairCandidate(
        candidate.candidate_id,
        candidate.parent_state_id,
        candidate.source_artifact_digest,
        candidate.model_family,
        candidate.initial_damage,
        candidate.initial_quality,
        candidate.initial_deployment_cost,
        incompatible,
        candidate.budget,
        JointRepairPrediction(
            JointRepairEvidenceStatus.UNSUPPORTED,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (),
            "runtime cannot edit quantized tensors",
        ),
    )
    decision = build_joint_repair_study((candidate,)).decide()

    assert decision.status is JointRepairDecisionStatus.UNSUPPORTED
    with pytest.raises(ValueError, match="terminal"):
        JointRepairPrediction(
            JointRepairEvidenceStatus.UNSUPPORTED,
            0.8,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (),
            "unsupported",
        )

"""Fail-closed promotion and deterministic replan tests."""

from __future__ import annotations

import pytest

from modelsurgeon.experiments.coordinator import (
    ApprovedCampaignPlan,
    AutonomousCampaignCoordinator,
    CoordinatorError,
    MeasuredCandidateEvidence,
    PromotionOutcome,
)
from modelsurgeon.surgery.contracts import TransactionState


def _plan() -> ApprovedCampaignPlan:
    return ApprovedCampaignPlan(
        "plan-1",
        "sha256:" + "a" * 64,
        "objective_contract_1",
        "space_1",
        ("candidate_1", "candidate_2", "candidate_3"),
        7,
        3,
        "approval-1",
    )


def _evidence(
    candidate_id: str = "candidate_1",
    *,
    measured: bool = True,
    complete: bool = True,
    constraints_passed: bool = True,
    transaction_state: TransactionState = TransactionState.COMMITTED,
    artifact_digest: str | None = "sha256:" + "b" * 64,
    source_artifact_digest: str = "sha256:" + "a" * 64,
    artifact_immutable: bool = True,
) -> MeasuredCandidateEvidence:
    return MeasuredCandidateEvidence(
        candidate_id,
        "state_" + candidate_id.removeprefix("candidate_") + "0" * 63,
        "evaluation_" + candidate_id.removeprefix("candidate_"),
        artifact_digest,
        source_artifact_digest,
        measured,
        complete,
        constraints_passed,
        transaction_state,
        artifact_immutable,
    )


def test_predicted_partial_and_failed_states_never_promote() -> None:
    coordinator = AutonomousCampaignCoordinator(_plan())
    predicted = coordinator.promote(_evidence(measured=False))
    assert predicted.outcome is PromotionOutcome.UNKNOWN
    rejected = coordinator.promote(_evidence("candidate_2", constraints_passed=False))
    assert rejected.outcome is PromotionOutcome.REJECTED
    uncommitted = coordinator.promote(
        _evidence("candidate_3", transaction_state=TransactionState.APPLIED)
    )
    assert uncommitted.outcome is PromotionOutcome.REJECTED


def test_measured_immutable_child_promotes_and_repeated_call_is_idempotent() -> None:
    coordinator = AutonomousCampaignCoordinator(_plan())
    evidence = _evidence()
    first = coordinator.promote(evidence)
    second = coordinator.promote(evidence)
    assert first.outcome is PromotionOutcome.PROMOTED
    assert first.promoted
    assert second == first


def test_source_change_and_duplicate_artifact_fail_closed() -> None:
    coordinator = AutonomousCampaignCoordinator(_plan())
    changed_source = coordinator.promote(_evidence(source_artifact_digest="sha256:" + "c" * 64))
    assert changed_source.outcome is PromotionOutcome.FAILED
    duplicate = coordinator.promote(_evidence("candidate_2", artifact_digest="sha256:" + "b" * 64))
    assert duplicate.outcome is PromotionOutcome.PROMOTED
    duplicate_again = coordinator.promote(
        _evidence("candidate_3", artifact_digest="sha256:" + "b" * 64)
    )
    assert duplicate_again.outcome is PromotionOutcome.FAILED


def test_replan_excludes_rejections_but_retains_unknowns() -> None:
    coordinator = AutonomousCampaignCoordinator(_plan())
    coordinator.promote(_evidence("candidate_1", constraints_passed=False))
    coordinator.promote(_evidence("candidate_2", measured=False))
    replan = coordinator.replan()
    assert replan.rejected_candidate_ids == ("candidate_1",)
    assert replan.next_candidate_ids == ("candidate_2", "candidate_3")
    assert replan.next_plan_id.startswith("campaign_plan_")


def test_plan_rejects_noncanonical_candidates() -> None:
    with pytest.raises(CoordinatorError, match="canonical candidate"):
        ApprovedCampaignPlan(
            "plan-1",
            "sha256:" + "a" * 64,
            "objective_contract_1",
            "space_1",
            ("bad",),
            7,
            1,
            "approval-1",
        )

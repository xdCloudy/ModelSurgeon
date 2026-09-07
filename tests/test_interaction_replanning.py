from __future__ import annotations

import pytest

from modelsurgeon.active_learning import (
    ExperimentResources,
    InteractionAcquisitionConfig,
    InteractionAction,
    InteractionCandidate,
    InteractionOutcome,
    InteractionOutcomeStatus,
    InteractionPlanningError,
    InteractionPlanState,
    plan_interaction_acquisition,
    replan_after_outcome,
)


def _candidate(candidate_id: str, state: str, score: float) -> InteractionCandidate:
    return InteractionCandidate(
        candidate_id,
        state,
        "predictor-v1",
        score,
        0.9,
        0.2,
        score,
        0.1,
        ExperimentResources(gpu_seconds=1.0),
    )


def test_stale_candidates_are_rejected_and_negative_outcomes_replan() -> None:
    config = InteractionAcquisitionConfig(
        max_candidates=2,
        max_resources=ExperimentResources(gpu_seconds=2.0),
    )
    plan = plan_interaction_acquisition(
        "state-a",
        (_candidate("cand_good", "state-a", 1.0), _candidate("cand_stale", "state-old", 9.0)),
        config=config,
    )
    assert tuple(item.candidate_id for item in plan.selections) == ("cand_good",)
    assert plan.rejected_stale_candidate_ids == ("cand_stale",)
    result = replan_after_outcome(
        plan,
        InteractionOutcome(
            "cand_good",
            "state-a",
            "state-b",
            InteractionOutcomeStatus.REJECTED,
            1.0,
            -1.0,
            2.0,
            "lineage-1",
            ExperimentResources(gpu_seconds=1.0),
        ),
        (_candidate("cand_next", "state-b", 0.8),),
        config=config,
    )
    assert result.action is InteractionAction.REPLAN
    assert result.plan.current_state_id == "state-b"
    assert len(result.plan.history) == 1
    assert result.plan.history[0].status is InteractionOutcomeStatus.REJECTED


def test_stale_outcome_and_zero_replan_failure_roll_back() -> None:
    plan = plan_interaction_acquisition("state-a", (_candidate("cand_one", "state-a", 1.0),))
    with pytest.raises(InteractionPlanningError, match="stale"):
        replan_after_outcome(
            plan,
            InteractionOutcome(
                "cand_one",
                "state-old",
                None,
                InteractionOutcomeStatus.ACCEPTED,
                1.0,
                1.0,
                0.0,
                "lineage-2",
                ExperimentResources(),
            ),
            (),
        )
    result = replan_after_outcome(
        plan,
        InteractionOutcome(
            "cand_one",
            "state-a",
            None,
            InteractionOutcomeStatus.FAILED,
            1.0,
            None,
            0.0,
            "lineage-3",
            ExperimentResources(),
        ),
        (),
        config=InteractionAcquisitionConfig(max_replans=0),
    )
    assert result.action is InteractionAction.ROLLBACK
    assert result.plan.state is InteractionPlanState.ROLLED_BACK

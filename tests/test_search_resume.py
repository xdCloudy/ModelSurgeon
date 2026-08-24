from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.config import ObjectiveDirection, ObjectiveNormalization, OptimizeMetric
from modelsurgeon.search.constraints import (
    BaselineReference,
    ConstraintMetric,
    ConstraintObservation,
    ConstraintSet,
    OptimizationConstraint,
)
from modelsurgeon.search.objectives import ObjectiveObservation, ObjectiveSet, ObjectiveTerm
from modelsurgeon.search.policies import (
    PredictedSearchCandidate,
    SearchPolicy,
    SearchPolicyConfig,
    SearchPolicyKind,
    SearchPolicyState,
)
from modelsurgeon.search.resume import (
    PendingSearchEvaluation,
    SearchBudgetSnapshot,
    SearchResumeError,
    SearchResumeSnapshot,
    SearchResumeStore,
    SearchRngState,
)


def _policy() -> SearchPolicy:
    objectives = ObjectiveSet(
        (
            ObjectiveTerm(
                OptimizeMetric.QUALITY,
                ObjectiveDirection.MAXIMIZE,
                normalization=ObjectiveNormalization.IDENTITY,
            ),
        )
    )
    constraints = ConstraintSet(
        (
            OptimizationConstraint(
                ConstraintMetric.QUALITY_RETENTION,
                0.9,
                BaselineReference.IMMUTABLE_SOURCE,
            ),
        )
    )
    return SearchPolicy(
        SearchPolicyConfig(SearchPolicyKind.GREEDY, evaluation_budget=3, seed=11),
        objectives,
        constraints,
    )


def _candidate(name: str, reward: float) -> PredictedSearchCandidate:
    return PredictedSearchCandidate(
        f"candidate_{name}",
        "state_parent",
        (ObjectiveObservation(OptimizeMetric.QUALITY, reward),),
        (
            ConstraintObservation(
                ConstraintMetric.QUALITY_RETENTION,
                1,
                BaselineReference.IMMUTABLE_SOURCE,
            ),
        ),
    )


def _snapshot(generation: int, state: SearchPolicyState) -> SearchResumeSnapshot:
    return SearchResumeSnapshot(
        "search_test",
        generation,
        state,
        SearchRngState(11, state.decision_index),
        ("checkpoint_root",),
        ("checkpoint_root",),
        SearchBudgetSnapshot(3, len(state.selected_candidate_ids), 1.25, 4096),
        (
            PendingSearchEvaluation(
                state.selected_candidate_ids[-1], "state_pending", "checkpoint_root"
            ),
        ),
        evidence_arrival_cursor=generation,
    )


def test_reboot_resume_produces_same_next_decision_for_fixed_evidence_order(
    tmp_path: Path,
) -> None:
    policy = _policy()
    candidates = (_candidate("a", 0.8), _candidate("b", 0.9), _candidate("c", 0.7))
    first = policy.select(candidates)
    uninterrupted = policy.select(candidates, first.next_state)
    path = tmp_path / "resume.sqlite3"
    with SearchResumeStore(path) as store:
        store.save(_snapshot(0, first.next_state), expected_generation=None)
    with SearchResumeStore(path) as rebooted:
        resumed = rebooted.load_latest("search_test")
        replayed = policy.select(candidates, resumed.policy_state)
        assert replayed.to_record() == uninterrupted.to_record()


def test_snapshot_generation_is_atomic_and_stale_writer_cannot_overwrite(
    tmp_path: Path,
) -> None:
    policy = _policy()
    first = policy.select((_candidate("a", 1),))
    initial = _snapshot(0, first.next_state)
    path = tmp_path / "resume.sqlite3"
    with SearchResumeStore(path) as store:
        store.save(initial, expected_generation=None)
        next_snapshot = replace(initial, generation=1, evidence_arrival_cursor=1)
        store.save(next_snapshot, expected_generation=0)
        with pytest.raises(SearchResumeError, match="stale"):
            store.save(replace(initial, generation=1), expected_generation=0)
        assert store.load_latest("search_test").generation == 1


def test_snapshot_rejects_frontier_or_budget_inconsistency() -> None:
    policy = _policy()
    state = policy.select((_candidate("a", 1),)).next_state
    with pytest.raises(SearchResumeError, match="subset"):
        SearchResumeSnapshot(
            "search_bad",
            0,
            state,
            SearchRngState(11, state.decision_index),
            ("checkpoint_missing",),
            ("checkpoint_root",),
            SearchBudgetSnapshot(3, 1),
            (),
            0,
        )
    with pytest.raises(SearchResumeError, match="reserved"):
        SearchResumeSnapshot(
            "search_bad",
            0,
            state,
            SearchRngState(11, state.decision_index),
            ("checkpoint_root",),
            ("checkpoint_root",),
            SearchBudgetSnapshot(3, 0),
            (),
            0,
        )

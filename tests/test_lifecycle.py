from pathlib import Path

import pytest

from modelsurgeon.search.lifecycle import (
    ArchitectureLifecycleCandidate,
    ArchitectureLifecycleError,
    ArchitectureSearchLifecycle,
    ArchitectureSearchResumeStore,
    EvaluationStatus,
    LifecycleEvidence,
    LifecycleStage,
)


def _candidate(name: str, *, predicted_only: bool = False) -> ArchitectureLifecycleCandidate:
    return ArchitectureLifecycleCandidate(
        f"architecture_candidate_{name}",
        f"architecture_{name}",
        f"equivalence_{name}",
        f"state_{name}",
        "checkpoint_root",
        predicted_only,
    )


def _evaluated(
    lifecycle: ArchitectureSearchLifecycle, candidate_id: str
) -> ArchitectureSearchLifecycle:
    return lifecycle.materialize(candidate_id).evaluate(
        candidate_id,
        LifecycleEvidence("evaluation_test", EvaluationStatus.MEASURED, True, "measured", 1),
    )


def test_lifecycle_is_resumable_and_next_decisions_are_deterministic(tmp_path: Path) -> None:
    lifecycle = ArchitectureSearchLifecycle.create(
        "search_test", (_candidate("b"), _candidate("a")), "checkpoint_root", 2
    )
    assert lifecycle.next_decision_ids(1) == ("architecture_candidate_a",)
    reserved = lifecycle.reserve(lifecycle.next_decision_ids(1))
    assert reserved.next_decision_ids(1) == ("architecture_candidate_b",)
    path = tmp_path / "lifecycle.sqlite3"
    with ArchitectureSearchResumeStore(path) as store:
        store.save(lifecycle.snapshot, expected_generation=None)
        store.save(reserved.snapshot, expected_generation=0)
    with ArchitectureSearchResumeStore(path) as store:
        rebooted = ArchitectureSearchLifecycle(store.load_latest("search_test"))
    assert rebooted.next_decision_ids(1) == reserved.next_decision_ids(1)


def test_only_measured_passing_evidence_can_reach_frontier() -> None:
    predicted = ArchitectureSearchLifecycle.create(
        "search_test", (_candidate("predicted", predicted_only=True),), "checkpoint_root", 1
    )
    evaluated = _evaluated(
        predicted.reserve(("architecture_candidate_predicted",)),
        "architecture_candidate_predicted",
    )
    with pytest.raises(ArchitectureLifecycleError, match="predicted-only"):
        evaluated.accept("architecture_candidate_predicted", "checkpoint_child")

    accepted = ArchitectureSearchLifecycle.create(
        "search_test", (_candidate("accepted"),), "checkpoint_root", 1
    )
    accepted = _evaluated(
        accepted.reserve(("architecture_candidate_accepted",)),
        "architecture_candidate_accepted",
    )
    accepted = accepted.accept("architecture_candidate_accepted", "checkpoint_child")
    assert accepted.snapshot.frontier_candidate_ids == ("architecture_candidate_accepted",)
    assert accepted.snapshot.candidates[0].stage is LifecycleStage.ACCEPTED


def test_duplicate_architecture_and_equivalent_path_are_charged_once() -> None:
    first = _candidate("first")
    duplicate = ArchitectureLifecycleCandidate(
        "architecture_candidate_duplicate",
        first.architecture_id,
        first.equivalence_id,
        "state_duplicate",
        "checkpoint_root",
        False,
    )
    lifecycle = ArchitectureSearchLifecycle.create(
        "search_test", (first, duplicate), "checkpoint_root", 2
    )
    reserved = lifecycle.reserve((first.candidate_id,))
    with pytest.raises(ArchitectureLifecycleError, match="charged twice"):
        reserved.reserve((duplicate.candidate_id,))
    assert reserved.snapshot.budget.evaluations_reserved == 1


def test_failure_and_rejection_are_terminal_and_persisted() -> None:
    lifecycle = ArchitectureSearchLifecycle.create(
        "search_test", (_candidate("failed"), _candidate("rejected")), "checkpoint_root", 2
    )
    failed = lifecycle.reserve(("architecture_candidate_failed",)).fail(
        "architecture_candidate_failed", "materialization crashed"
    )
    assert failed.snapshot.candidates[0].stage is LifecycleStage.FAILED
    rejected = lifecycle.reserve(("architecture_candidate_rejected",))
    rejected = _evaluated(rejected, "architecture_candidate_rejected").reject(
        "architecture_candidate_rejected"
    )
    assert rejected.snapshot.candidates[1].stage is LifecycleStage.REJECTED

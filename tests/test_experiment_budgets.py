from __future__ import annotations

from modelsurgeon.active_learning import (
    ExperimentBudget,
    ExperimentBudgetLedger,
    ExperimentResources,
    FailedAttemptBudgetPolicy,
)


def _budget(policy=FailedAttemptBudgetPolicy.RELEASE_UNUSED):
    return ExperimentBudget(2, 10.0, 4.0, 6.0, 100, policy)


def test_scheduler_stops_before_candidate_boundary_when_any_dimension_would_exceed() -> None:
    ledger = ExperimentBudgetLedger(_budget())
    first = ledger.reserve("cand_a", ExperimentResources(4.0, 1.0, 2.0, 40))
    assert first.reservation is not None
    ledger.complete(
        first.reservation,
        ExperimentResources(4.0, 1.0, 2.0, 40),
        succeeded=True,
    )

    stopped = ledger.reserve("cand_b", ExperimentResources(7.0, 1.0, 1.0, 10))

    assert not stopped.allowed and stopped.reservation is None
    assert stopped.exhausted_dimensions == ("wall-seconds",)
    assert ledger.snapshot().attempts == 1
    assert ledger.snapshot().active_candidate_id is None


def test_failed_attempt_release_unused_policy_charges_observed_use_and_count() -> None:
    ledger = ExperimentBudgetLedger(_budget())
    decision = ledger.reserve("cand_a", ExperimentResources(5.0, 2.0, 3.0, 50))
    assert decision.reservation is not None

    snapshot = ledger.complete(
        decision.reservation,
        ExperimentResources(1.0, 0.5, 0.25, 5),
        succeeded=False,
    )

    assert snapshot.failed == 1 and snapshot.attempts == 1
    assert snapshot.consumed == ExperimentResources(1.0, 0.5, 0.25, 5)
    assert snapshot.failed_attempt_policy is FailedAttemptBudgetPolicy.RELEASE_UNUSED


def test_failed_attempt_reserved_policy_charges_full_reservation() -> None:
    ledger = ExperimentBudgetLedger(_budget(FailedAttemptBudgetPolicy.CHARGE_RESERVED))
    decision = ledger.reserve("cand_a", ExperimentResources(5.0, 2.0, 3.0, 50))
    assert decision.reservation is not None

    snapshot = ledger.complete(
        decision.reservation,
        ExperimentResources(1.0, 0.5, 0.25, 5),
        succeeded=False,
    )

    assert snapshot.consumed == ExperimentResources(5.0, 2.0, 3.0, 50)
    assert snapshot.to_record()["failed_attempt_policy"] == "charge-full-reservation"

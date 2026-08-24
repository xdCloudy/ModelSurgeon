from pathlib import Path

import pytest

from modelsurgeon.search.constraints import (
    BaselineReference,
    ConstraintEvaluation,
    ConstraintMetric,
    ConstraintResult,
    OptimizationConstraint,
)
from modelsurgeon.search.lineage import (
    CheckpointLineageError,
    CheckpointLineageStore,
    LineageDecisionKind,
    MeasuredConstraintEvidence,
)
from modelsurgeon.surgery.contracts import TransactionState

CONSTRAINT = OptimizationConstraint(
    ConstraintMetric.QUALITY_RETENTION,
    0.95,
    BaselineReference.IMMUTABLE_SOURCE,
)
DIGEST = "sha256:" + "a" * 64


def _evidence(name: str, passed: bool) -> MeasuredConstraintEvidence:
    result = ConstraintResult(
        CONSTRAINT,
        0.99 if passed else 0.8,
        passed,
        None if passed else "threshold_violation",
    )
    return MeasuredConstraintEvidence(f"evaluation_{name}", ConstraintEvaluation(passed, (result,)))


class _Transaction:
    def __init__(self) -> None:
        self.state = TransactionState.APPLIED
        self.owns_mutable_inputs = True

    @property
    def transaction_id(self) -> str:
        return "transaction"

    def commit(self) -> None:
        self.state = TransactionState.COMMITTED
        self.owns_mutable_inputs = False

    def rollback(self) -> None:
        self.state = TransactionState.ROLLED_BACK
        self.owns_mutable_inputs = False


class _Lease:
    artifact_ids = ("candidate-temp", "candidate-log")

    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def test_keep_creates_one_parent_checkpoint_with_measured_evidence(tmp_path: Path) -> None:
    path = tmp_path / "lineage.sqlite3"
    with CheckpointLineageStore(path) as store:
        store.register_root("checkpoint_root", "state_root", DIGEST, _evidence("root", True))
        transaction = _Transaction()
        lease = _Lease()
        decision = store.decide(
            parent_checkpoint_id="checkpoint_root",
            candidate_id="candidate_safe",
            candidate_state_id="state_safe",
            artifact_digest=DIGEST,
            evidence=_evidence("safe", True),
            transaction=transaction,
            artifact_lease=lease,
        )
        assert decision.kind is LineageDecisionKind.KEEP
        assert decision.checkpoint_id is not None
        child = store.require_search_root(decision.checkpoint_id)
        assert child.parent_checkpoint_id == "checkpoint_root"
        assert child.evaluation_id == "evaluation_safe"
        assert transaction.state is TransactionState.COMMITTED
        assert lease.released is False

    with CheckpointLineageStore(path) as resumed:
        assert len(resumed.checkpoints()) == 2


def test_failed_constraints_rollback_release_and_never_create_root(tmp_path: Path) -> None:
    with CheckpointLineageStore(tmp_path / "lineage.sqlite3") as store:
        store.register_root("checkpoint_root", "state_root", DIGEST, _evidence("root", True))
        transaction = _Transaction()
        lease = _Lease()
        decision = store.decide(
            parent_checkpoint_id="checkpoint_root",
            candidate_id="candidate_unsafe",
            candidate_state_id="state_unsafe",
            artifact_digest=DIGEST,
            evidence=_evidence("unsafe", False),
            transaction=transaction,
            artifact_lease=lease,
        )
        assert decision.kind is LineageDecisionKind.ROLLBACK
        assert decision.released_artifact_ids == lease.artifact_ids
        assert transaction.state is TransactionState.ROLLED_BACK
        assert lease.released is True
        with pytest.raises(CheckpointLineageError, match="accepted"):
            store.require_search_root("checkpoint_candidate_unsafe")
        with pytest.raises(CheckpointLineageError, match="accepted"):
            store.decide(
                parent_checkpoint_id="checkpoint_candidate_unsafe",
                candidate_id="candidate_descendant",
                candidate_state_id="state_descendant",
                artifact_digest=DIGEST,
                evidence=_evidence("descendant", True),
                transaction=_Transaction(),
                artifact_lease=_Lease(),
            )


def test_invalid_parent_fails_before_touching_transaction_or_artifacts(tmp_path: Path) -> None:
    with CheckpointLineageStore(tmp_path / "lineage.sqlite3") as store:
        transaction = _Transaction()
        lease = _Lease()
        with pytest.raises(CheckpointLineageError, match="accepted"):
            store.decide(
                parent_checkpoint_id="checkpoint_missing",
                candidate_id="candidate_x",
                candidate_state_id="state_x",
                artifact_digest=DIGEST,
                evidence=_evidence("x", True),
                transaction=transaction,
                artifact_lease=lease,
            )
        assert transaction.state is TransactionState.APPLIED
        assert lease.released is False

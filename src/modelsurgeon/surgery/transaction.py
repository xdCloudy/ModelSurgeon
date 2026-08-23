"""Exclusive in-memory mutation transactions with bounded snapshots and rollback."""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from typing import Protocol, Self

from modelsurgeon.surgery.contracts import MutationContractError, TransactionState


class MutationSnapshotTarget(Protocol):
    """One mutable state target that owns its snapshot and restore representation."""

    def snapshot(self) -> object: ...

    def restore(self, snapshot: object) -> None: ...


class MutationTransactionError(MutationContractError):
    """Raised when exclusive transactional mutation invariants are violated."""


_ACTIVE_LOCK = threading.Lock()
_ACTIVE_OWNER_IDS: set[int] = set()


class InMemoryMutationTransaction(AbstractContextManager["InMemoryMutationTransaction"]):
    """Snapshot only declared changed targets and roll them back unless committed.

    Ownership is exclusive per ``owner`` object. Entering a nested or concurrent
    transaction for the same owner fails before any snapshot is taken. Snapshot
    representation is delegated to each target so tensor/framework adapters can
    preserve exact bytes without this core module importing a tensor framework.
    """

    def __init__(
        self,
        owner: object,
        targets: Mapping[str, MutationSnapshotTarget],
        changed_keys: Iterable[str],
    ) -> None:
        ordered = tuple(changed_keys)
        if not ordered or any(not key for key in ordered):
            raise MutationTransactionError("transaction requires non-empty changed target keys")
        if len(ordered) != len(set(ordered)):
            raise MutationTransactionError("changed target keys must be unique")
        canonical = tuple(sorted(ordered))
        missing = tuple(key for key in canonical if key not in targets)
        if missing:
            raise MutationTransactionError(
                "unknown changed targets: " + ", ".join(missing)
            )
        self._owner = owner
        self._owner_id = id(owner)
        self._targets = targets
        self.changed_keys = canonical
        identity = f"{self._owner_id}:{','.join(canonical)}:{uuid.uuid4().hex}"
        self._transaction_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        self._state = TransactionState.PREPARED
        self._snapshots: dict[str, object] = {}
        self._owns_mutable_inputs = False
        self._entered = False

    @property
    def transaction_id(self) -> str:
        return self._transaction_id

    @property
    def state(self) -> TransactionState:
        return self._state

    @property
    def owns_mutable_inputs(self) -> bool:
        return self._owns_mutable_inputs

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    def __enter__(self) -> Self:
        if self._entered:
            raise MutationTransactionError("transaction cannot be entered twice")
        with _ACTIVE_LOCK:
            if self._owner_id in _ACTIVE_OWNER_IDS:
                raise MutationTransactionError(
                    "nested or concurrent mutation transaction for the same owner is forbidden"
                )
            _ACTIVE_OWNER_IDS.add(self._owner_id)
        self._entered = True
        self._owns_mutable_inputs = True
        try:
            for key in self.changed_keys:
                self._snapshots[key] = self._targets[key].snapshot()
        except BaseException:
            self._release_ownership()
            self._snapshots.clear()
            raise
        return self

    def mark_applied(self) -> None:
        if not self._owns_mutable_inputs or self._state is not TransactionState.PREPARED:
            raise MutationTransactionError("only an active prepared transaction can be marked applied")
        self._state = TransactionState.APPLIED

    def commit(self) -> None:
        if not self._owns_mutable_inputs:
            raise MutationTransactionError("cannot commit a transaction without mutable ownership")
        if self._state not in {TransactionState.PREPARED, TransactionState.APPLIED}:
            raise MutationTransactionError(f"cannot commit transaction in state {self._state.value}")
        self._state = TransactionState.COMMITTED
        self._snapshots.clear()
        self._release_ownership()

    def rollback(self) -> None:
        if self._state is TransactionState.ROLLED_BACK:
            return
        if self._state is TransactionState.COMMITTED:
            raise MutationTransactionError("committed transactions cannot be rolled back")
        if not self._owns_mutable_inputs:
            raise MutationTransactionError("cannot roll back a transaction without mutable ownership")
        errors: list[BaseException] = []
        for key in reversed(self.changed_keys):
            if key not in self._snapshots:
                continue
            try:
                self._targets[key].restore(self._snapshots[key])
            except BaseException as error:
                errors.append(error)
        self._snapshots.clear()
        self._state = TransactionState.ROLLED_BACK
        self._release_ownership()
        if errors:
            raise BaseExceptionGroup("one or more mutation targets failed to restore", errors)

    def _release_ownership(self) -> None:
        if self._owns_mutable_inputs:
            with _ACTIVE_LOCK:
                _ACTIVE_OWNER_IDS.discard(self._owner_id)
            self._owns_mutable_inputs = False

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        if self._state in {TransactionState.PREPARED, TransactionState.APPLIED}:
            self.rollback()
        else:
            self._release_ownership()

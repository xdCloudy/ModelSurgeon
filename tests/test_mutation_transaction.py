"""Tests for exclusive in-memory mutation transaction ownership and rollback."""

from __future__ import annotations

import threading

import pytest

from modelsurgeon.surgery.transaction import (
    InMemoryMutationTransaction,
    MutationTransactionError,
)


class ByteTensorTarget:
    def __init__(self, payload: bytes) -> None:
        self.payload = bytearray(payload)
        self.snapshot_calls = 0
        self.restore_calls = 0

    def snapshot(self) -> object:
        self.snapshot_calls += 1
        return bytes(self.payload)

    def restore(self, snapshot: object) -> None:
        if not isinstance(snapshot, bytes):
            raise TypeError("byte tensor snapshots must be bytes")
        self.restore_calls += 1
        self.payload[:] = snapshot


def test_exception_rolls_back_only_declared_targets_byte_exactly() -> None:
    owner = object()
    changed = ByteTensorTarget(b"abcdef")
    untouched = ByteTensorTarget(b"uvwxyz")
    transaction = InMemoryMutationTransaction(
        owner,
        {"changed": changed, "untouched": untouched},
        ("changed",),
    )

    with pytest.raises(RuntimeError, match="boom"), transaction:
        changed.payload[:] = b"XXXXXX"
        transaction.mark_applied()
        raise RuntimeError("boom")

    assert bytes(changed.payload) == b"abcdef"
    assert bytes(untouched.payload) == b"uvwxyz"
    assert changed.snapshot_calls == 1
    assert changed.restore_calls == 1
    assert untouched.snapshot_calls == 0
    assert transaction.snapshot_count == 0
    assert transaction.state.value == "rolled_back"
    assert not transaction.owns_mutable_inputs


def test_commit_keeps_mutation_and_releases_snapshots() -> None:
    owner = object()
    target = ByteTensorTarget(b"before")
    transaction = InMemoryMutationTransaction(owner, {"weight": target}, ("weight",))

    with transaction:
        target.payload[:] = b"after!"
        transaction.mark_applied()
        transaction.commit()

    assert bytes(target.payload) == b"after!"
    assert target.restore_calls == 0
    assert transaction.snapshot_count == 0
    assert transaction.state.value == "committed"
    assert not transaction.owns_mutable_inputs


def test_uncommitted_normal_exit_is_explicit_rejection_and_rolls_back() -> None:
    owner = object()
    target = ByteTensorTarget(b"stable")

    with InMemoryMutationTransaction(owner, {"weight": target}, ("weight",)) as transaction:
        target.payload[:] = b"reject"
        transaction.mark_applied()

    assert bytes(target.payload) == b"stable"
    assert transaction.state.value == "rolled_back"


def test_keyboard_interrupt_rolls_back() -> None:
    owner = object()
    target = ByteTensorTarget(b"stable")
    transaction = InMemoryMutationTransaction(owner, {"weight": target}, ("weight",))

    with pytest.raises(KeyboardInterrupt), transaction:
        target.payload[:] = b"broken"
        transaction.mark_applied()
        raise KeyboardInterrupt

    assert bytes(target.payload) == b"stable"


def test_nested_transaction_for_same_owner_fails_before_snapshot() -> None:
    owner = object()
    first = ByteTensorTarget(b"first")
    second = ByteTensorTarget(b"second")

    with InMemoryMutationTransaction(owner, {"first": first}, ("first",)):
        nested = InMemoryMutationTransaction(owner, {"second": second}, ("second",))
        with pytest.raises(MutationTransactionError, match="nested or concurrent"):
            nested.__enter__()

    assert second.snapshot_calls == 0


def test_concurrent_transaction_for_same_owner_fails_safely() -> None:
    owner = object()
    first = ByteTensorTarget(b"first")
    second = ByteTensorTarget(b"second")
    entered = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []

    def hold_transaction() -> None:
        try:
            with InMemoryMutationTransaction(owner, {"first": first}, ("first",)):
                entered.set()
                release.wait(timeout=2.0)
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=hold_transaction)
    worker.start()
    assert entered.wait(timeout=2.0)
    try:
        competing = InMemoryMutationTransaction(owner, {"second": second}, ("second",))
        with pytest.raises(MutationTransactionError, match="nested or concurrent"):
            competing.__enter__()
    finally:
        release.set()
        worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert failures == []
    assert second.snapshot_calls == 0


def test_snapshot_failure_releases_exclusive_owner_lease() -> None:
    class FailingTarget(ByteTensorTarget):
        def snapshot(self) -> object:
            raise RuntimeError("snapshot failed")

    owner = object()
    failing = FailingTarget(b"x")
    with pytest.raises(RuntimeError, match="snapshot failed"):
        InMemoryMutationTransaction(owner, {"weight": failing}, ("weight",)).__enter__()

    healthy = ByteTensorTarget(b"ok")
    with InMemoryMutationTransaction(owner, {"weight": healthy}, ("weight",)):
        pass

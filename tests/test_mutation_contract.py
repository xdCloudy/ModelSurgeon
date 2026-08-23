"""Tests for deterministic and ownership-safe mutation contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery import (
    MutationCompatibility,
    MutationContractError,
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationPrecondition,
    MutationRequest,
    TransactionState,
    require_safe_transaction,
)


def _request() -> MutationRequest:
    return MutationRequest(
        MutationKind.REMOVE,
        (ComponentId.parse("model.layers.1.mlp.up_proj"),),
        (("channels", 64), ("reason", "candidate")),
    )


def test_request_identity_is_deterministic_serializable_and_immutable() -> None:
    first = _request()
    second = _request()
    assert first.to_record() == second.to_record()
    assert first.mutation_id == second.mutation_id
    assert len(first.mutation_id) == 64
    with pytest.raises(FrozenInstanceError):
        first.kind = MutationKind.MASK  # type: ignore[misc]


def test_plan_requires_targets_complete_canonical_and_preconditioned() -> None:
    request = _request()
    plan = MutationPlan(
        request,
        request.targets,
        (MutationPrecondition("revision", "abc123"),),
        MutationDelta(parameters=-64, storage_bytes=-128),
    )
    assert plan.expected_delta.parameters == -64
    with pytest.raises(MutationContractError, match="target must be affected"):
        MutationPlan(request, (), (), MutationDelta())


def test_unsupported_compatibility_and_unowned_transactions_fail_closed() -> None:
    with pytest.raises(MutationContractError, match="not representable"):
        MutationCompatibility(False, "not representable").require_supported()

    class Transaction:
        transaction_id = "tx-1"
        state = TransactionState.PREPARED
        owns_mutable_inputs = False

        def commit(self) -> None: ...

        def rollback(self) -> None: ...

    with pytest.raises(MutationContractError, match="owns mutable inputs"):
        require_safe_transaction(Transaction())


def test_noncanonical_requests_are_rejected_before_execution() -> None:
    left = ComponentId.parse("model.layers.2")
    right = ComponentId.parse("model.layers.1")
    with pytest.raises(MutationContractError, match="canonical"):
        MutationRequest(MutationKind.REMOVE, (left, right))
    with pytest.raises(MutationContractError, match="canonical keys"):
        MutationRequest(MutationKind.REMOVE, (right,), (("z", 1), ("a", 2)))

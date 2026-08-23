"""Deterministic, framework-neutral transactional mutation contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from modelsurgeon.graph import ComponentId

MUTATION_SCHEMA_VERSION: Literal[1] = 1
type MutationPrimitive = str | int | float | bool | None


class MutationContractError(ValueError):
    """Raised before mutation when a contract is incomplete or unsafe."""


class MutationKind(StrEnum):
    MASK = "mask"
    REMOVE = "remove"
    LOW_RANK = "low_rank"
    REQUANTIZE = "requantize"


class TransactionState(StrEnum):
    PREPARED = "prepared"
    APPLIED = "applied"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class MutationCompatibility:
    supported: bool
    reason: str
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.reason or len(self.constraints) != len(set(self.constraints)):
            raise MutationContractError(
                "compatibility requires a reason and unique constraint keys"
            )

    def require_supported(self) -> None:
        if not self.supported:
            raise MutationContractError(self.reason)


@dataclass(frozen=True, slots=True)
class MutationPrecondition:
    key: str
    expected: MutationPrimitive

    def __post_init__(self) -> None:
        if not self.key:
            raise MutationContractError("precondition keys must be non-empty")


@dataclass(frozen=True, slots=True)
class MutationDelta:
    parameters: int = 0
    flops: int = 0
    memory_bytes: int = 0
    storage_bytes: int = 0


@dataclass(frozen=True, slots=True)
class MutationRequest:
    kind: MutationKind
    targets: tuple[ComponentId, ...]
    parameters: tuple[tuple[str, MutationPrimitive], ...] = ()
    schema_version: Literal[1] = MUTATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MUTATION_SCHEMA_VERSION:
            raise MutationContractError(
                f"unsupported mutation schema version {self.schema_version}"
            )
        if not self.targets or tuple(sorted(self.targets)) != self.targets:
            raise MutationContractError("mutation targets must be non-empty and canonical")
        if len(self.targets) != len(set(self.targets)):
            raise MutationContractError("mutation targets must be unique")
        keys = [key for key, _ in self.parameters]
        if any(not key for key in keys) or keys != sorted(keys) or len(keys) != len(set(keys)):
            raise MutationContractError("mutation parameters must have unique canonical keys")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "targets": [str(target) for target in self.targets],
            "parameters": dict(self.parameters),
        }

    @property
    def mutation_id(self) -> str:
        encoded = json.dumps(
            self.to_record(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class MutationPlan:
    request: MutationRequest
    affected_components: tuple[ComponentId, ...]
    preconditions: tuple[MutationPrecondition, ...]
    expected_delta: MutationDelta

    def __post_init__(self) -> None:
        if tuple(sorted(self.affected_components)) != self.affected_components:
            raise MutationContractError("affected components must use canonical ordering")
        if len(self.affected_components) != len(set(self.affected_components)):
            raise MutationContractError("affected components must be unique")
        if not set(self.request.targets).issubset(self.affected_components):
            raise MutationContractError("every requested target must be affected")
        keys = [item.key for item in self.preconditions]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise MutationContractError("preconditions must have unique canonical keys")


@runtime_checkable
class MutationTransaction(Protocol):
    """Safe transaction ownership supplied by the format-specific engine."""

    @property
    def transaction_id(self) -> str: ...

    @property
    def state(self) -> TransactionState: ...

    @property
    def owns_mutable_inputs(self) -> bool: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


class TransactionalMutation(Protocol):
    """Implementation interface; applying is legal only inside an owning transaction."""

    def compatibility(self, request: MutationRequest) -> MutationCompatibility: ...

    def plan(self, request: MutationRequest) -> MutationPlan: ...

    def apply(self, plan: MutationPlan, transaction: MutationTransaction) -> None: ...

    def rollback(self, plan: MutationPlan, transaction: MutationTransaction) -> None: ...


def require_safe_transaction(transaction: MutationTransaction) -> None:
    if not transaction.owns_mutable_inputs or transaction.state is not TransactionState.PREPARED:
        raise MutationContractError(
            "mutation apply requires a prepared transaction that owns mutable inputs"
        )

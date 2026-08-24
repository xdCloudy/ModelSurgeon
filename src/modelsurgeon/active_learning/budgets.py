"""Candidate-boundary resource budgets for active-learning evaluations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

EXPERIMENT_BUDGET_SCHEMA_VERSION: Final[int] = 1


class ExperimentBudgetError(ValueError):
    """Raised when budget reservations or observations are invalid."""


class FailedAttemptBudgetPolicy(StrEnum):
    RELEASE_UNUSED = "release-unused-charge-observed"
    CHARGE_RESERVED = "charge-full-reservation"


@dataclass(frozen=True, slots=True)
class ExperimentResources:
    wall_seconds: float = 0.0
    tier_cost: float = 0.0
    gpu_seconds: float = 0.0
    disk_bytes: int = 0

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.wall_seconds, self.tier_cost, self.gpu_seconds)
        ):
            raise ExperimentBudgetError(
                "experiment resource values must be finite and non-negative"
            )
        if isinstance(self.disk_bytes, bool) or self.disk_bytes < 0:
            raise ExperimentBudgetError("experiment disk bytes must be non-negative")

    def to_record(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "tier_cost": self.tier_cost,
            "gpu_seconds": self.gpu_seconds,
            "disk_bytes": self.disk_bytes,
        }


@dataclass(frozen=True, slots=True)
class ExperimentBudget:
    max_attempts: int
    max_wall_seconds: float
    max_tier_cost: float
    max_gpu_seconds: float
    max_disk_bytes: int
    failed_attempt_policy: FailedAttemptBudgetPolicy = FailedAttemptBudgetPolicy.RELEASE_UNUSED

    def __post_init__(self) -> None:
        if self.max_attempts < 0 or isinstance(self.max_attempts, bool):
            raise ExperimentBudgetError("experiment attempt budget cannot be negative")
        ExperimentResources(
            self.max_wall_seconds,
            self.max_tier_cost,
            self.max_gpu_seconds,
            self.max_disk_bytes,
        )


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    candidate_id: str
    sequence: int
    resources: ExperimentResources


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    allowed: bool
    reservation: BudgetReservation | None
    exhausted_dimensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentBudgetSnapshot:
    attempts: int
    succeeded: int
    failed: int
    consumed: ExperimentResources
    active_candidate_id: str | None
    failed_attempt_policy: FailedAttemptBudgetPolicy
    schema_version: int = EXPERIMENT_BUDGET_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "attempts": self.attempts,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "consumed": self.consumed.to_record(),
            "active_candidate_id": self.active_candidate_id,
            "failed_attempt_policy": self.failed_attempt_policy.value,
        }


class ExperimentBudgetLedger:
    """Reserve and commit one evaluation at a time at clean candidate boundaries."""

    def __init__(self, budget: ExperimentBudget) -> None:
        self.budget = budget
        self._attempts = 0
        self._succeeded = 0
        self._failed = 0
        self._consumed = ExperimentResources()
        self._active: BudgetReservation | None = None
        self._sequence = 0

    def reserve(self, candidate_id: str, estimate: ExperimentResources) -> BudgetDecision:
        if self._active is not None:
            raise ExperimentBudgetError("complete the active candidate before reserving another")
        if not candidate_id.startswith("cand_"):
            raise ExperimentBudgetError("budget reservations require canonical candidate IDs")
        exhausted = self._exhausted(estimate)
        if exhausted:
            return BudgetDecision(False, None, exhausted)
        self._sequence += 1
        self._active = BudgetReservation(candidate_id, self._sequence, estimate)
        return BudgetDecision(True, self._active, ())

    def complete(
        self,
        reservation: BudgetReservation,
        observed: ExperimentResources,
        *,
        succeeded: bool,
    ) -> ExperimentBudgetSnapshot:
        if reservation != self._active:
            raise ExperimentBudgetError("budget completion does not match the active reservation")
        charged = observed
        if (
            not succeeded
            and self.budget.failed_attempt_policy is FailedAttemptBudgetPolicy.CHARGE_RESERVED
        ):
            charged = ExperimentResources(
                max(observed.wall_seconds, reservation.resources.wall_seconds),
                max(observed.tier_cost, reservation.resources.tier_cost),
                max(observed.gpu_seconds, reservation.resources.gpu_seconds),
                max(observed.disk_bytes, reservation.resources.disk_bytes),
            )
        self._attempts += 1
        self._succeeded += int(succeeded)
        self._failed += int(not succeeded)
        self._consumed = _add(self._consumed, charged)
        self._active = None
        return self.snapshot()

    def snapshot(self) -> ExperimentBudgetSnapshot:
        return ExperimentBudgetSnapshot(
            self._attempts,
            self._succeeded,
            self._failed,
            self._consumed,
            None if self._active is None else self._active.candidate_id,
            self.budget.failed_attempt_policy,
        )

    def _exhausted(self, estimate: ExperimentResources) -> tuple[str, ...]:
        dimensions: list[str] = []
        if self._attempts + 1 > self.budget.max_attempts:
            dimensions.append("attempt-count")
        if self._consumed.wall_seconds + estimate.wall_seconds > self.budget.max_wall_seconds:
            dimensions.append("wall-seconds")
        if self._consumed.tier_cost + estimate.tier_cost > self.budget.max_tier_cost:
            dimensions.append("tier-cost")
        if self._consumed.gpu_seconds + estimate.gpu_seconds > self.budget.max_gpu_seconds:
            dimensions.append("gpu-seconds")
        if self._consumed.disk_bytes + estimate.disk_bytes > self.budget.max_disk_bytes:
            dimensions.append("disk-bytes")
        return tuple(dimensions)


def _add(left: ExperimentResources, right: ExperimentResources) -> ExperimentResources:
    return ExperimentResources(
        left.wall_seconds + right.wall_seconds,
        left.tier_cost + right.tier_cost,
        left.gpu_seconds + right.gpu_seconds,
        left.disk_bytes + right.disk_bytes,
    )

"""Per-stage RAM, VRAM, disk, and runtime budget enforcement."""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from modelsurgeon.experiments.hardware import HardwareInventory
from modelsurgeon.experiments.state_machine import (
    CandidateState,
    CandidateWorkStage,
    ExperimentStateError,
    ExperimentStateMachine,
)
from modelsurgeon.instrumentation.memory_telemetry import (
    CudaMemoryProvider,
    process_rss_bytes,
)


class ResourceBudgetError(ValueError):
    """Raised when a resource budget cannot be validated or observed safely."""


class ResourceKind(StrEnum):
    RAM = "ram"
    VRAM = "vram"
    DISK = "disk"
    RUNTIME = "runtime"


@dataclass(frozen=True, slots=True)
class StageResourceBudget:
    max_ram_bytes: int | None = None
    max_vram_bytes: int | None = None
    max_disk_bytes: int | None = None
    max_runtime_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("max_ram_bytes", self.max_ram_bytes),
            ("max_vram_bytes", self.max_vram_bytes),
            ("max_disk_bytes", self.max_disk_bytes),
        ):
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise ResourceBudgetError(f"{name} must be positive when set")
        if self.max_runtime_seconds is not None and self.max_runtime_seconds <= 0:
            raise ResourceBudgetError("max_runtime_seconds must be positive when set")


@dataclass(frozen=True, slots=True)
class StageResourceEstimate:
    ram_bytes: int = 0
    vram_bytes: int = 0
    disk_bytes: int = 0
    runtime_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("ram_bytes", self.ram_bytes),
            ("vram_bytes", self.vram_bytes),
            ("disk_bytes", self.disk_bytes),
        ):
            if isinstance(value, bool) or value < 0:
                raise ResourceBudgetError(f"{name} must be non-negative")
        if self.runtime_seconds is not None and self.runtime_seconds < 0:
            raise ResourceBudgetError("runtime_seconds must be non-negative when known")


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    rss_bytes: int | None
    vram_bytes: int | None
    disk_free_bytes: int
    monotonic_seconds: float

    def __post_init__(self) -> None:
        for value in (self.rss_bytes, self.vram_bytes):
            if value is not None and value < 0:
                raise ResourceBudgetError("resource snapshot memory values cannot be negative")
        if self.disk_free_bytes < 0 or self.monotonic_seconds < 0:
            raise ResourceBudgetError("resource snapshot disk/time values cannot be negative")


@dataclass(frozen=True, slots=True)
class ResourceBudgetViolation:
    stage: str
    resource: ResourceKind
    limit: float
    observed: float

    def to_record(self) -> dict[str, str | float]:
        return {
            "stage": self.stage,
            "resource": self.resource.value,
            "limit": self.limit,
            "observed": self.observed,
        }


class ResourceBudgetExceeded(RuntimeError):
    """Explicit resource-exhausted outcome for a stage ceiling violation."""

    def __init__(self, violation: ResourceBudgetViolation) -> None:
        self.violation = violation
        super().__init__(
            f"{violation.stage}: {violation.resource.value} budget exceeded "
            f"({violation.observed:g} > {violation.limit:g})"
        )


class ResourceProbe(Protocol):
    def snapshot(self) -> ResourceSnapshot: ...


class HostResourceProbe:
    """Best-effort host process/disk probe with optional CUDA allocation provider."""

    def __init__(
        self,
        disk_path: str | Path,
        *,
        cuda: CudaMemoryProvider | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.disk_path = Path(disk_path)
        self.cuda = cuda
        self.monotonic = monotonic

    def snapshot(self) -> ResourceSnapshot:
        vram = None
        if self.cuda is not None:
            vram = max(self.cuda.allocated_bytes(), self.cuda.reserved_bytes())
        return ResourceSnapshot(
            process_rss_bytes(),
            vram,
            shutil.disk_usage(self.disk_path).free,
            self.monotonic(),
        )


def _raise_preflight(stage: str, resource: ResourceKind, limit: float, observed: float) -> None:
    raise ResourceBudgetExceeded(ResourceBudgetViolation(stage, resource, limit, observed))


def preflight_resource_budget(
    stage: str,
    budget: StageResourceBudget,
    inventory: HardwareInventory,
    estimate: StageResourceEstimate,
) -> None:
    """Reject estimates that exceed configured ceilings or known host capacity."""

    if not stage:
        raise ResourceBudgetError("resource budget stage name is required")
    if budget.max_ram_bytes is not None and estimate.ram_bytes > budget.max_ram_bytes:
        _raise_preflight(stage, ResourceKind.RAM, budget.max_ram_bytes, estimate.ram_bytes)
    available_ram = inventory.memory.available_bytes
    if available_ram is not None and estimate.ram_bytes > available_ram:
        _raise_preflight(stage, ResourceKind.RAM, available_ram, estimate.ram_bytes)

    if estimate.vram_bytes > 0 or budget.max_vram_bytes is not None:
        if not inventory.cuda.available:
            raise ResourceBudgetError("VRAM budget requested on a CPU-only hardware inventory")
        if budget.max_vram_bytes is not None and estimate.vram_bytes > budget.max_vram_bytes:
            _raise_preflight(stage, ResourceKind.VRAM, budget.max_vram_bytes, estimate.vram_bytes)
        known_totals = tuple(
            device.total_memory_bytes
            for device in inventory.cuda.devices
            if device.total_memory_bytes is not None
        )
        if known_totals:
            total_vram = sum(known_totals)
            if estimate.vram_bytes > total_vram:
                _raise_preflight(stage, ResourceKind.VRAM, total_vram, estimate.vram_bytes)

    if budget.max_disk_bytes is not None and estimate.disk_bytes > budget.max_disk_bytes:
        _raise_preflight(stage, ResourceKind.DISK, budget.max_disk_bytes, estimate.disk_bytes)
    if estimate.disk_bytes > inventory.disk.free_bytes:
        _raise_preflight(stage, ResourceKind.DISK, inventory.disk.free_bytes, estimate.disk_bytes)

    if (
        budget.max_runtime_seconds is not None
        and estimate.runtime_seconds is not None
        and estimate.runtime_seconds > budget.max_runtime_seconds
    ):
        _raise_preflight(
            stage,
            ResourceKind.RUNTIME,
            budget.max_runtime_seconds,
            estimate.runtime_seconds,
        )


class StageResourceBudgetGuard:
    """Track stage-local resource deltas and raise immediately at cooperative checkpoints."""

    def __init__(
        self,
        stage: str,
        budget: StageResourceBudget,
        inventory: HardwareInventory,
        estimate: StageResourceEstimate,
        probe: ResourceProbe,
    ) -> None:
        if not stage:
            raise ResourceBudgetError("resource budget stage name is required")
        self.stage = stage
        self.budget = budget
        self.inventory = inventory
        self.estimate = estimate
        self.probe = probe
        self._baseline: ResourceSnapshot | None = None

    def __enter__(self) -> StageResourceBudgetGuard:
        preflight_resource_budget(self.stage, self.budget, self.inventory, self.estimate)
        baseline = self.probe.snapshot()
        if self.budget.max_ram_bytes is not None and baseline.rss_bytes is None:
            raise ResourceBudgetError("RAM budget cannot be monitored on this host")
        if self.budget.max_vram_bytes is not None and baseline.vram_bytes is None:
            raise ResourceBudgetError("VRAM budget requires a live CUDA memory provider")
        self._baseline = baseline
        return self

    def _require_baseline(self) -> ResourceSnapshot:
        if self._baseline is None:
            raise ResourceBudgetError("resource budget guard is not active")
        return self._baseline

    def check(self) -> ResourceSnapshot:
        baseline = self._require_baseline()
        current = self.probe.snapshot()
        if current.monotonic_seconds < baseline.monotonic_seconds:
            raise ResourceBudgetError("resource probe monotonic clock moved backwards")

        if self.budget.max_ram_bytes is not None:
            if current.rss_bytes is None or baseline.rss_bytes is None:
                raise ResourceBudgetError("RAM budget became unobservable")
            ram_used = max(0, current.rss_bytes - baseline.rss_bytes)
            if ram_used > self.budget.max_ram_bytes:
                _raise_preflight(self.stage, ResourceKind.RAM, self.budget.max_ram_bytes, ram_used)

        if self.budget.max_vram_bytes is not None:
            if current.vram_bytes is None or baseline.vram_bytes is None:
                raise ResourceBudgetError("VRAM budget became unobservable")
            vram_used = max(0, current.vram_bytes - baseline.vram_bytes)
            if vram_used > self.budget.max_vram_bytes:
                _raise_preflight(
                    self.stage,
                    ResourceKind.VRAM,
                    self.budget.max_vram_bytes,
                    vram_used,
                )

        if self.budget.max_disk_bytes is not None:
            disk_used = max(0, baseline.disk_free_bytes - current.disk_free_bytes)
            if disk_used > self.budget.max_disk_bytes:
                _raise_preflight(
                    self.stage,
                    ResourceKind.DISK,
                    self.budget.max_disk_bytes,
                    disk_used,
                )

        if self.budget.max_runtime_seconds is not None:
            elapsed = current.monotonic_seconds - baseline.monotonic_seconds
            if elapsed > self.budget.max_runtime_seconds:
                _raise_preflight(
                    self.stage,
                    ResourceKind.RUNTIME,
                    self.budget.max_runtime_seconds,
                    elapsed,
                )
        return current

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_value, traceback
        if exc_type is None:
            self.check()
        self._baseline = None


def _expected_state(stage: CandidateWorkStage) -> CandidateState:
    if stage is CandidateWorkStage.MUTATION:
        return CandidateState.RUNNING
    return CandidateState.EVALUATING


def run_budgeted_stage[T](
    state_machine: ExperimentStateMachine,
    candidate_id: str,
    work_stage: CandidateWorkStage,
    guard: StageResourceBudgetGuard,
    operation: Callable[[StageResourceBudgetGuard], T],
) -> T:
    """Execute one active stage and persist resource exhaustion as resumable interruption."""

    expected = _expected_state(work_stage)
    current = state_machine.current(candidate_id)
    if current is not expected:
        actual = "<none>" if current is None else current.value
        raise ExperimentStateError(
            f"budgeted {work_stage.value} stage requires {expected.value} state, found {actual}"
        )
    try:
        with guard:
            return operation(guard)
    except ResourceBudgetExceeded as error:
        detail = (
            f"resource-exhausted:{error.violation.resource.value}:"
            f"{error.violation.observed:g}>{error.violation.limit:g}"
        )
        state_machine.transition(candidate_id, CandidateState.INTERRUPTED, detail)
        raise

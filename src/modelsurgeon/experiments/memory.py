"""Deterministic full, tensor, and streaming memory-mode planning."""

from __future__ import annotations

from dataclasses import dataclass

from modelsurgeon.config import MemoryMode


class MemoryPlanningError(ValueError):
    """Raised when resource estimates cannot satisfy the requested mode."""


def _non_negative(name: str, value: int) -> None:
    if isinstance(value, bool) or value < 0:
        raise MemoryPlanningError(f"{name} must be a non-negative byte count")


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Estimated peak resources for one execution mode."""

    peak_ram_bytes: int
    peak_vram_bytes: int
    scratch_bytes: int

    def __post_init__(self) -> None:
        _non_negative("peak RAM", self.peak_ram_bytes)
        _non_negative("peak VRAM", self.peak_vram_bytes)
        _non_negative("scratch", self.scratch_bytes)


@dataclass(frozen=True, slots=True)
class OperationMemoryEstimates:
    """Operation-specific estimates for every executable memory mode."""

    full: ResourceEstimate
    tensor: ResourceEstimate
    streaming: ResourceEstimate

    def for_mode(self, mode: MemoryMode) -> ResourceEstimate:
        if mode is MemoryMode.FULL:
            return self.full
        if mode is MemoryMode.TENSOR:
            return self.tensor
        if mode is MemoryMode.STREAMING:
            return self.streaming
        raise MemoryPlanningError("automatic mode has no direct resource estimate")


@dataclass(frozen=True, slots=True)
class ResourceCapacity:
    """Point-in-time available resources on the selected host and filesystem."""

    ram_bytes: int
    vram_bytes: int
    scratch_bytes: int

    def __post_init__(self) -> None:
        _non_negative("available RAM", self.ram_bytes)
        _non_negative("available VRAM", self.vram_bytes)
        _non_negative("available scratch", self.scratch_bytes)


@dataclass(frozen=True, slots=True)
class ResourceCeilings:
    """Optional user hard limits, applied in addition to host availability."""

    max_ram_bytes: int | None = None
    max_vram_bytes: int | None = None
    max_scratch_bytes: int | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum RAM", self.max_ram_bytes),
            ("maximum VRAM", self.max_vram_bytes),
            ("maximum scratch", self.max_scratch_bytes),
        ):
            if value is not None:
                _non_negative(name, value)

    def apply(self, available: ResourceCapacity) -> ResourceCapacity:
        return ResourceCapacity(
            min(available.ram_bytes, self.max_ram_bytes)
            if self.max_ram_bytes is not None
            else available.ram_bytes,
            min(available.vram_bytes, self.max_vram_bytes)
            if self.max_vram_bytes is not None
            else available.vram_bytes,
            min(available.scratch_bytes, self.max_scratch_bytes)
            if self.max_scratch_bytes is not None
            else available.scratch_bytes,
        )


@dataclass(frozen=True, slots=True)
class RejectedMemoryMode:
    mode: MemoryMode
    exceeded_resources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    """Selected execution mode with auditable peak estimates and limits."""

    mode: MemoryMode
    peak: ResourceEstimate
    effective_capacity: ResourceCapacity
    rejected_modes: tuple[RejectedMemoryMode, ...]


def _exceeded(estimate: ResourceEstimate, capacity: ResourceCapacity) -> tuple[str, ...]:
    exceeded: list[str] = []
    if estimate.peak_ram_bytes > capacity.ram_bytes:
        exceeded.append("ram")
    if estimate.peak_vram_bytes > capacity.vram_bytes:
        exceeded.append("vram")
    if estimate.scratch_bytes > capacity.scratch_bytes:
        exceeded.append("scratch")
    return tuple(exceeded)


def plan_memory_mode(
    requested_mode: MemoryMode,
    estimates: OperationMemoryEstimates,
    available: ResourceCapacity,
    ceilings: ResourceCeilings | None = None,
) -> MemoryPlan:
    """Choose the least restrictive fitting mode or validate an explicit mode.

    Automatic planning considers full, tensor, then streaming execution. A mode is
    selected only when all of its peak estimates fit both current availability and
    every configured user ceiling. Explicit requests never silently fall back.
    """

    effective = (ceilings or ResourceCeilings()).apply(available)
    modes = (
        (MemoryMode.FULL, MemoryMode.TENSOR, MemoryMode.STREAMING)
        if requested_mode is MemoryMode.AUTO
        else (requested_mode,)
    )
    rejected: list[RejectedMemoryMode] = []
    for mode in modes:
        estimate = estimates.for_mode(mode)
        exceeded = _exceeded(estimate, effective)
        if not exceeded:
            return MemoryPlan(mode, estimate, effective, tuple(rejected))
        rejected.append(RejectedMemoryMode(mode, exceeded))

    detail = "; ".join(
        f"{item.mode.value} exceeds {', '.join(item.exceeded_resources)}"
        for item in rejected
    )
    raise MemoryPlanningError(f"no permitted memory mode fits the resource limits: {detail}")

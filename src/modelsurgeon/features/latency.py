"""Warmed robust component latency profiling with explicit CPU/CUDA timing provenance."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

LATENCY_PROFILE_VERSION = "1"


class LatencyProfileError(ValueError):
    """Raised when a latency profile cannot preserve its measurement contract."""


class LatencyBackend(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class CudaLatencyTimer(Protocol):
    """CUDA event timer contract; implementations must synchronize completed work."""

    def measure_ns(self, operation: Callable[[], object]) -> int: ...


@dataclass(frozen=True, slots=True)
class LatencyProfileConfig:
    warmup_runs: int = 3
    sample_runs: int = 20

    def __post_init__(self) -> None:
        if self.warmup_runs < 0:
            raise LatencyProfileError("latency warmup count cannot be negative")
        if self.sample_runs < 3:
            raise LatencyProfileError("latency profiling requires at least three samples")


@dataclass(frozen=True, slots=True)
class LatencyEnvironment:
    backend: LatencyBackend
    timer: str
    device: str
    synchronization: str

    def __post_init__(self) -> None:
        if not self.timer or not self.device or not self.synchronization:
            raise LatencyProfileError("latency environment fields cannot be empty")

    def to_record(self) -> dict[str, str]:
        return {
            "backend": self.backend.value,
            "timer": self.timer,
            "device": self.device,
            "synchronization": self.synchronization,
        }


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    version: str
    sample_count: int
    warmup_count: int
    median_ns: float
    median_absolute_deviation_ns: float
    minimum_ns: int
    maximum_ns: int
    profiler_overhead_ns: float
    cpu_median_ns: float | None
    cuda_median_ns: float | None
    environment: LatencyEnvironment

    def __post_init__(self) -> None:
        numeric = (
            self.median_ns,
            self.median_absolute_deviation_ns,
            self.profiler_overhead_ns,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise LatencyProfileError("latency statistics must be finite and non-negative")
        if self.sample_count <= 0 or self.warmup_count < 0:
            raise LatencyProfileError("latency sample counts are invalid")
        if self.minimum_ns < 0 or self.maximum_ns < self.minimum_ns:
            raise LatencyProfileError("latency extrema are invalid")
        if self.environment.backend is LatencyBackend.CPU and self.cuda_median_ns is not None:
            raise LatencyProfileError("CPU profiles cannot fabricate CUDA timing")
        if self.environment.backend is LatencyBackend.CUDA and self.cuda_median_ns is None:
            raise LatencyProfileError("CUDA profiles require CUDA event timing")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "sample_count": self.sample_count,
            "warmup_count": self.warmup_count,
            "median_ns": self.median_ns,
            "median_absolute_deviation_ns": self.median_absolute_deviation_ns,
            "minimum_ns": self.minimum_ns,
            "maximum_ns": self.maximum_ns,
            "profiler_overhead_ns": self.profiler_overhead_ns,
            "cpu_median_ns": self.cpu_median_ns,
            "cuda_median_ns": self.cuda_median_ns,
            "environment": self.environment.to_record(),
        }


def _median_absolute_deviation(samples: tuple[int, ...]) -> float:
    median = statistics.median(samples)
    return float(statistics.median(abs(sample - median) for sample in samples))


def _measure_cpu_ns(
    operation: Callable[[], object],
    clock_ns: Callable[[], int],
    synchronize: Callable[[], None] | None,
) -> int:
    if synchronize is not None:
        synchronize()
    started = clock_ns()
    operation()
    if synchronize is not None:
        synchronize()
    elapsed = clock_ns() - started
    if elapsed < 0:
        raise LatencyProfileError("latency clock moved backwards")
    return elapsed


def profile_component_latency(
    operation: Callable[[], object],
    config: LatencyProfileConfig | None = None,
    *,
    backend: LatencyBackend = LatencyBackend.CPU,
    cuda_timer: CudaLatencyTimer | None = None,
    synchronize: Callable[[], None] | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    device: str = "cpu",
) -> LatencyProfile:
    """Profile one operation after warmup using robust medians and explicit timing provenance."""

    resolved = config or LatencyProfileConfig()
    if backend is LatencyBackend.CUDA and cuda_timer is None:
        raise LatencyProfileError("CUDA latency profiling requires a CUDA event timer")
    if backend is LatencyBackend.CPU and cuda_timer is not None:
        raise LatencyProfileError("CPU latency profiling cannot accept a CUDA timer")

    for _ in range(resolved.warmup_runs):
        operation()
    if synchronize is not None:
        synchronize()

    if backend is LatencyBackend.CUDA:
        assert cuda_timer is not None
        samples = tuple(cuda_timer.measure_ns(operation) for _ in range(resolved.sample_runs))
        overhead_samples = tuple(
            cuda_timer.measure_ns(lambda: None) for _ in range(resolved.sample_runs)
        )
        environment = LatencyEnvironment(
            LatencyBackend.CUDA,
            "cuda_event",
            device,
            "cuda_event_synchronize",
        )
    else:
        samples = tuple(
            _measure_cpu_ns(operation, clock_ns, synchronize)
            for _ in range(resolved.sample_runs)
        )
        overhead_samples = tuple(
            _measure_cpu_ns(lambda: None, clock_ns, synchronize)
            for _ in range(resolved.sample_runs)
        )
        environment = LatencyEnvironment(
            LatencyBackend.CPU,
            "perf_counter_ns",
            device,
            "none" if synchronize is None else "explicit_callback",
        )

    if any(sample < 0 for sample in (*samples, *overhead_samples)):
        raise LatencyProfileError("latency timer returned a negative duration")
    median = float(statistics.median(samples))
    overhead = float(statistics.median(overhead_samples))
    return LatencyProfile(
        version=LATENCY_PROFILE_VERSION,
        sample_count=len(samples),
        warmup_count=resolved.warmup_runs,
        median_ns=median,
        median_absolute_deviation_ns=_median_absolute_deviation(samples),
        minimum_ns=min(samples),
        maximum_ns=max(samples),
        profiler_overhead_ns=overhead,
        cpu_median_ns=median if backend is LatencyBackend.CPU else None,
        cuda_median_ns=median if backend is LatencyBackend.CUDA else None,
        environment=environment,
    )

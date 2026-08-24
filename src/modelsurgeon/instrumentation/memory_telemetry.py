"""Bounded process-RAM and optional CUDA allocation telemetry around named operations."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

MEMORY_TELEMETRY_VERSION = "1"


class MemoryTelemetryError(RuntimeError):
    """Raised when memory telemetry cannot preserve its lifecycle contract."""


class CudaMemoryProvider(Protocol):
    def reset_peak_stats(self) -> None: ...

    def allocated_bytes(self) -> int: ...

    def reserved_bytes(self) -> int: ...

    def max_allocated_bytes(self) -> int: ...

    def max_reserved_bytes(self) -> int: ...


class TorchCudaMemoryProvider:
    """Optional PyTorch-backed CUDA allocation telemetry without a hard torch dependency."""

    def __init__(self, device: int | str | None = None) -> None:
        try:
            torch: Any = import_module("torch")
        except Exception as error:
            raise MemoryTelemetryError(
                "PyTorch is unavailable for CUDA memory telemetry"
            ) from error
        try:
            available = bool(torch.cuda.is_available())
        except Exception as error:
            raise MemoryTelemetryError("CUDA availability probe failed") from error
        if not available:
            raise MemoryTelemetryError("CUDA is unavailable for memory telemetry")
        self._torch = torch
        self._device = device

    def reset_peak_stats(self) -> None:
        self._torch.cuda.reset_peak_memory_stats(self._device)

    def allocated_bytes(self) -> int:
        return int(self._torch.cuda.memory_allocated(self._device))

    def reserved_bytes(self) -> int:
        return int(self._torch.cuda.memory_reserved(self._device))

    def max_allocated_bytes(self) -> int:
        return int(self._torch.cuda.max_memory_allocated(self._device))

    def max_reserved_bytes(self) -> int:
        return int(self._torch.cuda.max_memory_reserved(self._device))


@dataclass(frozen=True, slots=True)
class MemoryTelemetryConfig:
    sampling_enabled: bool = False
    sample_interval_seconds: float = 0.05
    max_samples: int = 4096

    def __post_init__(self) -> None:
        if self.sample_interval_seconds <= 0:
            raise MemoryTelemetryError("memory sample interval must be positive")
        if self.max_samples < 2:
            raise MemoryTelemetryError(
                "memory telemetry requires capacity for at least two samples"
            )


@dataclass(frozen=True, slots=True)
class MemorySample:
    elapsed_seconds: float
    rss_bytes: int | None
    cuda_allocated_bytes: int | None
    cuda_reserved_bytes: int | None

    def __post_init__(self) -> None:
        if self.elapsed_seconds < 0:
            raise MemoryTelemetryError("memory sample time cannot be negative")
        for value in (self.rss_bytes, self.cuda_allocated_bytes, self.cuda_reserved_bytes):
            if value is not None and value < 0:
                raise MemoryTelemetryError("memory sample values cannot be negative")


@dataclass(frozen=True, slots=True)
class MemoryTelemetryReport:
    version: str
    operation: str
    sampling_enabled: bool
    sample_interval_seconds: float | None
    samples: tuple[MemorySample, ...]
    peak_rss_bytes: int | None
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None
    cuda_available: bool

    def __post_init__(self) -> None:
        if not self.operation:
            raise MemoryTelemetryError("memory telemetry operation name is required")
        if not self.samples:
            raise MemoryTelemetryError("memory telemetry requires at least one sample")
        if not self.sampling_enabled and self.sample_interval_seconds is not None:
            raise MemoryTelemetryError("disabled sampling cannot report an interval")
        if not self.cuda_available and (
            self.peak_cuda_allocated_bytes is not None or self.peak_cuda_reserved_bytes is not None
        ):
            raise MemoryTelemetryError("CPU-only telemetry cannot fabricate CUDA peaks")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "operation": self.operation,
            "sampling_enabled": self.sampling_enabled,
            "sample_interval_seconds": self.sample_interval_seconds,
            "sample_count": len(self.samples),
            "samples": [
                {
                    "elapsed_seconds": sample.elapsed_seconds,
                    "rss_bytes": sample.rss_bytes,
                    "cuda_allocated_bytes": sample.cuda_allocated_bytes,
                    "cuda_reserved_bytes": sample.cuda_reserved_bytes,
                }
                for sample in self.samples
            ],
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "cuda_available": self.cuda_available,
        }


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_rss_bytes() -> int | None:
    try:
        windll: Any = getattr(ctypes, "windll")  # noqa: B009
        kernel32: Any = windll.kernel32
        psapi: Any = windll.psapi
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else None
    except (AttributeError, OSError, ValueError):
        return None


def _proc_rss_bytes() -> int | None:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return None
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None
    try:
        fields = statm.read_text(encoding="ascii").split()
        resident_pages = int(fields[1])
        page_size = int(sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, ValueError):
        return None
    return resident_pages * page_size


def process_rss_bytes() -> int | None:
    """Return current process resident bytes where the host exposes a safe native probe."""

    if sys.platform == "win32":
        return _windows_rss_bytes()
    return _proc_rss_bytes()


def _sample(
    started: float,
    cuda: CudaMemoryProvider | None,
    monotonic: Callable[[], float],
) -> MemorySample:
    return MemorySample(
        elapsed_seconds=max(0.0, monotonic() - started),
        rss_bytes=process_rss_bytes(),
        cuda_allocated_bytes=None if cuda is None else cuda.allocated_bytes(),
        cuda_reserved_bytes=None if cuda is None else cuda.reserved_bytes(),
    )


def collect_memory_telemetry(
    operation_name: str,
    operation: Callable[[], object],
    config: MemoryTelemetryConfig | None = None,
    *,
    cuda: CudaMemoryProvider | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    report_callback: Callable[[MemoryTelemetryReport], None] | None = None,
) -> MemoryTelemetryReport:
    """Execute one operation while collecting bounded memory telemetry with guaranteed cleanup.

    ``report_callback`` receives the final bounded report from the ``finally`` path, so callers
    can persist partial telemetry even when ``operation`` raises. The original exception is then
    re-raised unchanged.
    """

    if not operation_name:
        raise MemoryTelemetryError("memory telemetry operation name is required")
    resolved = config or MemoryTelemetryConfig()
    if cuda is not None:
        cuda.reset_peak_stats()

    started = monotonic()
    samples: list[MemorySample] = [_sample(started, cuda, monotonic)]
    samples_lock = threading.Lock()
    stop = threading.Event()
    sampler: threading.Thread | None = None
    report: MemoryTelemetryReport | None = None

    if resolved.sampling_enabled:

        def sample_loop() -> None:
            while not stop.wait(resolved.sample_interval_seconds):
                with samples_lock:
                    if len(samples) >= resolved.max_samples - 1:
                        return
                    samples.append(_sample(started, cuda, monotonic))

        sampler = threading.Thread(
            target=sample_loop,
            name=f"modelsurgeon-memory-{operation_name}",
            daemon=True,
        )
        sampler.start()

    try:
        operation()
    finally:
        if sampler is not None:
            stop.set()
            sampler.join()
            if sampler.is_alive():
                raise MemoryTelemetryError("memory telemetry sampler failed to stop")
        with samples_lock:
            if len(samples) < resolved.max_samples:
                samples.append(_sample(started, cuda, monotonic))
            snapshot = tuple(samples)
        rss_values = tuple(
            sample.rss_bytes for sample in snapshot if sample.rss_bytes is not None
        )
        peak_rss = max(rss_values) if rss_values else None
        report = MemoryTelemetryReport(
            version=MEMORY_TELEMETRY_VERSION,
            operation=operation_name,
            sampling_enabled=resolved.sampling_enabled,
            sample_interval_seconds=(
                resolved.sample_interval_seconds if resolved.sampling_enabled else None
            ),
            samples=snapshot,
            peak_rss_bytes=peak_rss,
            peak_cuda_allocated_bytes=(
                None if cuda is None else cuda.max_allocated_bytes()
            ),
            peak_cuda_reserved_bytes=(
                None if cuda is None else cuda.max_reserved_bytes()
            ),
            cuda_available=cuda is not None,
        )
        if report_callback is not None:
            report_callback(report)

    if report is None:
        raise MemoryTelemetryError("memory telemetry report was not finalized")
    return report

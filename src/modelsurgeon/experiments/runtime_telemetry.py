"""Versioned per-stage timing, throughput, memory, and process-I/O telemetry."""

from __future__ import annotations

import ctypes
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from modelsurgeon.experiments.hardware import HardwareInventory
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.instrumentation.memory_telemetry import (
    CudaMemoryProvider,
    MemoryTelemetryConfig,
    MemoryTelemetryReport,
    collect_memory_telemetry,
)

if TYPE_CHECKING:
    from modelsurgeon.experiments.store import ExperimentMetadataStore, StoredStageTelemetry

STAGE_TELEMETRY_VERSION = "1"


class StageTelemetryError(RuntimeError):
    """Raised when stage telemetry cannot be measured or persisted safely."""


class StageTelemetryState(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


@dataclass(frozen=True, slots=True)
class StageThroughput:
    tokens: int | None = None
    candidates: int | None = None

    def __post_init__(self) -> None:
        for value in (self.tokens, self.candidates):
            if value is not None and (isinstance(value, bool) or value < 0):
                raise StageTelemetryError("stage throughput counts must be non-negative integers")

    def to_record(self) -> dict[str, int | None]:
        return {"tokens": self.tokens, "candidates": self.candidates}


@dataclass(frozen=True, slots=True)
class ProcessIOCounters:
    read_bytes: int
    write_bytes: int

    def __post_init__(self) -> None:
        if self.read_bytes < 0 or self.write_bytes < 0:
            raise StageTelemetryError("process I/O counters cannot be negative")


class _WindowsIOCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


def _windows_io_counters() -> ProcessIOCounters | None:
    try:
        windll: Any = getattr(ctypes, "windll")  # noqa: B009
        kernel32: Any = windll.kernel32
        counters = _WindowsIOCounters()
        handle = kernel32.GetCurrentProcess()
        ok = kernel32.GetProcessIoCounters(handle, ctypes.byref(counters))
        if not ok:
            return None
        return ProcessIOCounters(
            int(counters.ReadTransferCount),
            int(counters.WriteTransferCount),
        )
    except (AttributeError, OSError, ValueError):
        return None


def _proc_io_counters() -> ProcessIOCounters | None:
    path = Path("/proc/self/io")
    if not path.is_file():
        return None
    try:
        fields = {}
        for line in path.read_text(encoding="ascii").splitlines():
            key, separator, raw_value = line.partition(":")
            if separator:
                fields[key.strip()] = int(raw_value.strip())
        return ProcessIOCounters(fields["read_bytes"], fields["write_bytes"])
    except (KeyError, OSError, ValueError):
        return None


def process_io_counters() -> ProcessIOCounters | None:
    """Return process read/write byte counters where the host exposes them safely."""

    if sys.platform == "win32":
        return _windows_io_counters()
    return _proc_io_counters()


@dataclass(frozen=True, slots=True)
class HardwareNormalizationContext:
    """Stable hardware fields needed to group or compare runtime measurements."""

    os_name: str
    cpu_architecture: str
    cpu_processor: str
    logical_cores: int | None
    memory_total_bytes: int | None
    cuda_available: bool
    cuda_devices: tuple[tuple[str, int | None, str | None], ...]

    def __post_init__(self) -> None:
        if not self.os_name or not self.cpu_architecture:
            raise StageTelemetryError("hardware normalization requires OS and CPU architecture")
        if self.logical_cores is not None and self.logical_cores <= 0:
            raise StageTelemetryError("hardware logical-core count must be positive when known")
        if self.memory_total_bytes is not None and self.memory_total_bytes <= 0:
            raise StageTelemetryError("hardware memory total must be positive when known")
        if not self.cuda_available and self.cuda_devices:
            raise StageTelemetryError("CPU-only hardware context cannot contain CUDA devices")
        if self.cuda_devices != tuple(sorted(self.cuda_devices)):
            raise StageTelemetryError("CUDA hardware context must use canonical device ordering")

    @classmethod
    def from_inventory(cls, inventory: HardwareInventory) -> HardwareNormalizationContext:
        devices = tuple(
            sorted(
                (
                    device.name,
                    device.total_memory_bytes,
                    device.compute_capability,
                )
                for device in inventory.cuda.devices
            )
        )
        return cls(
            inventory.os_name,
            inventory.cpu.architecture,
            inventory.cpu.processor,
            inventory.cpu.logical_cores,
            inventory.memory.total_bytes,
            inventory.cuda.available,
            devices,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "os_name": self.os_name,
            "cpu_architecture": self.cpu_architecture,
            "cpu_processor": self.cpu_processor,
            "logical_cores": self.logical_cores,
            "memory_total_bytes": self.memory_total_bytes,
            "cuda_available": self.cuda_available,
            "cuda_devices": [
                {
                    "name": name,
                    "total_memory_bytes": total_memory_bytes,
                    "compute_capability": compute_capability,
                }
                for name, total_memory_bytes, compute_capability in self.cuda_devices
            ],
        }

    @property
    def context_id(self) -> str:
        import hashlib

        digest = hashlib.sha256(
            canonical_identity_json(self.to_record()).encode("utf-8")
        ).hexdigest()
        return f"hwctx_{digest}"

    def comparable_to(self, other: HardwareNormalizationContext) -> bool:
        return self.context_id == other.context_id


@dataclass(frozen=True, slots=True)
class StageTelemetrySnapshot:
    stage: str
    state: StageTelemetryState
    wall_seconds: float
    cpu_seconds: float
    throughput: StageThroughput
    peak_rss_bytes: int | None
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None
    io_read_bytes: int | None
    io_write_bytes: int | None
    hardware: HardwareNormalizationContext
    version: str = STAGE_TELEMETRY_VERSION

    def __post_init__(self) -> None:
        if self.version != STAGE_TELEMETRY_VERSION:
            raise StageTelemetryError(f"unsupported stage telemetry version {self.version}")
        if not self.stage:
            raise StageTelemetryError("stage telemetry requires a non-empty stage name")
        for value in (self.wall_seconds, self.cpu_seconds):
            if value < 0 or not _finite(value):
                raise StageTelemetryError("stage wall/CPU durations must be finite and non-negative")
        for value in (
            self.peak_rss_bytes,
            self.peak_cuda_allocated_bytes,
            self.peak_cuda_reserved_bytes,
            self.io_read_bytes,
            self.io_write_bytes,
        ):
            if value is not None and value < 0:
                raise StageTelemetryError("stage resource counters cannot be negative")
        if not self.hardware.cuda_available and (
            self.peak_cuda_allocated_bytes is not None
            or self.peak_cuda_reserved_bytes is not None
        ):
            raise StageTelemetryError("CPU-only stage telemetry cannot fabricate CUDA peaks")

    @property
    def tokens_per_second(self) -> float | None:
        if self.throughput.tokens is None or self.wall_seconds <= 0:
            return None
        return self.throughput.tokens / self.wall_seconds

    @property
    def candidates_per_second(self) -> float | None:
        if self.throughput.candidates is None or self.wall_seconds <= 0:
            return None
        return self.throughput.candidates / self.wall_seconds

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "stage": self.stage,
            "state": self.state.value,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            **self.throughput.to_record(),
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "io_read_bytes": self.io_read_bytes,
            "io_write_bytes": self.io_write_bytes,
            "tokens_per_second": self.tokens_per_second,
            "candidates_per_second": self.candidates_per_second,
            "hardware_context_id": self.hardware.context_id,
            "hardware": self.hardware.to_record(),
        }


def _finite(value: float) -> bool:
    import math

    return math.isfinite(value)


def _io_delta(
    before: ProcessIOCounters | None,
    after: ProcessIOCounters | None,
) -> tuple[int | None, int | None]:
    if before is None or after is None:
        return None, None
    if after.read_bytes < before.read_bytes or after.write_bytes < before.write_bytes:
        return None, None
    return after.read_bytes - before.read_bytes, after.write_bytes - before.write_bytes


@dataclass(frozen=True, slots=True)
class StageTelemetryExecution:
    result: object
    telemetry: StoredStageTelemetry


class StageTelemetryRecorder:
    """Run one stage and append a complete or partial immutable telemetry attempt."""

    def __init__(
        self,
        store: ExperimentMetadataStore,
        candidate_id: str,
        hardware: HardwareInventory,
        memory_config: MemoryTelemetryConfig | None = None,
        *,
        cuda: CudaMemoryProvider | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        process_time: Callable[[], float] = time.process_time,
        io_counters: Callable[[], ProcessIOCounters | None] = process_io_counters,
    ) -> None:
        if not candidate_id:
            raise StageTelemetryError("stage telemetry requires a candidate ID")
        self._store = store
        self._candidate_id = candidate_id
        self._hardware = HardwareNormalizationContext.from_inventory(hardware)
        self._memory_config = memory_config
        self._cuda = cuda
        self._monotonic = monotonic
        self._process_time = process_time
        self._io_counters = io_counters

    def run(
        self,
        stage: str,
        operation: Callable[[], object],
        *,
        throughput: Callable[[], StageThroughput] | None = None,
    ) -> StageTelemetryExecution:
        if not stage:
            raise StageTelemetryError("stage telemetry requires a non-empty stage name")
        started_wall = self._monotonic()
        started_cpu = self._process_time()
        started_io = self._io_counters()
        completed = False
        result_box: list[object] = []
        stored_box: list[StoredStageTelemetry] = []

        def measured_operation() -> object:
            nonlocal completed
            result = operation()
            result_box.append(result)
            completed = True
            return result

        def persist_report(memory: MemoryTelemetryReport) -> None:
            ended_wall = self._monotonic()
            ended_cpu = self._process_time()
            ended_io = self._io_counters()
            read_bytes, write_bytes = _io_delta(started_io, ended_io)
            counts = StageThroughput() if throughput is None else throughput()
            snapshot = StageTelemetrySnapshot(
                stage,
                StageTelemetryState.COMPLETE if completed else StageTelemetryState.PARTIAL,
                max(0.0, ended_wall - started_wall),
                max(0.0, ended_cpu - started_cpu),
                counts,
                memory.peak_rss_bytes,
                memory.peak_cuda_allocated_bytes,
                memory.peak_cuda_reserved_bytes,
                read_bytes,
                write_bytes,
                self._hardware,
            )
            stored_box.append(self._store.append_stage_telemetry(self._candidate_id, snapshot))

        collect_memory_telemetry(
            stage,
            measured_operation,
            self._memory_config,
            cuda=self._cuda,
            monotonic=self._monotonic,
            report_callback=persist_report,
        )
        if len(result_box) != 1 or len(stored_box) != 1:
            raise StageTelemetryError("completed stage did not produce exactly one telemetry snapshot")
        return StageTelemetryExecution(result_box[0], stored_box[0])

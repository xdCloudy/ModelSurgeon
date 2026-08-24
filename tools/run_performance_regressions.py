"""Run bounded CPU/GGUF/GPU fixtures against versioned regression budgets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from modelsurgeon.adapters.gguf import (
    GGUFDiskEstimate,
    GGUFValueType,
    GGUFWriteMetadata,
    GGUFWriteTensor,
    plan_gguf_output,
    preflight_gguf_disk,
    write_gguf_transactionally,
)
from modelsurgeon.evaluation.performance_regression import (
    PerformanceBudgetManifest,
    PerformanceMeasurement,
    PerformanceRegressionError,
    RegressionLane,
    evaluate_performance_regression,
    load_performance_budget_manifest,
)
from modelsurgeon.experiments.hardware import collect_hardware_inventory
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.runtime_telemetry import (
    HardwareNormalizationContext,
    ProcessIOCounters,
    process_io_counters,
)
from modelsurgeon.instrumentation.memory_telemetry import (
    MemoryTelemetryConfig,
    MemoryTelemetryReport,
    TorchCudaMemoryProvider,
    collect_memory_telemetry,
)

_MIB = 1024 * 1024
_FIXTURE_WORK_BYTES = {
    "pr-cpu-v1": {"cpu_canonical_hash": 8 * _MIB, "gguf_stream_write": 8 * _MIB},
    "consumer-cpu-v1": {
        "cpu_canonical_hash": 64 * _MIB,
        "gguf_stream_write": 64 * _MIB,
    },
    "consumer-gpu-v1": {"cuda_matmul": 1024},
}
_FIXTURE_LANES = {
    "pr-cpu-v1": RegressionLane.PR_CPU,
    "consumer-cpu-v1": RegressionLane.LARGE_CPU,
    "consumer-gpu-v1": RegressionLane.GPU,
}
_METRIC_UNITS = {
    "cpu_seconds": "seconds",
    "io_read_bytes": "bytes",
    "io_write_bytes": "bytes",
    "peak_rss_bytes": "bytes",
    "peak_vram_allocated_bytes": "bytes",
    "peak_vram_reserved_bytes": "bytes",
    "wall_seconds": "seconds",
    "work_items_per_second": "bytes_per_second",
}


class PerformanceRunnerError(RuntimeError):
    """Raised when a fixture cannot emit complete comparable telemetry."""


def _io_delta(
    before: ProcessIOCounters | None,
    after: ProcessIOCounters | None,
) -> tuple[int | None, int | None]:
    if before is None or after is None:
        return None, None
    if after.read_bytes < before.read_bytes or after.write_bytes < before.write_bytes:
        return None, None
    return after.read_bytes - before.read_bytes, after.write_bytes - before.write_bytes


def _cpu_hash(work_bytes: int) -> Callable[[Path], int]:
    chunk = bytes((index * 17 + 3) % 256 for index in range(256 * 1024))

    def operation(_: Path) -> int:
        remaining = work_bytes
        digest = hashlib.sha256()
        while remaining:
            selected = chunk[: min(remaining, len(chunk))]
            digest.update(selected)
            remaining -= len(selected)
        if len(digest.digest()) != 32:
            raise PerformanceRunnerError("hash fixture did not produce SHA-256")
        return work_bytes

    return operation


def _repeated_chunks(total_bytes: int, chunk_bytes: int = 256 * 1024) -> Iterator[bytes]:
    pattern = b"\x00\x00\x80\x3f" * (chunk_bytes // 4)
    remaining = total_bytes
    while remaining:
        selected = pattern[: min(remaining, len(pattern))]
        yield selected
        remaining -= len(selected)


def _gguf_stream(work_bytes: int) -> Callable[[Path], int]:
    if work_bytes % 4:
        raise PerformanceRunnerError("GGUF fixture bytes must align to F32 elements")

    def operation(directory: Path) -> int:
        output = directory / "streaming-fixture.gguf"
        metadata = (
            GGUFWriteMetadata("general.architecture", GGUFValueType.STRING, "benchmark"),
            GGUFWriteMetadata("benchmark.fixture", GGUFValueType.STRING, "deterministic-v1"),
        )
        tensor = GGUFWriteTensor(
            "benchmark.weight",
            (work_bytes // 4,),
            0,
            _repeated_chunks(work_bytes),
        )
        layout = plan_gguf_output(metadata, (tensor,))
        disk = preflight_gguf_disk(
            output,
            directory,
            GGUFDiskEstimate(layout.total_bytes, 0, alignment_bytes=layout.alignment),
        )
        result = write_gguf_transactionally(output, metadata, (tensor,), disk)
        if result.file_size != layout.total_bytes:
            raise PerformanceRunnerError("GGUF fixture output size mismatch")
        return result.file_size

    return operation


def _cuda_matmul(matrix_size: int) -> Callable[[Path], int]:
    def operation(_: Path) -> int:
        import torch

        left = torch.arange(
            matrix_size * matrix_size,
            device="cuda",
            dtype=torch.float32,
        ).reshape(matrix_size, matrix_size)
        right = left.transpose(0, 1).contiguous()
        output = left @ right
        checksum = float(output[0, 0].item())
        if not math.isfinite(checksum):
            raise PerformanceRunnerError("CUDA fixture produced non-finite output")
        torch.cuda.synchronize()
        return 2 * matrix_size**3

    return operation


def _case_operation(
    manifest: PerformanceBudgetManifest,
    case_id: str,
) -> Callable[[Path], int]:
    fixture = _FIXTURE_WORK_BYTES.get(manifest.fixture_id)
    if fixture is None or case_id not in fixture:
        raise PerformanceRunnerError(
            f"fixture {manifest.fixture_id!r} does not define case {case_id!r}"
        )
    work = fixture[case_id]
    if case_id == "cpu_canonical_hash":
        return _cpu_hash(work)
    if case_id == "gguf_stream_write":
        return _gguf_stream(work)
    if case_id == "cuda_matmul":
        return _cuda_matmul(work)
    raise PerformanceRunnerError(f"unknown performance case {case_id!r}")


def _run_once(
    case_id: str,
    operation: Callable[[Path], int],
    directory: Path,
    *,
    cuda: TorchCudaMemoryProvider | None,
) -> dict[str, float]:
    captured: list[MemoryTelemetryReport] = []
    result: list[int] = []
    started_io = process_io_counters()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()

    def execute() -> None:
        result.append(operation(directory))

    collect_memory_telemetry(
        case_id,
        execute,
        MemoryTelemetryConfig(True, 0.005, 8192),
        cuda=cuda,
        report_callback=captured.append,
    )
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    ended_io = process_io_counters()
    read_bytes, write_bytes = _io_delta(started_io, ended_io)
    if len(captured) != 1 or len(result) != 1 or wall_seconds <= 0:
        raise PerformanceRunnerError("fixture did not produce exactly one telemetry report")
    report = captured[0]
    values: dict[str, float] = {
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "work_items_per_second": result[0] / wall_seconds,
    }
    optional = {
        "peak_rss_bytes": report.peak_rss_bytes,
        "peak_vram_allocated_bytes": report.peak_cuda_allocated_bytes,
        "peak_vram_reserved_bytes": report.peak_cuda_reserved_bytes,
        "io_read_bytes": read_bytes,
        "io_write_bytes": write_bytes,
    }
    values.update({name: float(value) for name, value in optional.items() if value is not None})
    return values


def run_profile(
    manifest: PerformanceBudgetManifest,
    *,
    scratch_root: Path,
) -> tuple[PerformanceMeasurement, ...]:
    """Run only cases declared by one lane and return budget-selected measurements."""

    expected_lane = _FIXTURE_LANES.get(manifest.fixture_id)
    if expected_lane is None or expected_lane is not manifest.lane:
        raise PerformanceRunnerError(
            f"fixture {manifest.fixture_id!r} is not valid for lane {manifest.lane.value!r}"
        )
    for budget in manifest.budgets:
        expected_unit = _METRIC_UNITS.get(budget.metric)
        if expected_unit is None or expected_unit != budget.unit:
            raise PerformanceRunnerError(
                f"metric {budget.metric!r} requires canonical unit {expected_unit!r}"
            )
    inventory = collect_hardware_inventory(scratch_root)
    hardware = HardwareNormalizationContext.from_inventory(inventory)
    cuda = None
    if manifest.lane is RegressionLane.GPU:
        if not hardware.cuda_available:
            raise PerformanceRunnerError("GPU performance profile requires a CUDA device")
        cuda = TorchCudaMemoryProvider()
    case_ids = tuple(sorted({budget.case_id for budget in manifest.budgets}))
    measurements: list[PerformanceMeasurement] = []
    for case_id in case_ids:
        operation = _case_operation(manifest, case_id)
        for repetition in range(manifest.min_repetitions):
            gc.collect()
            if cuda is not None:
                import torch

                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            with tempfile.TemporaryDirectory(
                prefix=f"modelsurgeon-{case_id}-",
                dir=scratch_root,
            ) as temporary:
                values = _run_once(case_id, operation, Path(temporary), cuda=cuda)
            for budget in manifest.budgets:
                if budget.case_id != case_id:
                    continue
                value = values.get(budget.metric)
                if value is None:
                    raise PerformanceRunnerError(
                        f"case {case_id} did not expose required metric {budget.metric}"
                    )
                measurements.append(
                    PerformanceMeasurement(
                        case_id,
                        budget.stage,
                        budget.metric,
                        budget.unit,
                        value,
                        repetition,
                        manifest.fixture_id,
                        hardware,
                    )
                )
    return tuple(measurements)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budgets",
        type=Path,
        default=Path("tests/fixtures/performance_budgets_v1.json"),
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists():
            raise PerformanceRunnerError(f"output already exists: {arguments.output}")
        manifest = load_performance_budget_manifest(
            arguments.budgets.read_text(encoding="utf-8"),
            arguments.profile,
        )
        scratch = arguments.scratch or arguments.output.parent
        scratch.mkdir(parents=True, exist_ok=True)
        measurements = run_profile(manifest, scratch_root=scratch)
        report = evaluate_performance_regression(manifest, measurements)
        record = {
            **report.to_record(),
            "measurements": [item.to_record() for item in measurements],
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            canonical_identity_json(record) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "profile": manifest.profile,
                    "report_id": report.report_id,
                    "hardware_context_id": report.hardware.context_id,
                    "budgets": len(report.results),
                    "passed": report.passed,
                    "output": str(arguments.output),
                },
                indent=2,
            )
        )
        return 0 if report.passed else 1
    except (OSError, PerformanceRegressionError, PerformanceRunnerError) as error:
        print(f"performance regression error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

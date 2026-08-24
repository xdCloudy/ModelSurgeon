"""Pinned llama-bench prompt/generation throughput with child memory telemetry."""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from modelsurgeon.evaluation.llama_cpp import (
    LLAMA_CPP_VALIDATION_COMMIT,
    _read_bounded_log,
    _resolve_executable,
    _revision_matches,
)
from modelsurgeon.evaluation.llama_cpp_perplexity import _sha256_file

LLAMA_CPP_THROUGHPUT_SCHEMA_VERSION = 1


class LlamaCppThroughputError(RuntimeError):
    """Raised when a llama-bench configuration or model is unsafe."""


class GpuProcessMemoryProbe(Protocol):
    def __call__(self, process_id: int) -> int | None: ...


@dataclass(frozen=True, slots=True)
class LlamaCppThroughputConfig:
    executable: str | Path = "llama-bench"
    expected_revision: str = LLAMA_CPP_VALIDATION_COMMIT
    prompt_tokens: int = 512
    generation_tokens: int = 128
    batch_size: int = 512
    microbatch_size: int = 512
    threads: int = 1
    gpu_layers: int = 0
    repetitions: int = 5
    warmup: bool = True
    timeout_seconds: float = 300.0
    sample_interval_seconds: float = 0.01
    max_log_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        if not str(self.executable):
            raise LlamaCppThroughputError("llama-bench executable cannot be empty")
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", self.expected_revision) is None:
            raise LlamaCppThroughputError(
                "expected llama.cpp revision must be a 7-40 character hexadecimal commit"
            )
        if any(
            value <= 0
            for value in (
                self.prompt_tokens,
                self.generation_tokens,
                self.batch_size,
                self.microbatch_size,
                self.threads,
                self.repetitions,
            )
        ):
            raise LlamaCppThroughputError(
                "prompt, generation, batch, thread, and repetition counts must be positive"
            )
        if self.microbatch_size > self.batch_size:
            raise LlamaCppThroughputError("microbatch size cannot exceed batch size")
        if self.gpu_layers < 0:
            raise LlamaCppThroughputError("GPU layer count must be non-negative")
        if self.timeout_seconds <= 0 or self.sample_interval_seconds <= 0:
            raise LlamaCppThroughputError("timeout and sample interval must be positive")
        if self.max_log_bytes <= 0:
            raise LlamaCppThroughputError("maximum log bytes must be positive")

    @property
    def context_tokens(self) -> int:
        # llama-bench creates separate prompt-only and generation-only test instances
        # and sizes their shared context to the largest requested token count.
        return max(self.prompt_tokens, self.generation_tokens)

    def to_record(self) -> dict[str, object]:
        return {
            "expected_revision": self.expected_revision.lower(),
            "prompt_tokens": self.prompt_tokens,
            "generation_tokens": self.generation_tokens,
            "context_tokens": self.context_tokens,
            "batch_size": self.batch_size,
            "microbatch_size": self.microbatch_size,
            "threads": self.threads,
            "gpu_layers": self.gpu_layers,
            "repetitions": self.repetitions,
            "warmup": self.warmup,
            "timeout_seconds": self.timeout_seconds,
            "sample_interval_seconds": self.sample_interval_seconds,
            "max_log_bytes": self.max_log_bytes,
        }


@dataclass(frozen=True, slots=True)
class LlamaCppThroughputPhase:
    phase: str
    tokens: int
    average_tokens_per_second: float
    standard_deviation_tokens_per_second: float
    average_latency_ns: int
    standard_deviation_latency_ns: int
    latency_samples_ns: tuple[int, ...]
    throughput_samples: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.phase not in {"prompt", "generation"} or self.tokens <= 0:
            raise LlamaCppThroughputError("throughput phase and token count are invalid")
        values = (
            self.average_tokens_per_second,
            self.standard_deviation_tokens_per_second,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise LlamaCppThroughputError("throughput rates must be finite and non-negative")
        if self.average_tokens_per_second <= 0 or self.average_latency_ns <= 0:
            raise LlamaCppThroughputError("average throughput and latency must be positive")
        if self.standard_deviation_latency_ns < 0:
            raise LlamaCppThroughputError("latency deviation cannot be negative")
        if not self.latency_samples_ns or len(self.latency_samples_ns) != len(
            self.throughput_samples
        ):
            raise LlamaCppThroughputError("throughput phases require aligned samples")
        if any(value <= 0 for value in self.latency_samples_ns) or any(
            not math.isfinite(value) or value <= 0 for value in self.throughput_samples
        ):
            raise LlamaCppThroughputError("throughput samples must be finite and positive")

    def to_record(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "tokens": self.tokens,
            "average_tokens_per_second": self.average_tokens_per_second,
            "standard_deviation_tokens_per_second": (
                self.standard_deviation_tokens_per_second
            ),
            "average_latency_ns": self.average_latency_ns,
            "standard_deviation_latency_ns": self.standard_deviation_latency_ns,
            "latency_samples_ns": list(self.latency_samples_ns),
            "throughput_samples": list(self.throughput_samples),
        }


@dataclass(frozen=True, slots=True)
class LlamaCppBenchmarkEnvironment:
    build_commit: str
    build_number: int
    cpu_info: str
    gpu_info: str
    backends: str
    model_type: str
    model_size: int
    model_parameters: int

    def comparison_key(self) -> tuple[object, ...]:
        return (
            self.build_commit,
            self.build_number,
            self.cpu_info,
            self.gpu_info,
            self.backends,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "build_commit": self.build_commit,
            "build_number": self.build_number,
            "cpu_info": self.cpu_info,
            "gpu_info": self.gpu_info,
            "backends": self.backends,
            "model_type": self.model_type,
            "model_size": self.model_size,
            "model_parameters": self.model_parameters,
        }


@dataclass(frozen=True, slots=True)
class LlamaCppThroughputReport:
    model_path: Path
    model_sha256: str
    config: LlamaCppThroughputConfig
    command: tuple[str, ...]
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    peak_rss_bytes: int | None
    peak_vram_bytes: int | None
    memory_sample_count: int
    environment: LlamaCppBenchmarkEnvironment | None
    prompt: LlamaCppThroughputPhase | None
    generation: LlamaCppThroughputPhase | None
    parse_error: str | None
    schema_version: int = LLAMA_CPP_THROUGHPUT_SCHEMA_VERSION

    @property
    def successful(self) -> bool:
        return (
            not self.timed_out
            and self.returncode == 0
            and self.parse_error is None
            and self.environment is not None
            and self.prompt is not None
            and self.generation is not None
            and self.peak_rss_bytes is not None
            and self.peak_vram_bytes is not None
        )

    @property
    def failure_reason(self) -> str | None:
        if self.successful:
            return None
        if self.timed_out:
            return "llama-bench timed out"
        if self.returncode != 0:
            return f"llama-bench exited with status {self.returncode}"
        if self.peak_rss_bytes is None or self.peak_vram_bytes is None:
            return "llama-bench process memory telemetry was unavailable"
        return self.parse_error or "llama-bench did not produce complete measurements"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "successful": self.successful,
            "failure_reason": self.failure_reason,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "config": self.config.to_record(),
            "command": list(self.command),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
            "memory_sample_count": self.memory_sample_count,
            "environment": None if self.environment is None else self.environment.to_record(),
            "prompt": None if self.prompt is None else self.prompt.to_record(),
            "generation": None if self.generation is None else self.generation.to_record(),
            "parse_error": self.parse_error,
        }


@dataclass(frozen=True, slots=True)
class LlamaCppThroughputComparison:
    comparable: bool
    drift_fields: tuple[str, ...]
    reason: str | None
    prompt_speedup_ratio: float | None
    generation_speedup_ratio: float | None
    peak_rss_delta_bytes: int | None
    peak_vram_delta_bytes: int | None

    def to_record(self) -> dict[str, object]:
        return {
            "comparable": self.comparable,
            "drift_fields": list(self.drift_fields),
            "reason": self.reason,
            "prompt_speedup_ratio": self.prompt_speedup_ratio,
            "generation_speedup_ratio": self.generation_speedup_ratio,
            "peak_rss_delta_bytes": self.peak_rss_delta_bytes,
            "peak_vram_delta_bytes": self.peak_vram_delta_bytes,
        }


@dataclass(frozen=True, slots=True)
class _SampledCommandResult:
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    peak_rss_bytes: int | None
    peak_vram_bytes: int | None
    sample_count: int


class _WindowsProcessMemoryCounters(ctypes.Structure):
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


def _windows_process_rss(process_id: int) -> int | None:
    try:
        windll: Any = getattr(ctypes, "windll")  # noqa: B009
        open_process = windll.kernel32.OpenProcess
        open_process.argtypes = (ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong)
        open_process.restype = ctypes.c_void_p
        get_memory = windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_WindowsProcessMemoryCounters),
            ctypes.c_ulong,
        )
        get_memory.restype = ctypes.c_bool
        close_handle = windll.kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_bool
        handle = open_process(0x0400 | 0x0010, False, process_id)
        if not handle:
            return None
        try:
            counters = _WindowsProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            ok = get_memory(handle, ctypes.byref(counters), counters.cb)
            return int(counters.WorkingSetSize) if ok else None
        finally:
            close_handle(handle)
    except (AttributeError, ctypes.ArgumentError, OSError, ValueError):
        return None


def _proc_process_rss(process_id: int) -> int | None:
    path = Path(f"/proc/{process_id}/statm")
    sysconf = getattr(os, "sysconf", None)
    if sysconf is None:
        return None
    try:
        resident_pages = int(path.read_text(encoding="ascii").split()[1])
        return resident_pages * int(sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, ValueError):
        return None


def _process_rss(process_id: int) -> int | None:
    if sys.platform == "win32":
        return _windows_process_rss(process_id)
    return _proc_process_rss(process_id)


def _run_sampled(
    command: tuple[str, ...],
    *,
    timeout_seconds: float,
    sample_interval_seconds: float,
    max_log_bytes: int,
    gpu_memory_probe: GpuProcessMemoryProbe | None,
    monotonic: Callable[[], float] = time.monotonic,
) -> _SampledCommandResult:
    with tempfile.TemporaryFile() as stdout_stream, tempfile.TemporaryFile() as stderr_stream:
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_stream,
                stderr=stderr_stream,
                shell=False,
            )
        except OSError as error:
            raise LlamaCppThroughputError(
                f"failed to execute llama-bench command {command[0]!r}: {error}"
            ) from error
        started = monotonic()
        peak_rss: int | None = None
        peak_vram: int | None = 0 if gpu_memory_probe is None else None
        samples = 0
        timed_out = False
        try:
            while process.poll() is None:
                rss = _process_rss(process.pid)
                vram = None if gpu_memory_probe is None else gpu_memory_probe(process.pid)
                peak_rss = rss if peak_rss is None else max(peak_rss, rss or 0)
                peak_vram = vram if peak_vram is None else max(peak_vram, vram or 0)
                samples += 1
                remaining = timeout_seconds - (monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    process.kill()
                    break
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=min(sample_interval_seconds, remaining))
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        if timed_out:
            process.wait()
        returncode = None if timed_out else process.returncode
        stdout, stdout_truncated = _read_bounded_log(stdout_stream, max_log_bytes)
        stderr, stderr_truncated = _read_bounded_log(stderr_stream, max_log_bytes)
    return _SampledCommandResult(
        returncode,
        timed_out,
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
        peak_rss,
        peak_vram,
        samples,
    )


def _integer(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise LlamaCppThroughputError(f"llama-bench field {key!r} must be an integer")
    return value


def _number(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LlamaCppThroughputError(f"llama-bench field {key!r} must be numeric")
    return float(value)


def _string(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise LlamaCppThroughputError(f"llama-bench field {key!r} must be text")
    return value


def _phase(
    record: Mapping[str, object],
    phase: str,
    tokens: int,
) -> LlamaCppThroughputPhase:
    raw_ns = record.get("samples_ns")
    raw_ts = record.get("samples_ts")
    if not isinstance(raw_ns, list) or not isinstance(raw_ts, list):
        raise LlamaCppThroughputError("llama-bench samples must be arrays")
    latency_samples = tuple(
        value for value in raw_ns if isinstance(value, int) and not isinstance(value, bool)
    )
    throughput_samples = tuple(
        float(value)
        for value in raw_ts
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    if len(latency_samples) != len(raw_ns) or len(throughput_samples) != len(raw_ts):
        raise LlamaCppThroughputError("llama-bench samples contain invalid values")
    return LlamaCppThroughputPhase(
        phase,
        tokens,
        _number(record, "avg_ts"),
        _number(record, "stddev_ts"),
        _integer(record, "avg_ns"),
        _integer(record, "stddev_ns"),
        latency_samples,
        throughput_samples,
    )


def _parse(
    output: str,
    config: LlamaCppThroughputConfig,
) -> tuple[
    LlamaCppBenchmarkEnvironment,
    LlamaCppThroughputPhase,
    LlamaCppThroughputPhase,
]:
    try:
        value: object = json.loads(output)
    except json.JSONDecodeError as error:
        raise LlamaCppThroughputError("llama-bench stdout is not valid JSON") from error
    if not isinstance(value, list) or len(value) != 2 or any(
        not isinstance(item, dict) for item in value
    ):
        raise LlamaCppThroughputError(
            "llama-bench must return exactly one prompt and one generation result"
        )
    records = tuple(item for item in value if isinstance(item, dict))
    prompt_record = next(
        (
            record
            for record in records
            if record.get("n_prompt") == config.prompt_tokens and record.get("n_gen") == 0
        ),
        None,
    )
    generation_record = next(
        (
            record
            for record in records
            if record.get("n_prompt") == 0
            and record.get("n_gen") == config.generation_tokens
        ),
        None,
    )
    if prompt_record is None or generation_record is None:
        raise LlamaCppThroughputError(
            "llama-bench output does not match requested prompt/generation geometry"
        )
    if any(
        len(record.get("samples_ns", ())) != config.repetitions
        for record in records
        if isinstance(record.get("samples_ns"), list)
    ):
        raise LlamaCppThroughputError(
            "llama-bench sample count does not match requested repetitions"
        )
    for key, expected in (
        ("n_batch", config.batch_size),
        ("n_ubatch", config.microbatch_size),
        ("n_threads", config.threads),
        ("n_gpu_layers", config.gpu_layers),
    ):
        if any(_integer(record, key) != expected for record in records):
            raise LlamaCppThroughputError(
                f"llama-bench output field {key!r} drifted from requested configuration"
            )
    commit = _string(prompt_record, "build_commit").lower()
    if not _revision_matches(config.expected_revision, commit):
        raise LlamaCppThroughputError(
            f"unsupported llama.cpp revision {commit}; benchmark is pinned to "
            f"{config.expected_revision}"
        )
    environment = LlamaCppBenchmarkEnvironment(
        commit,
        _integer(prompt_record, "build_number"),
        _string(prompt_record, "cpu_info"),
        _string(prompt_record, "gpu_info"),
        _string(prompt_record, "backends"),
        _string(prompt_record, "model_type"),
        _integer(prompt_record, "model_size"),
        _integer(prompt_record, "model_n_params"),
    )
    environment_fields = (
        "build_commit",
        "build_number",
        "cpu_info",
        "gpu_info",
        "backends",
        "model_type",
        "model_size",
        "model_n_params",
    )
    if any(prompt_record.get(key) != generation_record.get(key) for key in environment_fields):
        raise LlamaCppThroughputError("llama-bench phase environments do not match")
    return (
        environment,
        _phase(prompt_record, "prompt", config.prompt_tokens),
        _phase(generation_record, "generation", config.generation_tokens),
    )


def _command(
    executable: Path,
    model: Path,
    config: LlamaCppThroughputConfig,
) -> tuple[str, ...]:
    command = (
        str(executable),
        "-m",
        str(model),
        "-p",
        str(config.prompt_tokens),
        "-n",
        str(config.generation_tokens),
        "-b",
        str(config.batch_size),
        "-ub",
        str(config.microbatch_size),
        "-t",
        str(config.threads),
        "-ngl",
        str(config.gpu_layers),
        "-r",
        str(config.repetitions),
        "-o",
        "json",
    )
    return command if config.warmup else (*command, "--no-warmup")


def benchmark_gguf_throughput(
    model_path: str | Path,
    *,
    config: LlamaCppThroughputConfig | None = None,
    gpu_memory_probe: GpuProcessMemoryProbe | None = None,
) -> LlamaCppThroughputReport:
    """Benchmark GGUF prompt/decode throughput and sample child-process memory."""

    settings = config or LlamaCppThroughputConfig()
    model = Path(model_path).expanduser().resolve()
    if not model.is_file():
        raise LlamaCppThroughputError(f"GGUF model does not exist: {model}")
    if model.suffix.lower() != ".gguf":
        raise LlamaCppThroughputError("llama-bench requires a .gguf model")
    if settings.gpu_layers > 0 and gpu_memory_probe is None:
        raise LlamaCppThroughputError(
            "GPU-offloaded benchmarks require a process VRAM probe"
        )
    executable = _resolve_executable(settings.executable, settings.expected_revision)
    command = _command(executable, model, settings)
    result = _run_sampled(
        command,
        timeout_seconds=settings.timeout_seconds,
        sample_interval_seconds=settings.sample_interval_seconds,
        max_log_bytes=settings.max_log_bytes,
        gpu_memory_probe=gpu_memory_probe,
    )
    environment: LlamaCppBenchmarkEnvironment | None = None
    prompt: LlamaCppThroughputPhase | None = None
    generation: LlamaCppThroughputPhase | None = None
    parse_error: str | None = None
    if not result.timed_out and result.returncode == 0:
        try:
            environment, prompt, generation = _parse(result.stdout, settings)
        except LlamaCppThroughputError as error:
            parse_error = str(error)
    return LlamaCppThroughputReport(
        model,
        _sha256_file(model),
        settings,
        command,
        result.returncode,
        result.timed_out,
        result.stdout,
        result.stderr,
        result.stdout_truncated,
        result.stderr_truncated,
        result.peak_rss_bytes,
        result.peak_vram_bytes,
        result.sample_count,
        environment,
        prompt,
        generation,
        parse_error,
    )


def compare_gguf_throughput(
    baseline: LlamaCppThroughputReport,
    candidate: LlamaCppThroughputReport,
) -> LlamaCppThroughputComparison:
    """Return performance ratios only when runtime configuration and hardware match."""

    drift: list[str] = []
    baseline_config = baseline.config.to_record()
    candidate_config = candidate.config.to_record()
    if baseline_config != candidate_config:
        drift.extend(
            f"config.{key}"
            for key in baseline_config
            if baseline_config[key] != candidate_config[key]
        )
    if (
        baseline.environment is not None
        and candidate.environment is not None
        and baseline.environment.comparison_key() != candidate.environment.comparison_key()
    ):
        drift.append("environment")
    if drift:
        return LlamaCppThroughputComparison(
            False,
            tuple(drift),
            "throughput benchmark configuration or hardware drifted",
            None,
            None,
            None,
            None,
        )
    if not baseline.successful or not candidate.successful:
        return LlamaCppThroughputComparison(
            False,
            (),
            "both throughput reports must be successful",
            None,
            None,
            None,
            None,
        )
    assert baseline.prompt is not None and baseline.generation is not None
    assert candidate.prompt is not None and candidate.generation is not None
    return LlamaCppThroughputComparison(
        True,
        (),
        None,
        candidate.prompt.average_tokens_per_second
        / baseline.prompt.average_tokens_per_second,
        candidate.generation.average_tokens_per_second
        / baseline.generation.average_tokens_per_second,
        (
            None
            if baseline.peak_rss_bytes is None or candidate.peak_rss_bytes is None
            else candidate.peak_rss_bytes - baseline.peak_rss_bytes
        ),
        (
            None
            if baseline.peak_vram_bytes is None or candidate.peak_vram_bytes is None
            else candidate.peak_vram_bytes - baseline.peak_vram_bytes
        ),
    )

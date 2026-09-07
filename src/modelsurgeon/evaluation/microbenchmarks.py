"""Profile-bound transformer kernel and offload microbenchmark evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

MICROBENCHMARK_SCHEMA_VERSION = 1
MIN_MICROBENCHMARK_WARMUPS = 2
MIN_MICROBENCHMARK_REPETITIONS = 10
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METRIC_SPECS: dict[str, tuple[str, str]] = {
    "prefill_latency_seconds": ("s", "lower"),
    "decode_latency_seconds": ("s", "lower"),
    "throughput_tokens_per_second": ("tokens/s", "higher"),
    "bandwidth_gbps": ("GB/s", "higher"),
    "peak_ram_bytes": ("bytes", "lower"),
    "peak_vram_bytes": ("bytes", "lower"),
}


class MicrobenchmarkError(ValueError):
    """Raised when a microbenchmark partition is incomplete or unsafe."""


class MicrobenchmarkKind(StrEnum):
    GEMM = "gemm"
    ATTENTION = "attention"
    KV_CACHE = "kv_cache"
    MEMORY_COPY = "memory_copy"
    LLAMA_CPP_OFFLOAD = "llama_cpp_offload"


class MicrobenchmarkOutcome(StrEnum):
    MEASURED = "measured"
    UNSTABLE = "unstable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class MicrobenchmarkMetricState(StrEnum):
    MEASURED = "measured"
    UNAVAILABLE = "unavailable"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MicrobenchmarkError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise MicrobenchmarkError(f"{label} must be a lowercase SHA-256")
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MicrobenchmarkError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise MicrobenchmarkError(f"{label} must be finite and non-negative")
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class MicrobenchmarkShape:
    """One exact kernel/offload shape; no extrapolation is implied."""

    kind: MicrobenchmarkKind
    name: str
    dimensions: tuple[int, ...]
    dtype: str
    codec: str | None
    batch: int
    context_tokens: int
    cpu_threads: int
    gpu_offload_layers: int | None
    boundary_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.name, "microbenchmark shape name")
        _text(self.dtype, "microbenchmark dtype")
        if not self.dimensions or any(dimension <= 0 for dimension in self.dimensions):
            raise MicrobenchmarkError("microbenchmark dimensions must be positive")
        if self.batch <= 0 or self.context_tokens <= 0 or self.cpu_threads <= 0:
            raise MicrobenchmarkError("batch, context, and CPU threads must be positive")
        if self.gpu_offload_layers is not None and self.gpu_offload_layers < 0:
            raise MicrobenchmarkError("GPU offload layers must be non-negative when available")
        if not self.boundary_tags or self.boundary_tags != tuple(sorted(set(self.boundary_tags))):
            raise MicrobenchmarkError("shape boundary tags must be unique and sorted")

    @property
    def shape_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"micro_shape_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "dimensions": list(self.dimensions),
            "dtype": self.dtype,
            "codec": self.codec,
            "batch": self.batch,
            "context_tokens": self.context_tokens,
            "cpu_threads": self.cpu_threads,
            "gpu_offload_layers": self.gpu_offload_layers,
            "boundary_tags": list(self.boundary_tags),
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["shape_id"] = self.shape_id
        return record


@dataclass(frozen=True, slots=True)
class MicrobenchmarkProtocol:
    """Preregistered shape set and resource/variance limits."""

    warmups: int
    repetitions: int
    max_seconds_per_shape: float
    max_relative_dispersion: float
    shapes: tuple[MicrobenchmarkShape, ...]
    protocol_revision: str
    schema_version: int = MICROBENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MICROBENCHMARK_SCHEMA_VERSION:
            raise MicrobenchmarkError("unsupported microbenchmark schema version")
        if self.warmups < MIN_MICROBENCHMARK_WARMUPS:
            raise MicrobenchmarkError("microbenchmarks require at least two warmups")
        if self.repetitions < MIN_MICROBENCHMARK_REPETITIONS:
            raise MicrobenchmarkError("microbenchmarks require at least ten repetitions")
        if self.max_seconds_per_shape <= 0 or self.max_relative_dispersion <= 0:
            raise MicrobenchmarkError("microbenchmark budgets must be positive")
        ids = tuple(shape.shape_id for shape in self.shapes)
        if not ids or len(ids) != len(set(ids)):
            raise MicrobenchmarkError("microbenchmark shapes must be unique")
        _text(self.protocol_revision, "microbenchmark protocol revision")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "warmups": self.warmups,
            "repetitions": self.repetitions,
            "max_seconds_per_shape": self.max_seconds_per_shape,
            "max_relative_dispersion": self.max_relative_dispersion,
            "shapes": [shape.to_record() for shape in self.shapes],
            "protocol_revision": self.protocol_revision,
        }

    @property
    def protocol_id(self) -> str:
        digest = hashlib.sha256(_canonical(self.to_record()).encode()).hexdigest()
        return f"micro_protocol_{digest}"


@dataclass(frozen=True, slots=True)
class MicrobenchmarkKey:
    profile_id: str
    hardware_context_id: str
    shape_id: str
    runtime_id: str
    runtime_revision: str
    tool_revision: str
    seed: int

    def __post_init__(self) -> None:
        _text(self.profile_id, "hardware profile ID")
        _text(self.hardware_context_id, "hardware context ID")
        _text(self.shape_id, "shape ID")
        for label, value in (
            ("runtime ID", self.runtime_id),
            ("runtime revision", self.runtime_revision),
            ("tool revision", self.tool_revision),
        ):
            _text(value, label)
        if self.seed < 0:
            raise MicrobenchmarkError("microbenchmark seed must be unsigned")

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "hardware_context_id": self.hardware_context_id,
            "shape_id": self.shape_id,
            "runtime_id": self.runtime_id,
            "runtime_revision": self.runtime_revision,
            "tool_revision": self.tool_revision,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class MicrobenchmarkObservation:
    repetition: int
    metrics: Mapping[str, float]
    hardware_context_id: str
    environment_fingerprint: str | None

    def __post_init__(self) -> None:
        if self.repetition < 0:
            raise MicrobenchmarkError("microbenchmark repetitions must be non-negative")
        _text(self.hardware_context_id, "observation hardware context ID")
        if self.environment_fingerprint is not None:
            _text(self.environment_fingerprint, "observation environment fingerprint")
        if set(self.metrics) != set(_METRIC_SPECS):
            raise MicrobenchmarkError("observations require every microbenchmark metric")
        for name, value in self.metrics.items():
            _finite_nonnegative(value, f"{name} observation")

    def to_record(self) -> dict[str, object]:
        return {
            "repetition": self.repetition,
            "metrics": dict(sorted(self.metrics.items())),
            "hardware_context_id": self.hardware_context_id,
            "environment_fingerprint": self.environment_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class MicrobenchmarkMetric:
    name: str
    state: MicrobenchmarkMetricState
    value: float | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    dispersion: float | None = None
    repetitions: int = 0
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.name not in _METRIC_SPECS:
            raise MicrobenchmarkError(f"unknown microbenchmark metric: {self.name}")
        if self.state is MicrobenchmarkMetricState.MEASURED:
            if self.value is None or self.confidence_low is None or self.confidence_high is None:
                raise MicrobenchmarkError("measured metrics require value and confidence bounds")
            for value, label in (
                (self.value, "metric value"),
                (self.confidence_low, "metric confidence_low"),
                (self.confidence_high, "metric confidence_high"),
                (self.dispersion, "metric dispersion"),
            ):
                _finite_nonnegative(value, label)
            if not self.confidence_low <= self.value <= self.confidence_high:
                raise MicrobenchmarkError("metric confidence bounds must contain the value")
            if self.repetitions < MIN_MICROBENCHMARK_REPETITIONS or self.reason is not None:
                raise MicrobenchmarkError("measured metrics require ten repetitions and no reason")
        elif (
            self.value is not None
            or self.confidence_low is not None
            or self.confidence_high is not None
        ):
            raise MicrobenchmarkError("unavailable metrics cannot carry measurements")
        elif not self.reason:
            raise MicrobenchmarkError("unavailable metrics require a reason")

    @property
    def unit(self) -> str:
        return _METRIC_SPECS[self.name][0]

    @property
    def direction(self) -> str:
        return _METRIC_SPECS[self.name][1]

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction,
            "state": self.state.value,
            "value": self.value,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "dispersion": self.dispersion,
            "repetitions": self.repetitions,
            "reason": self.reason,
        }


def _metric_summary(name: str, values: tuple[float, ...]) -> MicrobenchmarkMetric:
    if len(values) < MIN_MICROBENCHMARK_REPETITIONS:
        raise MicrobenchmarkError("measured metrics require ten repetitions")
    ordered = sorted(values)
    median = ordered[(len(ordered) - 1) // 2]
    mean = sum(ordered) / len(ordered)
    dispersion = math.sqrt(sum((value - mean) ** 2 for value in ordered) / len(ordered))
    return MicrobenchmarkMetric(
        name,
        MicrobenchmarkMetricState.MEASURED,
        median,
        ordered[0],
        ordered[-1],
        dispersion,
        len(values),
    )


@dataclass(frozen=True, slots=True)
class MicrobenchmarkResult:
    key: MicrobenchmarkKey
    shape: MicrobenchmarkShape
    outcome: MicrobenchmarkOutcome
    observations: tuple[MicrobenchmarkObservation, ...] = ()
    metrics: tuple[MicrobenchmarkMetric, ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.shape.shape_id != self.key.shape_id:
            raise MicrobenchmarkError("microbenchmark key and shape IDs do not match")
        if self.outcome in {MicrobenchmarkOutcome.MEASURED, MicrobenchmarkOutcome.UNSTABLE}:
            if len(self.observations) < MIN_MICROBENCHMARK_REPETITIONS:
                raise MicrobenchmarkError("measured microbenchmarks require ten observations")
            repetitions = tuple(item.repetition for item in self.observations)
            if repetitions != tuple(sorted(set(repetitions))):
                raise MicrobenchmarkError("observation repetitions must be unique and sorted")
            if any(
                item.hardware_context_id != self.key.hardware_context_id
                for item in self.observations
            ):
                raise MicrobenchmarkError("microbenchmark changed hardware context")
            fingerprints = {item.environment_fingerprint for item in self.observations}
            if None in fingerprints and len(fingerprints) > 1:
                raise MicrobenchmarkError("environment fingerprint availability changed")
            if len(fingerprints - {None}) > 1:
                raise MicrobenchmarkError("materially drifting environment detected")
            names = tuple(item.name for item in self.metrics)
            if names != tuple(sorted(set(names))) or set(names) != set(_METRIC_SPECS):
                raise MicrobenchmarkError("results require every sorted microbenchmark metric")
            if any(item.state is not MicrobenchmarkMetricState.MEASURED for item in self.metrics):
                raise MicrobenchmarkError("timed results require measured metrics")
            if self.outcome is MicrobenchmarkOutcome.MEASURED and self.reason is not None:
                raise MicrobenchmarkError("measured results cannot carry a reason")
            if self.outcome is MicrobenchmarkOutcome.UNSTABLE:
                _text(self.reason or "", "unstable result reason")
        else:
            if self.observations or self.metrics:
                raise MicrobenchmarkError("terminal non-measured results cannot carry observations")
            _text(self.reason or "", "microbenchmark outcome reason")

    @property
    def result_id(self) -> str:
        identity = {
            "schema_version": MICROBENCHMARK_SCHEMA_VERSION,
            "key": self.key.to_record(),
            "shape": self.shape.to_record(),
            "outcome": self.outcome.value,
            "observations": [item.to_record() for item in self.observations],
            "reason": self.reason,
        }
        digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        return f"micro_result_{digest}"

    def to_record(self) -> dict[str, object]:
        return {
            "result_id": self.result_id,
            "key": self.key.to_record(),
            "shape": self.shape.to_record(),
            "outcome": self.outcome.value,
            "observations": [item.to_record() for item in self.observations],
            "metrics": [item.to_record() for item in self.metrics],
            "reason": self.reason,
        }


def build_microbenchmark_result(
    key: MicrobenchmarkKey,
    shape: MicrobenchmarkShape,
    observations: tuple[MicrobenchmarkObservation, ...],
    *,
    max_relative_dispersion: float = 0.20,
) -> MicrobenchmarkResult:
    """Summarize repeated observations and mark high-variance cells unstable."""

    if max_relative_dispersion <= 0:
        raise MicrobenchmarkError("maximum relative dispersion must be positive")
    if len(observations) < MIN_MICROBENCHMARK_REPETITIONS:
        raise MicrobenchmarkError("microbenchmark results require ten observations")
    metrics = tuple(
        _metric_summary(name, tuple(observation.metrics[name] for observation in observations))
        for name in sorted(_METRIC_SPECS)
    )
    unstable = any(
        (metric.dispersion or 0.0) / metric.value > max_relative_dispersion
        if metric.value
        else (metric.dispersion or 0.0) > 0
        for metric in metrics
    )
    return MicrobenchmarkResult(
        key,
        shape,
        MicrobenchmarkOutcome.UNSTABLE if unstable else MicrobenchmarkOutcome.MEASURED,
        observations,
        metrics,
        "repeated observations exceed the preregistered dispersion limit" if unstable else None,
    )


@dataclass(frozen=True, slots=True)
class MicrobenchmarkPartition:
    """Complete content-addressed shape partition for one profile/runtime/seed."""

    protocol: MicrobenchmarkProtocol
    profile_id: str
    hardware_context_id: str
    runtime_id: str
    runtime_revision: str
    tool_revision: str
    seed: int
    results: tuple[MicrobenchmarkResult, ...]
    schema_version: int = MICROBENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MICROBENCHMARK_SCHEMA_VERSION:
            raise MicrobenchmarkError("unsupported microbenchmark partition schema version")
        for label, value in (
            ("profile ID", self.profile_id),
            ("hardware context ID", self.hardware_context_id),
            ("runtime ID", self.runtime_id),
            ("runtime revision", self.runtime_revision),
            ("tool revision", self.tool_revision),
        ):
            _text(value, label)
        if self.seed < 0:
            raise MicrobenchmarkError("partition seed must be unsigned")
        expected = {shape.shape_id for shape in self.protocol.shapes}
        actual = {result.key.shape_id for result in self.results}
        if actual != expected or len(actual) != len(self.results):
            raise MicrobenchmarkError("partition must retain one result for every protocol shape")
        for result in self.results:
            key = result.key
            if (
                key.profile_id != self.profile_id
                or key.hardware_context_id != self.hardware_context_id
                or key.runtime_id != self.runtime_id
                or key.runtime_revision != self.runtime_revision
                or key.tool_revision != self.tool_revision
                or key.seed != self.seed
            ):
                raise MicrobenchmarkError("partition result provenance does not match partition")

    @property
    def partition_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"micro_partition_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol": self.protocol.to_record(),
            "profile_id": self.profile_id,
            "hardware_context_id": self.hardware_context_id,
            "runtime_id": self.runtime_id,
            "runtime_revision": self.runtime_revision,
            "tool_revision": self.tool_revision,
            "seed": self.seed,
            "results": [result.to_record() for result in self.results],
        }

    def validity_report(self) -> dict[str, object]:
        counts = {outcome.value: 0 for outcome in MicrobenchmarkOutcome}
        for result in self.results:
            counts[result.outcome.value] += 1
        return {
            "partition_id": self.partition_id,
            "protocol_id": self.protocol.protocol_id,
            "profile_id": self.profile_id,
            "shape_count": len(self.results),
            "outcomes": counts,
            "eligible_result_ids": [
                result.result_id
                for result in self.results
                if result.outcome is MicrobenchmarkOutcome.MEASURED
            ],
            "claim_boundary": "isolated kernel evidence; no end-to-end model speedup claim",
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["partition_id"] = self.partition_id
        record["validity"] = self.validity_report()
        return record


def _shape(
    kind: MicrobenchmarkKind,
    name: str,
    dimensions: tuple[int, ...],
    dtype: str,
    tags: tuple[str, ...],
    *,
    codec: str | None = None,
    batch: int = 1,
    context_tokens: int = 512,
    cpu_threads: int = 8,
    gpu_offload_layers: int | None = None,
) -> MicrobenchmarkShape:
    return MicrobenchmarkShape(
        kind,
        name,
        dimensions,
        dtype,
        codec,
        batch,
        context_tokens,
        cpu_threads,
        gpu_offload_layers,
        tuple(sorted(tags)),
    )


DEFAULT_MICROBENCHMARK_PROTOCOL = MicrobenchmarkProtocol(
    MIN_MICROBENCHMARK_WARMUPS,
    MIN_MICROBENCHMARK_REPETITIONS,
    300.0,
    0.20,
    (
        _shape(
            MicrobenchmarkKind.GEMM,
            "gemm_tensor_core_boundary",
            (16, 4096, 4096),
            "fp16",
            ("simd_8", "tensor_core_16"),
        ),
        _shape(
            MicrobenchmarkKind.ATTENTION,
            "attention_head_width_boundary",
            (32, 64, 80),
            "bf16",
            ("cache_line_64", "head_width_64", "head_width_80"),
        ),
        _shape(
            MicrobenchmarkKind.KV_CACHE,
            "kv_cache_context_boundary",
            (2, 32, 512, 128),
            "bf16",
            ("context_511", "context_512", "context_513"),
            context_tokens=513,
        ),
        _shape(
            MicrobenchmarkKind.MEMORY_COPY,
            "gguf_block_copy_boundary",
            (256, 257, 4096),
            "q4_k_m",
            ("cache_line_64", "gguf_block_256", "gguf_block_257"),
            codec="Q4_K_M",
        ),
        _shape(
            MicrobenchmarkKind.LLAMA_CPP_OFFLOAD,
            "offload_split_boundary",
            (4096, 4096),
            "q6_k",
            ("gpu_layers_0", "gpu_layers_1", "gpu_layers_32"),
            codec="Q6_K",
            gpu_offload_layers=1,
        ),
    ),
    "modelsurgeon-v1.3-kernel-microbenchmarks-v1",
)


def render_microbenchmark_protocol(*, format: str = "json") -> str:
    """Render the frozen shape and repetition protocol."""

    payload = {
        "schema_version": MICROBENCHMARK_SCHEMA_VERSION,
        "minimum_warmups": MIN_MICROBENCHMARK_WARMUPS,
        "minimum_repetitions": MIN_MICROBENCHMARK_REPETITIONS,
        "metrics": [{"name": name, "unit": unit} for name, (unit, _) in _METRIC_SPECS.items()],
        "protocol": DEFAULT_MICROBENCHMARK_PROTOCOL.to_record(),
        "outcomes": [item.value for item in MicrobenchmarkOutcome],
    }
    if format == "json":
        return _canonical(payload) + "\n"
    if format == "markdown":
        return (
            "\n".join(
                (
                    "# Kernel and offload microbenchmark protocol",
                    "",
                    "- Warmups: 2",
                    "- Timed repetitions: 10",
                    "- Variance above 20% relative dispersion: unstable, excluded from predictors",
                    "- Claim boundary: isolated kernel evidence only",
                )
            )
            + "\n"
        )
    raise MicrobenchmarkError("microbenchmark protocol format must be json or markdown")

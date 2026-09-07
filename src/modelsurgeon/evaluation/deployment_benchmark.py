"""Format-neutral physical deployment benchmark evidence and statistics."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from modelsurgeon.experiments.identity import canonical_identity_json

DEPLOYMENT_BENCHMARK_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DeploymentBenchmarkError(ValueError):
    """Raised when deployment evidence is incomplete or unsafe to interpret."""


class DeploymentFormat(StrEnum):
    """Physical artifact formats covered by the shared deployment contract."""

    HF = "hf"
    GGUF = "gguf"


class DeploymentOutcome(StrEnum):
    """Terminal outcome retained for every deployment benchmark cell."""

    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    TIMEOUT = "timeout"
    OOM = "oom"
    DRIFT = "drift"
    INVALID_ARTIFACT = "invalid_artifact"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class DeploymentMetricDefinition:
    """Canonical metric name and unit shared by HF and GGUF runners."""

    name: str
    unit: str


DEPLOYMENT_METRICS: tuple[DeploymentMetricDefinition, ...] = (
    DeploymentMetricDefinition("load_time_seconds", "s"),
    DeploymentMetricDefinition("prefill_tokens_per_second", "tokens/s"),
    DeploymentMetricDefinition("decode_tokens_per_second", "tokens/s"),
    DeploymentMetricDefinition("latency_seconds", "s"),
    DeploymentMetricDefinition("peak_ram_bytes", "bytes"),
    DeploymentMetricDefinition("peak_vram_bytes", "bytes"),
    DeploymentMetricDefinition("disk_bytes", "bytes"),
)
_METRIC_UNITS = {item.name: item.unit for item in DEPLOYMENT_METRICS}


def _require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeploymentBenchmarkError(f"{label} is required")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeploymentBenchmarkError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise DeploymentBenchmarkError(f"{label} must be finite and non-negative")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class DeploymentMeasurementPolicy:
    """Frozen repetition policy; the minimums prevent one-shot claims."""

    warmups: int = 2
    repetitions: int = 7
    timeout_seconds: float = 300.0
    drift_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.warmups < 2:
            raise DeploymentBenchmarkError("deployment benchmarks require at least two warmups")
        if self.repetitions < 7:
            raise DeploymentBenchmarkError(
                "deployment benchmarks require at least seven repetitions"
            )
        if self.timeout_seconds <= 0:
            raise DeploymentBenchmarkError("deployment timeout must be positive")
        if not 0 <= self.drift_tolerance < 1:
            raise DeploymentBenchmarkError("deployment drift tolerance must be in [0, 1)")

    def to_record(self) -> dict[str, object]:
        return {
            "warmups": self.warmups,
            "repetitions": self.repetitions,
            "timeout_seconds": self.timeout_seconds,
            "drift_tolerance": self.drift_tolerance,
        }


@dataclass(frozen=True, slots=True)
class DeploymentRuntimeProfile:
    """Runtime controls and hardware context retained with every result."""

    runtime: str
    cpu_threads: int = 1
    gpu_offload: int = 0
    device: str = "cpu"
    hardware: str = "unknown"
    policy: DeploymentMeasurementPolicy = DeploymentMeasurementPolicy()

    def __post_init__(self) -> None:
        _require_text(self.runtime, "runtime")
        _require_text(self.device, "device")
        _require_text(self.hardware, "hardware")
        if self.cpu_threads <= 0 or self.gpu_offload < 0:
            raise DeploymentBenchmarkError(
                "CPU threads must be positive and GPU offload non-negative"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "cpu_threads": self.cpu_threads,
            "gpu_offload": self.gpu_offload,
            "device": self.device,
            "hardware": self.hardware,
            "policy": self.policy.to_record(),
        }


@dataclass(frozen=True, slots=True)
class DeploymentArtifact:
    """Digest and size identity for a physical model artifact."""

    identifier: str
    format: DeploymentFormat
    digest: str
    size_bytes: int
    path: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.identifier, "artifact identifier")
        if not _SHA256.fullmatch(self.digest):
            raise DeploymentBenchmarkError("artifact digest must be a lowercase SHA-256")
        if self.size_bytes <= 0:
            raise DeploymentBenchmarkError("artifact size must be positive")

    @classmethod
    def from_path(
        cls, path: Path, format: DeploymentFormat, *, identifier: str | None = None
    ) -> DeploymentArtifact:
        if not path.is_file():
            raise DeploymentBenchmarkError(f"artifact is not a file: {path}")
        return cls(identifier or path.name, format, _sha256(path), path.stat().st_size, str(path))

    def to_record(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "format": self.format.value,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class DeploymentMetricSummary:
    """Median, p95, dispersion, and retained samples for one metric."""

    name: str
    unit: str
    median: float
    p95: float
    dispersion: float
    samples: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = _METRIC_UNITS.get(self.name)
        if expected is None or self.unit != expected:
            raise DeploymentBenchmarkError(f"unknown or mismatched deployment metric: {self.name}")
        if len(self.samples) < 7:
            raise DeploymentBenchmarkError("deployment metrics require seven repetitions")
        if any(not math.isfinite(value) or value < 0 for value in self.samples):
            raise DeploymentBenchmarkError(
                "deployment metric samples must be finite and non-negative"
            )
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.median, self.p95, self.dispersion)
        ):
            raise DeploymentBenchmarkError(
                "deployment metric summary must be finite and non-negative"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "median": self.median,
            "p95": self.p95,
            "dispersion": self.dispersion,
            "samples": list(self.samples),
        }


def summarize_samples(name: str, samples: Sequence[float]) -> DeploymentMetricSummary:
    """Summarize a repeated metric using deterministic rank-based p95."""

    values = tuple(_finite_nonnegative(value, f"{name} sample") for value in samples)
    if len(values) < 7:
        raise DeploymentBenchmarkError("deployment metrics require seven repetitions")
    ordered = sorted(values)
    median = ordered[(len(ordered) - 1) // 2]
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    mean = sum(values) / len(values)
    dispersion = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return DeploymentMetricSummary(
        name, _METRIC_UNITS[name], median, ordered[p95_index], dispersion, values
    )


@dataclass(frozen=True, slots=True)
class DeploymentBenchmarkRecord:
    """One immutable deployment cell, including an explicit terminal outcome."""

    artifact: DeploymentArtifact
    profile: DeploymentRuntimeProfile
    outcome: DeploymentOutcome
    metrics: tuple[DeploymentMetricSummary, ...] = ()
    command: tuple[str, ...] = ()
    reason: str | None = None
    schema_version: int = DEPLOYMENT_BENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DEPLOYMENT_BENCHMARK_SCHEMA_VERSION:
            raise DeploymentBenchmarkError("unsupported deployment benchmark schema version")
        names = tuple(item.name for item in self.metrics)
        if len(names) != len(set(names)):
            raise DeploymentBenchmarkError("deployment metric names must be unique")
        if self.outcome is DeploymentOutcome.MEASURED:
            if set(names) != set(_METRIC_UNITS):
                raise DeploymentBenchmarkError(
                    "measured deployment records require every shared metric"
                )
            if self.reason is not None:
                raise DeploymentBenchmarkError("measured deployment records cannot carry a reason")
        elif not self.reason or not self.reason.strip():
            raise DeploymentBenchmarkError("terminal deployment outcomes require a reason")

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact": self.artifact.to_record(),
            "profile": self.profile.to_record(),
            "command": list(self.command),
        }

    @property
    def record_id(self) -> str:
        digest = hashlib.sha256(
            canonical_identity_json(self._identity_record()).encode()
        ).hexdigest()
        return f"deployment_{digest}"

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record.update(
            {
                "record_id": self.record_id,
                "outcome": self.outcome.value,
                "reason": self.reason,
                "metrics": [item.to_record() for item in self.metrics],
            }
        )
        return record


def record_from_mapping(raw: Mapping[str, object]) -> DeploymentBenchmarkRecord:
    """Strictly read a persisted runner result."""

    if raw.get("schema_version") != DEPLOYMENT_BENCHMARK_SCHEMA_VERSION:
        raise DeploymentBenchmarkError("unsupported deployment benchmark schema version")
    artifact_raw = raw.get("artifact")
    profile_raw = raw.get("profile")
    if not isinstance(artifact_raw, Mapping) or not isinstance(profile_raw, Mapping):
        raise DeploymentBenchmarkError("deployment result requires artifact and profile objects")
    policy_raw = profile_raw.get("policy")
    if not isinstance(policy_raw, Mapping):
        raise DeploymentBenchmarkError("deployment profile requires a policy object")
    policy = DeploymentMeasurementPolicy(
        int(policy_raw.get("warmups", 0)),
        int(policy_raw.get("repetitions", 0)),
        float(policy_raw.get("timeout_seconds", 0)),
        float(policy_raw.get("drift_tolerance", -1)),
    )
    artifact = DeploymentArtifact(
        _require_text(artifact_raw.get("identifier"), "artifact identifier"),
        DeploymentFormat(_require_text(artifact_raw.get("format"), "artifact format")),
        _require_text(artifact_raw.get("digest"), "artifact digest"),
        int(artifact_raw.get("size_bytes", 0)),
        cast(str | None, artifact_raw.get("path")),
    )
    metrics_raw = raw.get("metrics", [])
    if not isinstance(metrics_raw, list):
        raise DeploymentBenchmarkError("deployment metrics must be an array")
    metrics = tuple(
        summarize_samples(
            _require_text(item.get("name"), f"metrics[{index}].name"),
            tuple(
                float(cast(str | int | float, value))
                for value in cast(list[object], item.get("samples", []))
            ),
        )
        for index, item in enumerate(metrics_raw)
        if isinstance(item, Mapping)
    )
    if len(metrics) != len(metrics_raw):
        raise DeploymentBenchmarkError("deployment metric entries must be objects")
    command_raw = raw.get("command", [])
    if not isinstance(command_raw, list) or not all(isinstance(item, str) for item in command_raw):
        raise DeploymentBenchmarkError("deployment command must be a string array")
    outcome = DeploymentOutcome(_require_text(raw.get("outcome"), "deployment outcome"))
    record = DeploymentBenchmarkRecord(
        artifact,
        DeploymentRuntimeProfile(
            _require_text(profile_raw.get("runtime"), "runtime"),
            int(profile_raw.get("cpu_threads", 0)),
            int(profile_raw.get("gpu_offload", -1)),
            _require_text(profile_raw.get("device"), "device"),
            _require_text(profile_raw.get("hardware"), "hardware"),
            policy,
        ),
        outcome,
        metrics,
        tuple(command_raw),
        cast(str | None, raw.get("reason")),
    )
    if raw.get("record_id") != record.record_id:
        raise DeploymentBenchmarkError("deployment record identity does not match its content")
    return record


def render_deployment_protocol(*, format: str = "json") -> str:
    """Render the frozen shared metric and outcome contract."""

    payload = {
        "schema_version": DEPLOYMENT_BENCHMARK_SCHEMA_VERSION,
        "minimum_warmups": 2,
        "minimum_repetitions": 7,
        "metrics": [{"name": item.name, "unit": item.unit} for item in DEPLOYMENT_METRICS],
        "outcomes": [item.value for item in DeploymentOutcome],
    }
    if format == "json":
        return canonical_identity_json(payload) + "\n"
    if format == "markdown":
        lines = [
            "# Physical deployment benchmark protocol",
            "",
            "| Metric | Unit |",
            "| --- | --- |",
        ]
        lines.extend(f"| `{item.name}` | `{item.unit}` |" for item in DEPLOYMENT_METRICS)
        lines.extend(["", "Minimum warmups: 2", "", "Minimum repetitions: 7", ""])
        return "\n".join(lines)
    raise DeploymentBenchmarkError("deployment protocol format must be json or markdown")

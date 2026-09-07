"""Leakage-safe hardware deployment cost dataset records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from modelsurgeon.surgery.artifact_outcome import ArtifactState, PhysicalArtifactOutcome

if TYPE_CHECKING:
    from modelsurgeon.evaluation.deployment_benchmark import DeploymentBenchmarkRecord

HARDWARE_COST_DATASET_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METRIC_UNITS = {
    "load_time_seconds": "s",
    "prefill_tokens_per_second": "tokens/s",
    "decode_tokens_per_second": "tokens/s",
    "latency_seconds": "s",
    "peak_ram_bytes": "bytes",
    "peak_vram_bytes": "bytes",
    "disk_bytes": "bytes",
}


class HardwareCostDatasetError(ValueError):
    """Raised when a hardware-cost example or split is unsafe to train on."""


class HardwareCostOutcome(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"
    UNSTABLE = "unstable"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HardwareCostDatasetError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise HardwareCostDatasetError(f"{label} must be a lowercase SHA-256")
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HardwareCostDatasetError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise HardwareCostDatasetError(f"{label} must be finite and non-negative")
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class HardwareCostMetric:
    """One target with its full repeated-run distribution retained."""

    name: str
    samples: tuple[float, ...]
    median: float
    p95: float
    dispersion: float

    def __post_init__(self) -> None:
        unit = _METRIC_UNITS.get(self.name)
        if unit is None:
            raise HardwareCostDatasetError(f"unknown hardware cost metric: {self.name}")
        if len(self.samples) < 7:
            raise HardwareCostDatasetError("hardware cost metrics require seven repetitions")
        values = tuple(_finite_nonnegative(value, f"{self.name} sample") for value in self.samples)
        ordered = sorted(values)
        expected_median = ordered[(len(ordered) - 1) // 2]
        expected_p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
        mean = sum(values) / len(values)
        expected_dispersion = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        if (
            self.median != expected_median
            or self.p95 != expected_p95
            or not math.isclose(self.dispersion, expected_dispersion, rel_tol=0, abs_tol=1e-12)
        ):
            raise HardwareCostDatasetError(f"{self.name} summary does not reconcile samples")

    @property
    def unit(self) -> str:
        return _METRIC_UNITS[self.name]

    @classmethod
    def from_samples(cls, name: str, samples: tuple[float, ...]) -> HardwareCostMetric:
        values = tuple(_finite_nonnegative(value, f"{name} sample") for value in samples)
        if len(values) < 7:
            raise HardwareCostDatasetError("hardware cost metrics require seven repetitions")
        ordered = sorted(values)
        median = ordered[(len(ordered) - 1) // 2]
        p95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
        mean = sum(values) / len(values)
        dispersion = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        return cls(name, values, median, p95, dispersion)

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "samples": list(self.samples),
            "median": self.median,
            "p95": self.p95,
            "dispersion": self.dispersion,
        }


@dataclass(frozen=True, slots=True)
class HardwareCostProvenance:
    """Benchmark and configuration provenance attached to every example."""

    benchmark_record_id: str
    protocol_revision: str
    tool_revision: str
    command: tuple[str, ...]
    configuration: Mapping[str, object]

    def __post_init__(self) -> None:
        _text(self.benchmark_record_id, "benchmark record ID")
        _text(self.protocol_revision, "benchmark protocol revision")
        _text(self.tool_revision, "benchmark tool revision")
        if not all(isinstance(item, str) and item for item in self.command):
            raise HardwareCostDatasetError("benchmark command must contain non-empty strings")
        try:
            _canonical(dict(self.configuration))
        except (TypeError, ValueError) as error:
            raise HardwareCostDatasetError(
                "benchmark configuration must be canonical JSON"
            ) from error

    def to_record(self) -> dict[str, object]:
        return {
            "benchmark_record_id": self.benchmark_record_id,
            "protocol_revision": self.protocol_revision,
            "tool_revision": self.tool_revision,
            "command": list(self.command),
            "configuration": dict(self.configuration),
        }


@dataclass(frozen=True, slots=True)
class HardwareCostExample:
    """One artifact-bound supervised cost example."""

    model_family: str
    model_revision: str
    architecture_state_id: str
    source_artifact_digest: str
    artifact_digest: str | None
    artifact_size_bytes: int | None
    hardware_profile_id: str
    hardware_context_id: str
    runtime_id: str
    runtime_revision: str
    runtime_settings: Mapping[str, object]
    seed: int
    outcome: HardwareCostOutcome
    metrics: tuple[HardwareCostMetric, ...]
    provenance: HardwareCostProvenance | None
    lineage_group_id: str
    reason: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("model family", self.model_family),
            ("model revision", self.model_revision),
            ("architecture state ID", self.architecture_state_id),
            ("hardware profile ID", self.hardware_profile_id),
            ("hardware context ID", self.hardware_context_id),
            ("runtime ID", self.runtime_id),
            ("runtime revision", self.runtime_revision),
            ("lineage group ID", self.lineage_group_id),
        ):
            _text(value, label)
        _digest(self.source_artifact_digest, "source artifact digest")
        if self.artifact_digest is not None:
            _digest(self.artifact_digest, "artifact digest")
        if self.artifact_size_bytes is not None and self.artifact_size_bytes <= 0:
            raise HardwareCostDatasetError("artifact size must be positive when available")
        if self.seed < 0:
            raise HardwareCostDatasetError("dataset seed must be unsigned")
        try:
            _canonical(dict(self.runtime_settings))
        except (TypeError, ValueError) as error:
            raise HardwareCostDatasetError("runtime settings must be canonical JSON") from error
        names = tuple(item.name for item in self.metrics)
        if names != tuple(sorted(set(names))):
            raise HardwareCostDatasetError("hardware cost metrics must be unique and sorted")
        if self.outcome is HardwareCostOutcome.MEASURED:
            if (
                self.artifact_digest is None
                or self.artifact_size_bytes is None
                or self.provenance is None
            ):
                raise HardwareCostDatasetError("measured examples require artifact and provenance")
            if set(names) != set(_METRIC_UNITS):
                raise HardwareCostDatasetError("measured examples require every deployment metric")
            if self.reason is not None:
                raise HardwareCostDatasetError("measured examples cannot carry a reason")
        else:
            if (
                self.metrics
                or self.artifact_digest is not None
                or self.artifact_size_bytes is not None
            ):
                raise HardwareCostDatasetError(
                    "non-measured examples cannot carry partial measurements"
                )
            _text(self.reason or "", "dataset outcome reason")

    @property
    def example_id(self) -> str:
        identity = {
            "schema_version": HARDWARE_COST_DATASET_SCHEMA_VERSION,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "architecture_state_id": self.architecture_state_id,
            "source_artifact_digest": self.source_artifact_digest,
            "artifact_digest": self.artifact_digest,
            "artifact_size_bytes": self.artifact_size_bytes,
            "hardware_profile_id": self.hardware_profile_id,
            "hardware_context_id": self.hardware_context_id,
            "runtime_id": self.runtime_id,
            "runtime_revision": self.runtime_revision,
            "runtime_settings": dict(self.runtime_settings),
            "seed": self.seed,
            "outcome": self.outcome.value,
            "metrics": [item.to_record() for item in self.metrics],
            "provenance": None if self.provenance is None else self.provenance.to_record(),
            "lineage_group_id": self.lineage_group_id,
            "reason": self.reason,
        }
        digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        return f"hardware_cost_{digest}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": HARDWARE_COST_DATASET_SCHEMA_VERSION,
            "example_id": self.example_id,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "architecture_state_id": self.architecture_state_id,
            "source_artifact_digest": self.source_artifact_digest,
            "artifact_digest": self.artifact_digest,
            "artifact_size_bytes": self.artifact_size_bytes,
            "hardware_profile_id": self.hardware_profile_id,
            "hardware_context_id": self.hardware_context_id,
            "runtime_id": self.runtime_id,
            "runtime_revision": self.runtime_revision,
            "runtime_settings": dict(self.runtime_settings),
            "seed": self.seed,
            "outcome": self.outcome.value,
            "metrics": [item.to_record() for item in self.metrics],
            "provenance": None if self.provenance is None else self.provenance.to_record(),
            "lineage_group_id": self.lineage_group_id,
            "reason": self.reason,
        }


def build_hardware_cost_example(
    *,
    model_family: str,
    model_revision: str,
    architecture_state_id: str,
    source_artifact_digest: str,
    physical_outcome: PhysicalArtifactOutcome,
    deployment: DeploymentBenchmarkRecord,
    hardware_profile_id: str,
    hardware_context_id: str,
    runtime_id: str,
    runtime_revision: str,
    runtime_settings: Mapping[str, object],
    seed: int,
    provenance: HardwareCostProvenance,
    lineage_group_id: str | None = None,
) -> HardwareCostExample:
    """Join physical lineage and deployment evidence without dropping repetitions."""

    from modelsurgeon.evaluation.deployment_benchmark import DeploymentOutcome

    if physical_outcome.artifact.state is not ArtifactState.COMPLETE:
        raise HardwareCostDatasetError("cost examples require a complete physical artifact")
    if deployment.outcome is not DeploymentOutcome.MEASURED:
        raise HardwareCostDatasetError("cost examples require a measured deployment record")
    if physical_outcome.artifact.digest != deployment.artifact.digest:
        raise HardwareCostDatasetError("physical and deployment artifact digests do not match")
    if physical_outcome.artifact.size_bytes != deployment.artifact.size_bytes:
        raise HardwareCostDatasetError("physical and deployment artifact sizes do not match")
    if provenance.benchmark_record_id != deployment.record_id:
        raise HardwareCostDatasetError("provenance does not name the deployment record")
    metrics = tuple(
        HardwareCostMetric.from_samples(item.name, item.samples)
        for item in sorted(deployment.metrics, key=lambda item: item.name)
    )
    group = (
        lineage_group_id
        or hashlib.sha256(
            _canonical(
                {
                    "model_revision": model_revision,
                    "source_artifact_digest": source_artifact_digest,
                    "hardware_profile_id": hardware_profile_id,
                }
            ).encode()
        ).hexdigest()
    )
    return HardwareCostExample(
        model_family,
        model_revision,
        architecture_state_id,
        source_artifact_digest,
        physical_outcome.artifact.digest,
        physical_outcome.artifact.size_bytes,
        hardware_profile_id,
        hardware_context_id,
        runtime_id,
        runtime_revision,
        runtime_settings,
        seed,
        HardwareCostOutcome.MEASURED,
        metrics,
        provenance,
        f"lineage_{group}",
    )


@dataclass(frozen=True, slots=True)
class HardwareCostSplit:
    """Group-level split with no duplicated lineage across partitions."""

    train_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]
    test_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = (
            set(self.train_group_ids),
            set(self.validation_group_ids),
            set(self.test_group_ids),
        )
        if any(not group for group in groups):
            raise HardwareCostDatasetError("every cost dataset split must contain groups")
        if any(
            len(group) != len(values)
            for group, values in zip(
                groups,
                (self.train_group_ids, self.validation_group_ids, self.test_group_ids),
                strict=True,
            )
        ):
            raise HardwareCostDatasetError("split group IDs must be unique")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise HardwareCostDatasetError("cost dataset lineage groups leak across splits")

    def to_record(self) -> dict[str, object]:
        return {
            "train_group_ids": list(self.train_group_ids),
            "validation_group_ids": list(self.validation_group_ids),
            "test_group_ids": list(self.test_group_ids),
        }


@dataclass(frozen=True, slots=True)
class HardwareCostDataset:
    """Validated, profile-partitionable training data with retained outcomes."""

    examples: tuple[HardwareCostExample, ...]
    split: HardwareCostSplit
    dataset_revision: str
    schema_version: int = HARDWARE_COST_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != HARDWARE_COST_DATASET_SCHEMA_VERSION:
            raise HardwareCostDatasetError("unsupported hardware cost dataset schema version")
        _text(self.dataset_revision, "hardware cost dataset revision")
        ids = tuple(example.example_id for example in self.examples)
        if not ids or len(ids) != len(set(ids)):
            raise HardwareCostDatasetError("cost dataset examples must be non-empty and unique")
        all_groups = (
            set(self.split.train_group_ids)
            | set(self.split.validation_group_ids)
            | set(self.split.test_group_ids)
        )
        if any(example.lineage_group_id not in all_groups for example in self.examples):
            raise HardwareCostDatasetError("every example lineage must appear in exactly one split")

    @property
    def dataset_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"hardware_cost_dataset_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_revision": self.dataset_revision,
            "split": self.split.to_record(),
            "examples": [example.to_record() for example in self.examples],
        }

    def partition_by_profile(self) -> dict[str, tuple[HardwareCostExample, ...]]:
        partitions: dict[str, list[HardwareCostExample]] = {}
        for example in self.examples:
            partitions.setdefault(example.hardware_profile_id, []).append(example)
        return {key: tuple(value) for key, value in sorted(partitions.items())}

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["dataset_id"] = self.dataset_id
        record["coverage"] = {
            "examples": len(self.examples),
            "profiles": len(self.partition_by_profile()),
            "outcomes": {
                outcome.value: sum(example.outcome is outcome for example in self.examples)
                for outcome in HardwareCostOutcome
            },
        }
        return record

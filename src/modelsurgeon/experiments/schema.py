"""Versioned experiment and supervised mutation-example records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from modelsurgeon.experiments.hardware import HardwareInventory
from modelsurgeon.features.schema import FeatureRecord, PrecisionProvenance
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.serialization import MutationRunRecord

EXPERIMENT_SCHEMA_VERSION: Literal[1] = 1
MUTATION_EXAMPLE_SCHEMA_VERSION: Literal[1] = 1


class ExperimentSchemaError(ValueError):
    """Raised when an experiment dataset record is incomplete or inconsistent."""


class MetricState(StrEnum):
    """Why a metric does or does not have a numeric value."""

    ABSENT = "absent"
    SKIPPED = "skipped"
    FAILED = "failed"
    MEASURED = "measured"


class ExperimentOutcomeKind(StrEnum):
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


class QuantizationControlKind(StrEnum):
    NONE = "none"
    MATCHED_REQUANTIZATION = "matched_requantization"


@dataclass(frozen=True, slots=True)
class ModelTarget:
    identifier: str
    revision: str
    family: str
    format: str
    parameter_count: int | None = None
    quantization: str | None = None

    def __post_init__(self) -> None:
        if any(not value for value in (self.identifier, self.revision, self.family, self.format)):
            raise ExperimentSchemaError("model target identity fields are required")
        if self.parameter_count is not None and self.parameter_count <= 0:
            raise ExperimentSchemaError("model parameter count must be positive when known")
        if self.quantization is not None and not self.quantization:
            raise ExperimentSchemaError("model quantization cannot be blank")

    def to_record(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "family": self.family,
            "format": self.format,
            "parameter_count": self.parameter_count,
            "quantization": self.quantization,
        }


@dataclass(frozen=True, slots=True)
class DatasetTarget:
    identifier: str
    revision: str
    split: str
    manifest_id: str
    tokenizer: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        values = (
            self.identifier,
            self.revision,
            self.split,
            self.manifest_id,
            self.tokenizer,
            self.tokenizer_revision,
        )
        if any(not value for value in values):
            raise ExperimentSchemaError("dataset target identity fields are required")

    def to_record(self) -> dict[str, str]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "split": self.split,
            "manifest_id": self.manifest_id,
            "tokenizer": self.tokenizer,
            "tokenizer_revision": self.tokenizer_revision,
        }


@dataclass(frozen=True, slots=True)
class VersionContext:
    tool_revision: str
    config_digest: str
    evaluator_version: str
    feature_schema_version: int
    mutation_record_schema_version: int

    def __post_init__(self) -> None:
        if any(not value for value in (self.tool_revision, self.config_digest, self.evaluator_version)):
            raise ExperimentSchemaError("tool, config, and evaluator versions are required")
        if self.feature_schema_version <= 0 or self.mutation_record_schema_version <= 0:
            raise ExperimentSchemaError("referenced schema versions must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "tool_revision": self.tool_revision,
            "config_digest": self.config_digest,
            "evaluator_version": self.evaluator_version,
            "feature_schema_version": self.feature_schema_version,
            "mutation_record_schema_version": self.mutation_record_schema_version,
        }


@dataclass(frozen=True, slots=True)
class SeedContext:
    experiment_seed: int
    data_seed: int
    mutation_seed: int

    def __post_init__(self) -> None:
        for value in (self.experiment_seed, self.data_seed, self.mutation_seed):
            if isinstance(value, bool) or value < 0 or value >= 1 << 64:
                raise ExperimentSchemaError("experiment seeds must be unsigned 64-bit integers")

    def to_record(self) -> dict[str, int]:
        return {
            "experiment_seed": self.experiment_seed,
            "data_seed": self.data_seed,
            "mutation_seed": self.mutation_seed,
        }


@dataclass(frozen=True, slots=True)
class StageTiming:
    stage: str
    wall_seconds: float
    cpu_seconds: float | None = None
    tokens: int | None = None
    candidates: int | None = None

    def __post_init__(self) -> None:
        if not self.stage:
            raise ExperimentSchemaError("timing stage is required")
        numeric = (self.wall_seconds,) if self.cpu_seconds is None else (
            self.wall_seconds,
            self.cpu_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in numeric):
            raise ExperimentSchemaError("stage timings must be finite and non-negative")
        for value in (self.tokens, self.candidates):
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ExperimentSchemaError("stage throughput counts must be non-negative")

    def to_record(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "tokens": self.tokens,
            "candidates": self.candidates,
        }


@dataclass(frozen=True, slots=True)
class MetricObservation:
    name: str
    state: MetricState
    value: float | None = None
    unit: str | None = None
    reason: str | None = None
    precision: PrecisionProvenance | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ExperimentSchemaError("metric name is required")
        if self.unit is not None and not self.unit:
            raise ExperimentSchemaError("metric unit cannot be blank")
        if self.state is MetricState.MEASURED:
            if self.value is None or not math.isfinite(self.value):
                raise ExperimentSchemaError("measured metrics require a finite value")
            if self.reason is not None:
                raise ExperimentSchemaError("measured metrics cannot carry a missingness reason")
            return
        if self.value is not None:
            raise ExperimentSchemaError("non-measured metrics cannot carry numeric values")
        if not self.reason:
            raise ExperimentSchemaError("absent, skipped, and failed metrics require a reason")
        if self.precision is not None:
            raise ExperimentSchemaError("non-measured metrics cannot claim precision provenance")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "state": self.state.value,
            "value": self.value,
            "unit": self.unit,
            "reason": self.reason,
            "precision": None if self.precision is None else self.precision.to_record(),
        }


@dataclass(frozen=True, slots=True)
class QuantizationControl:
    kind: QuantizationControlKind
    complete: bool
    codecs: tuple[str, ...] = ()
    affected_components: tuple[ComponentId, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.codecs != tuple(sorted(set(self.codecs))) or any(not item for item in self.codecs):
            raise ExperimentSchemaError("quantization control codecs must be unique and canonical")
        if self.affected_components != tuple(sorted(set(self.affected_components))):
            raise ExperimentSchemaError(
                "quantization control components must be unique and canonical"
            )
        if self.seed is not None and (
            isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64
        ):
            raise ExperimentSchemaError("quantization control seed must be unsigned 64-bit")
        if self.kind is QuantizationControlKind.NONE:
            if self.codecs or self.affected_components or self.seed is not None or not self.complete:
                raise ExperimentSchemaError("no-op quantization control cannot carry codec work")
        elif not self.codecs or not self.affected_components or self.seed is None:
            raise ExperimentSchemaError(
                "matched requantization control requires codecs, components, and seed"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "complete": self.complete,
            "codecs": list(self.codecs),
            "affected_components": [str(item) for item in self.affected_components],
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    kind: ExperimentOutcomeKind
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind is ExperimentOutcomeKind.SUCCEEDED:
            if self.reason is not None:
                raise ExperimentSchemaError("successful outcomes cannot carry a failure reason")
        elif not self.reason:
            raise ExperimentSchemaError("rejected and failed outcomes require a reason")

    def to_record(self) -> dict[str, str | None]:
        return {"kind": self.kind.value, "reason": self.reason}


def _validate_named_metrics(
    metrics: tuple[MetricObservation, ...],
    label: str,
) -> None:
    names = tuple(item.name for item in metrics)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ExperimentSchemaError(f"{label} metrics must use unique canonical names")


def _validate_timings(timings: tuple[StageTiming, ...]) -> None:
    stages = tuple(item.stage for item in timings)
    if stages != tuple(sorted(stages)) or len(stages) != len(set(stages)):
        raise ExperimentSchemaError("stage timings must use unique canonical stage names")


def _feature_identity(feature: FeatureRecord) -> tuple[str, str, str, str]:
    return (
        str(feature.component_id),
        feature.name,
        feature.extractor,
        feature.extractor_version,
    )


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    run_id: str
    experiment_id: str
    attempt_id: str
    model: ModelTarget
    dataset: DatasetTarget
    components: tuple[ComponentId, ...]
    mutation: MutationRunRecord
    baseline_metrics: tuple[MetricObservation, ...]
    post_metrics: tuple[MetricObservation, ...]
    delta_metrics: tuple[MetricObservation, ...]
    outcome: ExperimentOutcome
    hardware: HardwareInventory
    versions: VersionContext
    seeds: SeedContext
    timings: tuple[StageTiming, ...] = ()
    quantization_control: QuantizationControl | None = None
    schema_version: Literal[1] = EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ExperimentSchemaError(
                f"unsupported experiment schema version {self.schema_version}"
            )
        if any(not value for value in (self.run_id, self.experiment_id, self.attempt_id)):
            raise ExperimentSchemaError("run, experiment, and attempt identities are required")
        if self.components != tuple(sorted(set(self.components))) or not self.components:
            raise ExperimentSchemaError("experiment components must be non-empty and canonical")
        if self.components != self.mutation.plan.affected_components:
            raise ExperimentSchemaError(
                "experiment components must match the canonical mutation affected set"
            )
        if self.mutation.provenance.input_revision != self.model.revision:
            raise ExperimentSchemaError("mutation and model input revisions must match")
        _validate_named_metrics(self.baseline_metrics, "baseline")
        _validate_named_metrics(self.post_metrics, "post")
        _validate_named_metrics(self.delta_metrics, "delta")
        _validate_timings(self.timings)

    def to_record(self, *, redact_local_paths: bool = True) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "attempt_id": self.attempt_id,
            "model": self.model.to_record(),
            "dataset": self.dataset.to_record(),
            "components": [str(item) for item in self.components],
            "mutation": self.mutation.to_record(redact_local_paths=redact_local_paths),
            "baseline_metrics": [item.to_record() for item in self.baseline_metrics],
            "post_metrics": [item.to_record() for item in self.post_metrics],
            "delta_metrics": [item.to_record() for item in self.delta_metrics],
            "outcome": self.outcome.to_record(),
            "hardware": self.hardware.to_record(),
            "versions": self.versions.to_record(),
            "seeds": self.seeds.to_record(),
            "timings": [item.to_record() for item in self.timings],
            "quantization_control": (
                None if self.quantization_control is None else self.quantization_control.to_record()
            ),
        }


@dataclass(frozen=True, slots=True)
class MutationExampleRecord:
    example_id: str
    experiment_id: str
    model: ModelTarget
    dataset: DatasetTarget
    components: tuple[ComponentId, ...]
    mutation: MutationRunRecord
    pre_mutation_features: tuple[FeatureRecord, ...]
    baseline_metrics: tuple[MetricObservation, ...]
    post_metrics: tuple[MetricObservation, ...]
    delta_metrics: tuple[MetricObservation, ...]
    outcome: ExperimentOutcome
    hardware: HardwareInventory
    versions: VersionContext
    seeds: SeedContext
    timings: tuple[StageTiming, ...] = ()
    quantization_control: QuantizationControl | None = None
    schema_version: Literal[1] = MUTATION_EXAMPLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MUTATION_EXAMPLE_SCHEMA_VERSION:
            raise ExperimentSchemaError(
                f"unsupported mutation-example schema version {self.schema_version}"
            )
        if not self.example_id or not self.experiment_id:
            raise ExperimentSchemaError("example and experiment identities are required")
        if self.components != tuple(sorted(set(self.components))) or not self.components:
            raise ExperimentSchemaError("example components must be non-empty and canonical")
        if self.components != self.mutation.plan.affected_components:
            raise ExperimentSchemaError(
                "example components must match the canonical mutation affected set"
            )
        if self.mutation.provenance.input_revision != self.model.revision:
            raise ExperimentSchemaError("mutation example input revisions must match")
        feature_keys = tuple(_feature_identity(item) for item in self.pre_mutation_features)
        if feature_keys != tuple(sorted(feature_keys)) or len(feature_keys) != len(
            set(feature_keys)
        ):
            raise ExperimentSchemaError(
                "pre-mutation features must use unique canonical identities"
            )
        _validate_named_metrics(self.baseline_metrics, "baseline")
        _validate_named_metrics(self.post_metrics, "post")
        _validate_named_metrics(self.delta_metrics, "delta")
        _validate_timings(self.timings)

    @property
    def mutation_id(self) -> str:
        return self.mutation.mutation_id

    def to_record(self, *, redact_local_paths: bool = True) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "example_id": self.example_id,
            "experiment_id": self.experiment_id,
            "mutation_id": self.mutation_id,
            "model": self.model.to_record(),
            "dataset": self.dataset.to_record(),
            "components": [str(item) for item in self.components],
            "mutation": self.mutation.to_record(redact_local_paths=redact_local_paths),
            "pre_mutation_features": [item.to_record() for item in self.pre_mutation_features],
            "baseline_metrics": [item.to_record() for item in self.baseline_metrics],
            "post_metrics": [item.to_record() for item in self.post_metrics],
            "delta_metrics": [item.to_record() for item in self.delta_metrics],
            "outcome": self.outcome.to_record(),
            "hardware": self.hardware.to_record(),
            "versions": self.versions.to_record(),
            "seeds": self.seeds.to_record(),
            "timings": [item.to_record() for item in self.timings],
            "quantization_control": (
                None if self.quantization_control is None else self.quantization_control.to_record()
            ),
        }

"""Immutable cumulative physical artifact outcomes and reconciliation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

PHYSICAL_ARTIFACT_OUTCOME_SCHEMA_VERSION = 1


class PhysicalOutcomeError(ValueError):
    """Raised when physical lineage or publication evidence is incomplete."""


class ArtifactFormat(StrEnum):
    HUGGINGFACE = "safetensors"
    GGUF = "GGUF"


class ArtifactState(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PhysicalOutcomeError(f"{label} is required")


def _digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PhysicalOutcomeError(f"{label} must be a lowercase SHA-256")


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise PhysicalOutcomeError(f"{label} must be positive")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class ArchitectureState:
    source_parameters: int
    active_parameters: int
    tensor_count: int
    architecture_digest: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.source_parameters, "source parameter count"),
            (self.active_parameters, "active parameter count"),
            (self.tensor_count, "tensor count"),
        ):
            _positive(value, label)
        if self.active_parameters > self.source_parameters:
            raise PhysicalOutcomeError("active parameters cannot exceed source parameters")
        _digest(self.architecture_digest, "architecture digest")

    def to_record(self) -> dict[str, object]:
        return {
            "source_parameters": self.source_parameters,
            "active_parameters": self.active_parameters,
            "tensor_count": self.tensor_count,
            "architecture_digest": self.architecture_digest,
        }


@dataclass(frozen=True, slots=True)
class TensorOutcome:
    locator: str
    old_shape: tuple[int, ...]
    new_shape: tuple[int, ...]
    old_storage_bytes: int
    new_storage_bytes: int
    component_identity_digest: str

    def __post_init__(self) -> None:
        _text(self.locator, "tensor locator")
        if not self.old_shape or not self.new_shape:
            raise PhysicalOutcomeError("tensor shapes are required")
        if any(size <= 0 for size in (*self.old_shape, *self.new_shape)):
            raise PhysicalOutcomeError("tensor shapes must be positive")
        _positive(self.old_storage_bytes, "old tensor storage")
        _positive(self.new_storage_bytes, "new tensor storage")
        _digest(self.component_identity_digest, "component identity digest")

    @property
    def storage_delta(self) -> int:
        return self.new_storage_bytes - self.old_storage_bytes

    def to_record(self) -> dict[str, object]:
        return {
            "locator": self.locator,
            "old_shape": list(self.old_shape),
            "new_shape": list(self.new_shape),
            "old_storage_bytes": self.old_storage_bytes,
            "new_storage_bytes": self.new_storage_bytes,
            "storage_delta": self.storage_delta,
            "component_identity_digest": self.component_identity_digest,
        }


@dataclass(frozen=True, slots=True)
class FormatArtifactEvidence:
    format: ArtifactFormat
    state: ArtifactState
    digest: str | None
    size_bytes: int | None
    format_schema_version: int
    reloadable: bool | None
    generation_smoke: bool | None
    details: Mapping[str, object]
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.format_schema_version <= 0:
            raise PhysicalOutcomeError("format schema version must be positive")
        if not isinstance(self.details, Mapping):
            raise PhysicalOutcomeError("format details must be a mapping")
        if self.state is ArtifactState.COMPLETE:
            if self.digest is None or self.size_bytes is None:
                raise PhysicalOutcomeError("complete artifacts require digest and size")
            _digest(self.digest, "artifact digest")
            _positive(self.size_bytes, "artifact size")
            if self.reloadable is not True or self.generation_smoke is not True:
                raise PhysicalOutcomeError(
                    "complete artifacts require reload and generation evidence"
                )
            if self.reason is not None:
                raise PhysicalOutcomeError("complete artifacts cannot carry a failure reason")
        else:
            if self.digest is not None or self.size_bytes is not None:
                raise PhysicalOutcomeError("incomplete artifacts cannot publish digest or size")
            _text(self.reason or "", "incomplete artifact reason")

    def to_record(self) -> dict[str, object]:
        return {
            "format": self.format.value,
            "state": self.state.value,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "format_schema_version": self.format_schema_version,
            "reloadable": self.reloadable,
            "generation_smoke": self.generation_smoke,
            "details": dict(self.details),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DeploymentEvidence:
    runtime: str
    prompt_tokens_per_second: float | None
    decode_tokens_per_second: float | None
    peak_ram_bytes: int | None
    peak_vram_bytes: int | None

    def __post_init__(self) -> None:
        _text(self.runtime, "deployment runtime")
        values = (
            self.prompt_tokens_per_second,
            self.decode_tokens_per_second,
        )
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in values):
            raise PhysicalOutcomeError("deployment speeds must be finite and non-negative")
        for value, label in (
            (self.peak_ram_bytes, "peak RAM"),
            (self.peak_vram_bytes, "peak VRAM"),
        ):
            if value is not None and value < 0:
                raise PhysicalOutcomeError(f"{label} must be non-negative")

    def to_record(self) -> dict[str, object]:
        return {
            "runtime": self.runtime,
            "prompt_tokens_per_second": self.prompt_tokens_per_second,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "peak_ram_bytes": self.peak_ram_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
        }


@dataclass(frozen=True, slots=True)
class PhysicalArtifactOutcome:
    source_outcome_id: str
    parent_outcome_id: str | None
    mutation_id: str
    architecture: ArchitectureState
    tensor_outcomes: tuple[TensorOutcome, ...]
    artifact: FormatArtifactEvidence
    deployment: DeploymentEvidence | None
    stage_parameter_delta: int
    stage_storage_delta: int
    cumulative_parameter_delta: int
    cumulative_storage_delta: int
    identity_remap_record: Mapping[str, object]
    schema_version: int = PHYSICAL_ARTIFACT_OUTCOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("source outcome ID", self.source_outcome_id),
            ("mutation ID", self.mutation_id),
        ):
            _text(value, label)
        if self.parent_outcome_id is not None:
            _text(self.parent_outcome_id, "parent outcome ID")
        if not self.tensor_outcomes:
            raise PhysicalOutcomeError("physical outcomes require tensor evidence")
        locators = tuple(item.locator for item in self.tensor_outcomes)
        if locators != tuple(sorted(set(locators))):
            raise PhysicalOutcomeError("tensor outcomes must be uniquely ordered by locator")
        if self.stage_storage_delta != sum(item.storage_delta for item in self.tensor_outcomes):
            raise PhysicalOutcomeError("stage storage delta does not reconcile tensor evidence")
        if self.artifact.state is ArtifactState.COMPLETE and self.deployment is None:
            raise PhysicalOutcomeError("complete artifacts require deployment evidence")
        if not isinstance(self.identity_remap_record, Mapping):
            raise PhysicalOutcomeError("identity remap evidence must be a mapping")
        if self.schema_version != PHYSICAL_ARTIFACT_OUTCOME_SCHEMA_VERSION:
            raise PhysicalOutcomeError("unknown physical outcome schema version")

    @property
    def outcome_id(self) -> str:
        return "outcome_" + hashlib.sha256(
            _canonical(self._identity_record()).encode()
        ).hexdigest()

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_outcome_id": self.source_outcome_id,
            "parent_outcome_id": self.parent_outcome_id,
            "mutation_id": self.mutation_id,
            "architecture": self.architecture.to_record(),
            "tensor_outcomes": [item.to_record() for item in self.tensor_outcomes],
            "artifact": self.artifact.to_record(),
            "deployment": None if self.deployment is None else self.deployment.to_record(),
            "stage_parameter_delta": self.stage_parameter_delta,
            "stage_storage_delta": self.stage_storage_delta,
            "cumulative_parameter_delta": self.cumulative_parameter_delta,
            "cumulative_storage_delta": self.cumulative_storage_delta,
            "identity_remap": dict(self.identity_remap_record),
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["outcome_id"] = self.outcome_id
        return record


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    outcome_id: str
    cumulative_parameter_delta: int
    cumulative_storage_delta: int
    stage_count: int
    valid: bool

    def to_record(self) -> dict[str, object]:
        return {
            "outcome_id": self.outcome_id,
            "cumulative_parameter_delta": self.cumulative_parameter_delta,
            "cumulative_storage_delta": self.cumulative_storage_delta,
            "stage_count": self.stage_count,
            "valid": self.valid,
        }


@dataclass(frozen=True, slots=True)
class PhysicalOutcomeSequence:
    source_parameters: int
    source_storage_bytes: int
    outcomes: tuple[PhysicalArtifactOutcome, ...]

    def __post_init__(self) -> None:
        _positive(self.source_parameters, "sequence source parameters")
        _positive(self.source_storage_bytes, "sequence source storage")
        previous_id: str | None = None
        parameter_total = 0
        storage_total = 0
        for outcome in self.outcomes:
            if outcome.parent_outcome_id != previous_id:
                raise PhysicalOutcomeError("outcome parent does not match the preceding stage")
            parameter_total += outcome.stage_parameter_delta
            storage_total += outcome.stage_storage_delta
            if outcome.cumulative_parameter_delta != parameter_total:
                raise PhysicalOutcomeError(
                    "cumulative parameter delta double-counts or omits a stage"
                )
            if outcome.cumulative_storage_delta != storage_total:
                raise PhysicalOutcomeError(
                    "cumulative storage delta double-counts or omits a stage"
                )
            previous_id = outcome.outcome_id

    def reconcile(self) -> ReconciliationReport:
        if not self.outcomes:
            raise PhysicalOutcomeError("cannot reconcile an empty physical outcome sequence")
        final = self.outcomes[-1]
        return ReconciliationReport(
            final.outcome_id,
            final.cumulative_parameter_delta,
            final.cumulative_storage_delta,
            len(self.outcomes),
            True,
        )


def _record_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise PhysicalOutcomeError(f"{label} must be an object")
    return dict(value)


def _record_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise PhysicalOutcomeError(f"{label} must be a string")
    return value


def _record_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise PhysicalOutcomeError(f"{label} must be an integer")
    return value


def _record_float(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PhysicalOutcomeError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise PhysicalOutcomeError(f"{label} must be finite")
    return float(value)


def _record_optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _record_string(value, label)


def _record_optional_bool(value: object, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise PhysicalOutcomeError(f"{label} must be a boolean")
    return value


def import_physical_outcome(raw: Mapping[str, object]) -> PhysicalArtifactOutcome:
    """Parse one complete typed outcome and verify its content address."""
    if raw.get("schema_version") != PHYSICAL_ARTIFACT_OUTCOME_SCHEMA_VERSION:
        raise PhysicalOutcomeError("unknown physical outcome schema version")
    architecture_raw = _record_object(raw.get("architecture"), "architecture")
    architecture = ArchitectureState(
        _record_int(architecture_raw.get("source_parameters"), "architecture.source_parameters"),
        _record_int(architecture_raw.get("active_parameters"), "architecture.active_parameters"),
        _record_int(architecture_raw.get("tensor_count"), "architecture.tensor_count"),
        _record_string(
            architecture_raw.get("architecture_digest"),
            "architecture.architecture_digest",
        ),
    )
    tensor_values = raw.get("tensor_outcomes")
    if not isinstance(tensor_values, list):
        raise PhysicalOutcomeError("tensor_outcomes must be an array")
    tensors: list[TensorOutcome] = []
    for index, value in enumerate(tensor_values):
        item = _record_object(value, f"tensor_outcomes[{index}]")
        old_shape = item.get("old_shape")
        new_shape = item.get("new_shape")
        if (
            not isinstance(old_shape, list)
            or not isinstance(new_shape, list)
            or not all(
                isinstance(size, int) and not isinstance(size, bool)
                for size in (*old_shape, *new_shape)
            )
        ):
            raise PhysicalOutcomeError("tensor shapes must be integer arrays")
        tensors.append(
            TensorOutcome(
                _record_string(item.get("locator"), f"tensor_outcomes[{index}].locator"),
                tuple(old_shape),
                tuple(new_shape),
                    _record_int(
                        item.get("old_storage_bytes"),
                        f"tensor_outcomes[{index}].old_storage_bytes",
                    ),
                    _record_int(
                        item.get("new_storage_bytes"),
                        f"tensor_outcomes[{index}].new_storage_bytes",
                    ),
                _record_string(
                    item.get("component_identity_digest"),
                    f"tensor_outcomes[{index}].component_identity_digest",
                ),
            )
        )
    artifact_raw = _record_object(raw.get("artifact"), "artifact")
    artifact = FormatArtifactEvidence(
        ArtifactFormat(_record_string(artifact_raw.get("format"), "artifact.format")),
        ArtifactState(_record_string(artifact_raw.get("state"), "artifact.state")),
        _record_optional_string(artifact_raw.get("digest"), "artifact.digest"),
        None
        if artifact_raw.get("size_bytes") is None
        else _record_int(artifact_raw.get("size_bytes"), "artifact.size_bytes"),
        _record_int(artifact_raw.get("format_schema_version"), "artifact.format_schema_version"),
        _record_optional_bool(artifact_raw.get("reloadable"), "artifact.reloadable"),
        _record_optional_bool(artifact_raw.get("generation_smoke"), "artifact.generation_smoke"),
        _record_object(artifact_raw.get("details"), "artifact.details"),
        _record_optional_string(artifact_raw.get("reason"), "artifact.reason"),
    )
    deployment_raw = raw.get("deployment")
    deployment: DeploymentEvidence | None
    if deployment_raw is None:
        deployment = None
    else:
        item = _record_object(deployment_raw, "deployment")
        deployment = DeploymentEvidence(
            _record_string(item.get("runtime"), "deployment.runtime"),
            None
            if item.get("prompt_tokens_per_second") is None
            else _record_float(
                item["prompt_tokens_per_second"],
                "deployment.prompt_tokens_per_second",
            ),
            None
            if item.get("decode_tokens_per_second") is None
            else _record_float(
                item["decode_tokens_per_second"],
                "deployment.decode_tokens_per_second",
            ),
            None
            if item.get("peak_ram_bytes") is None
            else _record_int(item.get("peak_ram_bytes"), "deployment.peak_ram_bytes"),
            None
            if item.get("peak_vram_bytes") is None
            else _record_int(item.get("peak_vram_bytes"), "deployment.peak_vram_bytes"),
        )
    identity = _record_object(raw.get("identity_remap"), "identity_remap")
    outcome = PhysicalArtifactOutcome(
        _record_string(raw.get("source_outcome_id"), "source_outcome_id"),
        _record_optional_string(raw.get("parent_outcome_id"), "parent_outcome_id"),
        _record_string(raw.get("mutation_id"), "mutation_id"),
        architecture,
        tuple(tensors),
        artifact,
        deployment,
        _record_int(raw.get("stage_parameter_delta"), "stage_parameter_delta"),
        _record_int(raw.get("stage_storage_delta"), "stage_storage_delta"),
        _record_int(raw.get("cumulative_parameter_delta"), "cumulative_parameter_delta"),
        _record_int(raw.get("cumulative_storage_delta"), "cumulative_storage_delta"),
        identity,
    )
    if raw.get("outcome_id") != outcome.outcome_id:
        raise PhysicalOutcomeError("imported outcome ID does not match its content")
    return outcome


def build_example_physical_outcomes() -> tuple[PhysicalArtifactOutcome, PhysicalArtifactOutcome]:
    """Return one validated HF and native-GGUF fixture for import/reconciliation tests."""
    common_tensor = TensorOutcome(
        "model.layers.0.mlp.down_proj.weight",
        (100, 10),
        (90, 10),
        4000,
        3600,
        "c" * 64,
    )
    deployment = DeploymentEvidence("fixture-runtime", 10.0, 20.0, 1000, 2000)
    return (
        PhysicalArtifactOutcome(
            "source_hf",
            None,
            "mutation_hf",
            ArchitectureState(1000, 900, 1, "a" * 64),
            (common_tensor,),
            FormatArtifactEvidence(
                ArtifactFormat.HUGGINGFACE,
                ArtifactState.COMPLETE,
                "b" * 64,
                3600,
                1,
                True,
                True,
                {"index": "fixture.safetensors.index.json"},
            ),
            deployment,
            -100,
            -400,
            -100,
            -400,
            {"mappings": []},
        ),
        PhysicalArtifactOutcome(
            "source_gguf",
            None,
            "mutation_gguf",
            ArchitectureState(1000, 900, 1, "d" * 64),
            (common_tensor,),
            FormatArtifactEvidence(
                ArtifactFormat.GGUF,
                ArtifactState.COMPLETE,
                "e" * 64,
                3600,
                1,
                True,
                True,
                {"gguf_type": "Q8_0", "alignment": 32},
            ),
            deployment,
            -100,
            -400,
            -100,
            -400,
            {"mappings": []},
        ),
    )

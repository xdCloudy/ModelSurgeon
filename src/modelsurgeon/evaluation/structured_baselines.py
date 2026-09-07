"""Pinned structured-baseline matrix and physical artifact reconciliation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from modelsurgeon.adapters import CompetitorIdentity, ModelFormat
from modelsurgeon.evaluation.unstructured_baselines import (
    BaselineBudget,
    BaselineHardware,
    BaselineModel,
    CalibrationCorpus,
)

STRUCTURED_BASELINE_SCHEMA_VERSION = 1


class StructuredBaselineError(ValueError):
    """Raised when a structured baseline contract cannot be reconciled."""


class StructuredMethod(StrEnum):
    LLM_PRUNER = "llm_pruner"
    SLICE_GPT = "slice_gpt"
    SHORT_GPT = "short_gpt"
    MINITRON = "minitron"


class StructuredOperation(StrEnum):
    WIDTH = "width"
    HEAD = "head"
    DEPTH = "depth"
    ROTATION_LOW_RANK = "rotation_low_rank"


class StructuredOutcome(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class StructuredCompatibility(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StructuredBaselineError(f"{label} is required")


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise StructuredBaselineError(
            "structured baseline identity must be canonical JSON"
        ) from error


@dataclass(frozen=True, slots=True)
class StructuredMethodSpec:
    """Independent upstream card; methods are never treated as interchangeable."""

    method: StructuredMethod
    identity: CompetitorIdentity
    operations: tuple[StructuredOperation, ...]
    supported_families: tuple[str, ...]
    model_format: ModelFormat
    license_declared: bool
    recovery_required: bool

    def __post_init__(self) -> None:
        if not self.operations or not self.supported_families:
            raise StructuredBaselineError(
                "structured method requires finite capability declarations"
            )
        if len(set(self.operations)) != len(self.operations):
            raise StructuredBaselineError("structured method operations must be unique")
        if not self.license_declared and self.identity.license != "NO_DECLARED_LICENSE":
            raise StructuredBaselineError("undeclared licenses must be explicit")
        if self.recovery_required is not True:
            raise StructuredBaselineError("structured baselines require an explicit recovery stage")

    def to_record(self) -> dict[str, object]:
        return {
            "method": self.method.value,
            "identity": self.identity.to_record(),
            "operations": [operation.value for operation in self.operations],
            "supported_families": list(self.supported_families),
            "model_format": self.model_format.value,
            "license_declared": self.license_declared,
            "recovery_required": self.recovery_required,
        }


@dataclass(frozen=True, slots=True)
class PhysicalCompressionTarget:
    """Requested physical compression target, not a quality-only proxy."""

    fraction: float
    unit: str = "fraction_of_source_parameters"

    def __post_init__(self) -> None:
        if not 0 < self.fraction < 1 or not math.isfinite(self.fraction):
            raise StructuredBaselineError(
                "physical compression target must be finite and between zero and one"
            )

    def to_record(self) -> dict[str, object]:
        return {"fraction": self.fraction, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class ArtifactReconciliation:
    """Requested/achieved structure and artifact facts required for a claim."""

    requested_fraction: float
    achieved_fraction: float | None
    parameters_before: int
    parameters_after: int | None
    file_size_before: int
    file_size_after: int | None
    artifact_sha256: str | None
    reloadable: bool | None
    generation_smoke: bool | None

    def __post_init__(self) -> None:
        if self.parameters_before <= 0 or self.file_size_before <= 0:
            raise StructuredBaselineError("artifact reconciliation needs positive source sizes")
        if not 0 < self.requested_fraction < 1:
            raise StructuredBaselineError("requested compression fraction is invalid")
        measured = (
            self.achieved_fraction,
            self.parameters_after,
            self.file_size_after,
            self.artifact_sha256,
            self.reloadable,
            self.generation_smoke,
        )
        if any(value is None for value in measured) and any(
            value is not None for value in measured
        ):
            raise StructuredBaselineError("artifact reconciliation must be complete or unavailable")
        if self.achieved_fraction is not None:
            if not 0 <= self.achieved_fraction < 1:
                raise StructuredBaselineError("achieved compression fraction is invalid")
            if self.parameters_after is None or self.parameters_after <= 0:
                raise StructuredBaselineError("measured artifact needs a positive parameter count")
            if self.file_size_after is None or self.file_size_after <= 0:
                raise StructuredBaselineError("measured artifact needs a positive file size")
            if not self.artifact_sha256 or len(self.artifact_sha256) != 64:
                raise StructuredBaselineError("measured artifact needs a SHA-256 digest")
            if self.reloadable is not True or self.generation_smoke is not True:
                raise StructuredBaselineError(
                    "measured structured artifacts must reload and generate"
                )

    def to_record(self) -> dict[str, object]:
        return {
            "requested_fraction": self.requested_fraction,
            "achieved_fraction": self.achieved_fraction,
            "parameters_before": self.parameters_before,
            "parameters_after": self.parameters_after,
            "file_size_before": self.file_size_before,
            "file_size_after": self.file_size_after,
            "artifact_sha256": self.artifact_sha256,
            "reloadable": self.reloadable,
            "generation_smoke": self.generation_smoke,
        }


@dataclass(frozen=True, slots=True)
class StructuredEvidenceCell:
    """A method/model/operation/target/seed result with explicit failure states."""

    model: BaselineModel
    method: StructuredMethod
    operation: StructuredOperation
    target: PhysicalCompressionTarget
    seed: int
    corpus: CalibrationCorpus
    hardware: BaselineHardware
    budget: BaselineBudget
    outcome: StructuredOutcome
    reconciliation: ArtifactReconciliation
    reason: str
    runtime_seconds: float | None = None
    gpu_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise StructuredBaselineError("structured baseline seed must be unsigned")
        _text(self.reason, "structured outcome reason")
        if self.outcome is StructuredOutcome.MEASURED:
            if self.reconciliation.achieved_fraction is None:
                raise StructuredBaselineError(
                    "measured structured cells need artifact reconciliation"
                )
            if self.runtime_seconds is None or self.gpu_seconds is None:
                raise StructuredBaselineError("measured structured cells need runtime metrics")
        elif self.runtime_seconds is not None or self.gpu_seconds is not None:
            raise StructuredBaselineError(
                "unmeasured structured cells cannot claim runtime metrics"
            )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": STRUCTURED_BASELINE_SCHEMA_VERSION,
            "model": self.model.to_record(),
            "method": self.method.value,
            "operation": self.operation.value,
            "target": self.target.to_record(),
            "seed": self.seed,
            "corpus": self.corpus.to_record(),
            "hardware": self.hardware.to_record(),
            "budget": self.budget.to_record(),
        }

    @property
    def cell_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"structured_{digest}"

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record.update(
            {
                "cell_id": self.cell_id,
                "outcome": self.outcome.value,
                "reconciliation": self.reconciliation.to_record(),
                "reason": self.reason,
                "runtime_seconds": self.runtime_seconds,
                "gpu_seconds": self.gpu_seconds,
            }
        )
        return record


@dataclass(frozen=True, slots=True)
class StructuredCompatibilityCell:
    model_rung: str
    method: StructuredMethod
    operation: StructuredOperation
    model_format: ModelFormat
    outcome: StructuredCompatibility
    reason: str

    def __post_init__(self) -> None:
        for label, value in (
            ("compatibility model rung", self.model_rung),
            ("compatibility reason", self.reason),
        ):
            _text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "model_rung": self.model_rung,
            "method": self.method.value,
            "operation": self.operation.value,
            "model_format": self.model_format.value,
            "outcome": self.outcome.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class StructuredBaselineProtocol:
    models: tuple[BaselineModel, ...]
    methods: tuple[StructuredMethodSpec, ...]
    targets: tuple[PhysicalCompressionTarget, ...]
    operations: tuple[StructuredOperation, ...]
    seeds: tuple[int, ...]
    corpus: CalibrationCorpus
    hardware: BaselineHardware
    budget: BaselineBudget
    compatibility: tuple[StructuredCompatibilityCell, ...]
    cells: tuple[StructuredEvidenceCell, ...]

    def __post_init__(self) -> None:
        if len({model.family for model in self.models}) < 2:
            raise StructuredBaselineError("structured protocol requires two model families")
        if tuple(target.fraction for target in self.targets) != (0.1, 0.2):
            raise StructuredBaselineError("structured protocol must freeze 10% and 20% targets")
        if len(self.seeds) < 3 or self.seeds != tuple(sorted(set(self.seeds))):
            raise StructuredBaselineError("structured protocol requires three sorted unique seeds")
        if set(spec.method for spec in self.methods) != set(StructuredMethod):
            raise StructuredBaselineError("structured protocol must include four baseline methods")
        expected = {
            (model.rung, spec.method, operation, target.fraction, seed)
            for model, spec, operation, target, seed in product(
                self.models, self.methods, self.operations, self.targets, self.seeds
            )
        }
        actual = {
            (cell.model.rung, cell.method, cell.operation, cell.target.fraction, cell.seed)
            for cell in self.cells
        }
        if actual != expected or len(actual) != len(self.cells):
            raise StructuredBaselineError("structured protocol must retain every matched cell")
        if any(
            cell.corpus != self.corpus
            or cell.hardware != self.hardware
            or cell.budget != self.budget
            for cell in self.cells
        ):
            raise StructuredBaselineError(
                "structured cells must share corpus, hardware, and budget"
            )
        expected_compatibility = {
            (model.rung, spec.method, operation, model_format)
            for model, spec, operation, model_format in product(
                self.models,
                self.methods,
                self.operations,
                (ModelFormat.SAFETENSORS, ModelFormat.GGUF),
            )
        }
        actual_compatibility = {
            (item.model_rung, item.method, item.operation, item.model_format)
            for item in self.compatibility
        }
        if actual_compatibility != expected_compatibility:
            raise StructuredBaselineError("structured compatibility matrix is incomplete")

    @property
    def protocol_id(self) -> str:
        digest = hashlib.sha256(_canonical(self.to_identity_record()).encode()).hexdigest()
        return f"structured_protocol_{digest}"

    def to_identity_record(self) -> dict[str, object]:
        return {
            "schema_version": STRUCTURED_BASELINE_SCHEMA_VERSION,
            "models": [model.to_record() for model in self.models],
            "methods": [method.to_record() for method in self.methods],
            "targets": [target.to_record() for target in self.targets],
            "operations": [operation.value for operation in self.operations],
            "seeds": list(self.seeds),
            "corpus": self.corpus.to_record(),
            "hardware": self.hardware.to_record(),
            "budget": self.budget.to_record(),
            "compatibility": [item.to_record() for item in self.compatibility],
            "cells": [cell.to_record() for cell in self.cells],
        }

    def to_record(self) -> dict[str, object]:
        record = self.to_identity_record()
        record["protocol_id"] = self.protocol_id
        return record


def build_default_structured_baseline_protocol() -> StructuredBaselineProtocol:
    """Build the four-method physical-compression matrix without running baselines."""

    models = (
        BaselineModel(
            "100M",
            "HuggingFaceTB/SmolLM2-135M",
            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            "llama",
        ),
        BaselineModel(
            "500M",
            "Qwen/Qwen2.5-0.5B",
            "060db6499f32faf8b98477b0a26969ef7d8b9987",
            "qwen",
        ),
    )
    methods = (
        StructuredMethodSpec(
            StructuredMethod.LLM_PRUNER,
            CompetitorIdentity(
                "LLM-Pruner",
                "128a07d977f9b205d60ab14cfbc6a78f8a8e39d2",
                "Apache-2.0",
                "https://github.com/horseee/LLM-Pruner",
            ),
            (StructuredOperation.WIDTH, StructuredOperation.HEAD, StructuredOperation.DEPTH),
            ("llama", "qwen"),
            ModelFormat.SAFETENSORS,
            True,
            True,
        ),
        StructuredMethodSpec(
            StructuredMethod.SLICE_GPT,
            CompetitorIdentity("SliceGPT", "954700a71c599e2e5f67632865b288f007326241", "MIT", "https://github.com/AbhinavDutta/slicegpt"),
            (StructuredOperation.WIDTH, StructuredOperation.ROTATION_LOW_RANK),
            ("llama", "qwen"),
            ModelFormat.SAFETENSORS,
            True,
            True,
        ),
        StructuredMethodSpec(
            StructuredMethod.SHORT_GPT,
            CompetitorIdentity("ShortGPT", "78d9615fdcae6d90368832bd0a86c49c323549b9", "MIT", "https://github.com/sramshetty/ShortGPT"),
            (StructuredOperation.DEPTH,),
            ("llama",),
            ModelFormat.SAFETENSORS,
            True,
            True,
        ),
        StructuredMethodSpec(
            StructuredMethod.MINITRON,
            CompetitorIdentity(
                "Minitron",
                "fed740c91886155e1f272159ef662536bb673593",
                "NO_DECLARED_LICENSE",
                "https://github.com/NVlabs/Minitron",
            ),
            (StructuredOperation.WIDTH, StructuredOperation.HEAD, StructuredOperation.DEPTH),
            ("llama", "qwen"),
            ModelFormat.SAFETENSORS,
            False,
            True,
        ),
    )
    operations = tuple(StructuredOperation)
    corpus = CalibrationCorpus(
        "wikitext-2-raw-v1",
        "modelsurgeon-benchmark-datasets-v1.1",
        "test",
        "CC-BY-SA-4.0",
        100_000,
        "tokenizer-revision-bound-to-model-checkpoint",
    )
    hardware = BaselineHardware(
        "consumer-reference",
        "pinned-host-cpu",
        "pinned-host-gpu",
        "77e3b4f8bc2f2aedd78202a01b4bee2c7942a298",
    )
    budget = BaselineBudget(100_000, 100_000, 3_600.0, 24.0)
    targets = (PhysicalCompressionTarget(0.1), PhysicalCompressionTarget(0.2))
    compatibility = tuple(
        StructuredCompatibilityCell(
            model.rung,
            spec.method,
            operation,
            model_format,
            StructuredCompatibility.UNSUPPORTED
            if model_format is ModelFormat.GGUF
            or not spec.license_declared
            or model.family not in spec.supported_families
            or operation not in spec.operations
            else StructuredCompatibility.UNKNOWN,
            "GGUF is outside the published checkpoint contract"
            if model_format is ModelFormat.GGUF
            else "license is not declared for this upstream source"
            if not spec.license_declared
            else "pinned executable and artifact smoke are required before support is claimed"
            if model.family in spec.supported_families and operation in spec.operations
            else "method does not declare this model-family and operation combination",
        )
        for model, spec, operation, model_format in product(
            models, methods, operations, (ModelFormat.SAFETENSORS, ModelFormat.GGUF)
        )
    )
    cells = tuple(
        StructuredEvidenceCell(
            model,
            spec.method,
            operation,
            target,
            seed,
            corpus,
            hardware,
            budget,
            StructuredOutcome.UNSUPPORTED,
            ArtifactReconciliation(target.fraction, None, 1, None, 1, None, None, None, None),
            "pinned upstream executable is not bundled; retain incompatibility or "
            "execute through the adapter before claiming a measured artifact",
        )
        for model, spec, operation, target, seed in product(
            models, methods, operations, targets, (11, 23, 47)
        )
    )
    return StructuredBaselineProtocol(
        models,
        methods,
        targets,
        operations,
        (11, 23, 47),
        corpus,
        hardware,
        budget,
        compatibility,
        cells,
    )


DEFAULT_STRUCTURED_BASELINE_PROTOCOL = build_default_structured_baseline_protocol()


def render_structured_baseline_protocol(
    protocol: StructuredBaselineProtocol = DEFAULT_STRUCTURED_BASELINE_PROTOCOL,
) -> str:
    return _canonical(protocol.to_record()) + "\n"

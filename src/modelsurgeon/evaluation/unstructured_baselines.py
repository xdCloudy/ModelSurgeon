"""Revision-pinned Wanda/SparseGPT baseline protocol and evidence contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from modelsurgeon.adapters import (
    CompetitorBudget,
    CompetitorIdentity,
    ModelFormat,
    SubprocessCompetitorAdapter,
)
from modelsurgeon.evaluation.model_ladder import (
    PERMISSIVE_MODEL_LADDER,
    ModelEvaluationLadder,
    ModelLadderTarget,
)

UNSTRUCTURED_BASELINE_SCHEMA_VERSION = 1


class UnstructuredBaselineError(ValueError):
    """Raised when a matched baseline protocol or evidence cell is invalid."""


class UnstructuredMethod(StrEnum):
    WANDA = "wanda"
    SPARSEGPT = "sparsegpt"


class CompatibilityOutcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class BaselineOutcome(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise UnstructuredBaselineError(f"{label} is required")


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise UnstructuredBaselineError("baseline identity must be canonical JSON") from error


@dataclass(frozen=True, slots=True)
class UnstructuredMethodSpec:
    """Source and invocation identity for one published external baseline."""

    method: UnstructuredMethod
    identity: CompetitorIdentity
    model_format: ModelFormat
    operation: str
    calibration_required: bool
    structured_speedup_claim: bool = False

    def __post_init__(self) -> None:
        _text(self.operation, "baseline operation")
        if not self.calibration_required:
            raise UnstructuredBaselineError("unstructured baselines require calibration provenance")
        if self.structured_speedup_claim:
            raise UnstructuredBaselineError(
                "unstructured masks cannot claim structured deployment speedup"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "method": self.method.value,
            "identity": self.identity.to_record(),
            "model_format": self.model_format.value,
            "operation": self.operation,
            "calibration_required": self.calibration_required,
            "structured_speedup_claim": self.structured_speedup_claim,
        }


@dataclass(frozen=True, slots=True)
class CalibrationCorpus:
    """Identical calibration identity bound to every matched method cell."""

    identifier: str
    revision: str
    split: str
    license: str
    token_count: int
    tokenizer_revision: str

    def __post_init__(self) -> None:
        for label, value in (
            ("calibration identifier", self.identifier),
            ("calibration revision", self.revision),
            ("calibration split", self.split),
            ("calibration license", self.license),
            ("tokenizer revision", self.tokenizer_revision),
        ):
            _text(value, label)
        if self.token_count <= 0:
            raise UnstructuredBaselineError("calibration token_count must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "split": self.split,
            "license": self.license,
            "token_count": self.token_count,
            "tokenizer_revision": self.tokenizer_revision,
        }


@dataclass(frozen=True, slots=True)
class SparsityTarget:
    """Unstructured target expressed as a fraction of removable weights."""

    fraction: float
    pattern: str = "unstructured"

    def __post_init__(self) -> None:
        if not 0 < self.fraction < 1 or not math.isfinite(self.fraction):
            raise UnstructuredBaselineError(
                "sparsity fraction must be finite and between zero and one"
            )
        if self.pattern != "unstructured":
            raise UnstructuredBaselineError("this protocol only admits unstructured targets")

    @property
    def label(self) -> str:
        return f"{self.fraction:.0%}"

    def to_record(self) -> dict[str, object]:
        return {"fraction": self.fraction, "unit": "proportion", "pattern": self.pattern}


@dataclass(frozen=True, slots=True)
class BaselineHardware:
    """Pinned hardware/evaluator identity used for matched comparisons."""

    platform: str
    cpu: str
    accelerator: str
    evaluator_revision: str

    def __post_init__(self) -> None:
        for label, value in (
            ("hardware platform", self.platform),
            ("hardware CPU", self.cpu),
            ("hardware accelerator", self.accelerator),
            ("evaluator revision", self.evaluator_revision),
        ):
            _text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "platform": self.platform,
            "cpu": self.cpu,
            "accelerator": self.accelerator,
            "evaluator_revision": self.evaluator_revision,
        }


@dataclass(frozen=True, slots=True)
class BaselineBudget:
    """Equal optimization/evaluation ceilings for every method cell."""

    optimization_tokens: int
    evaluation_tokens: int
    max_wall_seconds: float
    max_gpu_memory_gib: float

    def __post_init__(self) -> None:
        if self.optimization_tokens <= 0 or self.evaluation_tokens <= 0:
            raise UnstructuredBaselineError("baseline token budgets must be positive")
        if self.max_wall_seconds <= 0 or self.max_gpu_memory_gib <= 0:
            raise UnstructuredBaselineError("baseline resource budgets must be positive")

    def to_record(self) -> dict[str, int | float]:
        return {
            "optimization_tokens": self.optimization_tokens,
            "evaluation_tokens": self.evaluation_tokens,
            "max_wall_seconds": self.max_wall_seconds,
            "max_gpu_memory_gib": self.max_gpu_memory_gib,
        }


@dataclass(frozen=True, slots=True)
class BaselineMetric:
    """A measured or explicitly unavailable metric with units and aggregation."""

    name: str
    unit: str
    aggregation: str
    value: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("baseline metric name", self.name),
            ("baseline metric unit", self.unit),
            ("baseline metric aggregation", self.aggregation),
        ):
            _text(value, label)
        if self.value is None:
            _text(self.reason or "", "unavailable baseline metric reason")
        elif not math.isfinite(self.value):
            raise UnstructuredBaselineError("baseline metrics must be finite")
        elif self.reason is not None:
            raise UnstructuredBaselineError(
                "measured baseline metrics cannot carry an unavailable reason"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BaselineModel:
    """A selected ladder target retained in the baseline identity."""

    rung: str
    identifier: str
    revision: str
    family: str

    @classmethod
    def from_target(cls, target: ModelLadderTarget) -> BaselineModel:
        return cls(target.rung, target.identifier, target.revision, target.family.value)

    def to_record(self) -> dict[str, str]:
        return {
            "rung": self.rung,
            "identifier": self.identifier,
            "revision": self.revision,
            "family": self.family,
        }


@dataclass(frozen=True, slots=True)
class BaselineEvidenceCell:
    """One matched method/model/sparsity/seed result, including negative states."""

    model: BaselineModel
    method: UnstructuredMethod
    sparsity: SparsityTarget
    seed: int
    corpus: CalibrationCorpus
    hardware: BaselineHardware
    budget: BaselineBudget
    outcome: BaselineOutcome
    metrics: tuple[BaselineMetric, ...]
    reason: str
    artifact_sha256: str | None = None
    artifact_size_bytes: int | None = None
    schema_version: int = UNSTRUCTURED_BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != UNSTRUCTURED_BASELINE_SCHEMA_VERSION:
            raise UnstructuredBaselineError("unsupported unstructured baseline schema version")
        if self.seed < 0:
            raise UnstructuredBaselineError("baseline seed must be unsigned")
        _text(self.reason, "baseline outcome reason")
        names = tuple(metric.name for metric in self.metrics)
        if len(names) != len(set(names)):
            raise UnstructuredBaselineError("baseline metric names must be unique")
        if self.outcome is BaselineOutcome.MEASURED:
            if not self.artifact_sha256 or self.artifact_size_bytes is None:
                raise UnstructuredBaselineError("measured baseline cells require artifact evidence")
            if len(self.artifact_sha256) != 64 or self.artifact_size_bytes <= 0:
                raise UnstructuredBaselineError("measured baseline artifact evidence is invalid")
        elif self.artifact_sha256 is not None or self.artifact_size_bytes is not None:
            raise UnstructuredBaselineError("non-measured baseline cells cannot claim artifacts")

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_record(),
            "method": self.method.value,
            "sparsity": self.sparsity.to_record(),
            "seed": self.seed,
            "corpus": self.corpus.to_record(),
            "hardware": self.hardware.to_record(),
            "budget": self.budget.to_record(),
        }

    @property
    def cell_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"baseline_{digest}"

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record.update(
            {
                "cell_id": self.cell_id,
                "outcome": self.outcome.value,
                "metrics": [metric.to_record() for metric in self.metrics],
                "reason": self.reason,
                "artifact_sha256": self.artifact_sha256,
                "artifact_size_bytes": self.artifact_size_bytes,
            }
        )
        return record


@dataclass(frozen=True, slots=True)
class BaselineCompatibility:
    """Matrix row that prevents unsupported formats from disappearing."""

    model_rung: str
    method: UnstructuredMethod
    model_format: ModelFormat
    outcome: CompatibilityOutcome
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
            "model_format": self.model_format.value,
            "outcome": self.outcome.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class UnstructuredBaselineProtocol:
    """Complete matched Wanda/SparseGPT preregistration and compatibility matrix."""

    models: tuple[BaselineModel, ...]
    methods: tuple[UnstructuredMethodSpec, ...]
    targets: tuple[SparsityTarget, ...]
    seeds: tuple[int, ...]
    corpus: CalibrationCorpus
    hardware: BaselineHardware
    budget: BaselineBudget
    comparison_controls: tuple[str, ...]
    compatibility: tuple[BaselineCompatibility, ...]
    cells: tuple[BaselineEvidenceCell, ...]
    schema_version: int = UNSTRUCTURED_BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != UNSTRUCTURED_BASELINE_SCHEMA_VERSION:
            raise UnstructuredBaselineError("unsupported baseline protocol schema version")
        if len(self.models) < 2 or len({model.family for model in self.models}) < 2:
            raise UnstructuredBaselineError("baseline protocol requires two model families")
        if tuple(self.seeds) != tuple(sorted(set(self.seeds))) or len(self.seeds) < 3:
            raise UnstructuredBaselineError("baseline protocol requires three sorted unique seeds")
        if any(seed < 0 for seed in self.seeds):
            raise UnstructuredBaselineError("baseline seeds must be unsigned")
        method_ids = tuple(spec.method for spec in self.methods)
        if set(method_ids) != set(UnstructuredMethod):
            raise UnstructuredBaselineError("protocol must include Wanda and SparseGPT")
        if self.comparison_controls != ("magnitude_mean_absolute", "seeded_random"):
            raise UnstructuredBaselineError(
                "protocol must retain magnitude and seeded-random comparison controls"
            )
        if tuple(target.fraction for target in self.targets) != (0.2, 0.5):
            raise UnstructuredBaselineError("protocol must freeze 20% and 50% sparsity targets")
        expected = {
            (model.rung, method, target.fraction, seed)
            for model, method, target, seed in product(
                self.models, method_ids, self.targets, self.seeds
            )
        }
        actual = {
            (cell.model.rung, cell.method, cell.sparsity.fraction, cell.seed)
            for cell in self.cells
        }
        if actual != expected or len(actual) != len(self.cells):
            raise UnstructuredBaselineError("baseline protocol must retain every matched cell")
        if any(
            cell.corpus != self.corpus
            or cell.hardware != self.hardware
            or cell.budget != self.budget
            for cell in self.cells
        ):
            raise UnstructuredBaselineError("matched cells must share corpus, hardware, and budget")
        expected_compatibility = {
            (model.rung, method.method, model_format)
            for model, method, model_format in product(
                self.models,
                self.methods,
                (ModelFormat.SAFETENSORS, ModelFormat.GGUF),
            )
        }
        actual_compatibility = {
            (item.model_rung, item.method, item.model_format) for item in self.compatibility
        }
        if actual_compatibility != expected_compatibility:
            raise UnstructuredBaselineError(
                "compatibility matrix must cover every model, method, and format"
            )

    @property
    def protocol_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"baseline_protocol_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "models": [model.to_record() for model in self.models],
            "methods": [method.to_record() for method in self.methods],
            "targets": [target.to_record() for target in self.targets],
            "seeds": list(self.seeds),
            "corpus": self.corpus.to_record(),
            "hardware": self.hardware.to_record(),
            "budget": self.budget.to_record(),
            "comparison_controls": list(self.comparison_controls),
            "compatibility": [item.to_record() for item in self.compatibility],
            "cells": [cell.to_record() for cell in self.cells],
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["protocol_id"] = self.protocol_id
        return record


def build_baseline_adapter(
    spec: UnstructuredMethodSpec,
    *,
    version_command: tuple[str, ...],
    command_template: tuple[str, ...],
    budget: CompetitorBudget,
) -> SubprocessCompetitorAdapter:
    """Build a subprocess adapter while keeping method provenance typed."""

    return SubprocessCompetitorAdapter(
        spec.identity,
        model_format=spec.model_format,
        operations=frozenset({spec.operation}),
        version_command=version_command,
        command_template=command_template,
        budget=budget,
        config={"method": spec.method.value, "revision": spec.identity.revision},
    )


def build_default_unstructured_baseline_protocol(
    ladder: ModelEvaluationLadder = PERMISSIVE_MODEL_LADDER,
) -> UnstructuredBaselineProtocol:
    """Build a two-family, two-method, two-target, three-seed baseline matrix."""

    selected_targets = (ladder.targets[0], ladder.targets[2])
    models = tuple(BaselineModel.from_target(target) for target in selected_targets)
    methods = (
        UnstructuredMethodSpec(
            UnstructuredMethod.WANDA,
            CompetitorIdentity(
                "Wanda",
                "8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6",
                "MIT",
                "https://github.com/locuslab/wanda",
            ),
            ModelFormat.SAFETENSORS,
            "unstructured_prune",
            True,
        ),
        UnstructuredMethodSpec(
            UnstructuredMethod.SPARSEGPT,
            CompetitorIdentity(
                "SparseGPT",
                "147d2159dc4f3e9f73e47b32c04d7b3708f44436",
                "Apache-2.0",
                "https://github.com/IST-DASLab/sparsegpt",
            ),
            ModelFormat.SAFETENSORS,
            "unstructured_prune",
            True,
        ),
    )
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
    compatibility = tuple(
        BaselineCompatibility(
            model.rung,
            spec.method,
            model_format,
            CompatibilityOutcome.UNKNOWN
            if model_format is ModelFormat.SAFETENSORS
            else CompatibilityOutcome.UNSUPPORTED,
            "external executable and artifact smoke are required before support is claimed"
            if model_format is ModelFormat.SAFETENSORS
            else "GGUF is not a supported input format for these published dense baselines",
        )
        for model, spec, model_format in product(
            models, methods, (ModelFormat.SAFETENSORS, ModelFormat.GGUF)
        )
    )
    metrics = (
        BaselineMetric("perplexity", "nats/token", "token-weighted mean", reason="not executed"),
        BaselineMetric(
            "artifact_size", "bytes", "final artifact byte count", reason="not executed"
        ),
        BaselineMetric("wall_time", "seconds", "elapsed wall time", reason="not executed"),
        BaselineMetric("gpu_time", "seconds", "sum of GPU execution phases", reason="not executed"),
        BaselineMetric(
            "peak_memory",
            "bytes",
            "peak measured host/device memory",
            reason="not executed",
        ),
        BaselineMetric(
            "failure_rate",
            "proportion",
            "failed repetitions / attempts",
            reason="not executed",
        ),
    )
    cells = tuple(
        BaselineEvidenceCell(
            model,
            method.method,
            target,
            seed,
            corpus,
            hardware,
            budget,
            BaselineOutcome.UNSUPPORTED,
            metrics,
            "pinned source revision is recorded, but its runnable executable is not "
            "bundled in this repository or CI; no performance claim is made",
        )
        for model, method, target, seed in product(
            models,
            methods,
            (SparsityTarget(0.2), SparsityTarget(0.5)),
            (11, 23, 47),
        )
    )
    return UnstructuredBaselineProtocol(
        models,
        methods,
        (SparsityTarget(0.2), SparsityTarget(0.5)),
        (11, 23, 47),
        corpus,
        hardware,
        budget,
        ("magnitude_mean_absolute", "seeded_random"),
        compatibility,
        cells,
    )


DEFAULT_UNSTRUCTURED_BASELINE_PROTOCOL = build_default_unstructured_baseline_protocol()


def render_unstructured_baseline_protocol(
    protocol: UnstructuredBaselineProtocol = DEFAULT_UNSTRUCTURED_BASELINE_PROTOCOL,
) -> str:
    """Render a deterministic canonical baseline protocol manifest."""

    return _canonical(protocol.to_record()) + "\n"

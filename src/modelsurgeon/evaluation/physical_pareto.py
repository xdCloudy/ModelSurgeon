"""Physical compression and quality-loss Pareto study evidence for v1.2."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from modelsurgeon.evaluation.deployment_benchmark import (
    DeploymentBenchmarkRecord,
    DeploymentFormat,
    DeploymentOutcome,
)
from modelsurgeon.surgery.artifact_outcome import (
    ArtifactFormat,
    ArtifactState,
    PhysicalArtifactOutcome,
)

PHYSICAL_PARETO_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_METRIC_SPECS: dict[str, tuple[str, str]] = {
    "source_artifact_size_bytes": ("bytes", "lower"),
    "artifact_size_bytes": ("bytes", "lower"),
    "active_parameters": ("parameters", "lower"),
    "quality_loss": ("score", "lower"),
    "load_time_seconds": ("s", "lower"),
    "prefill_tokens_per_second": ("tokens/s", "higher"),
    "decode_tokens_per_second": ("tokens/s", "higher"),
    "latency_seconds": ("s", "lower"),
    "peak_ram_bytes": ("bytes", "lower"),
    "peak_vram_bytes": ("bytes", "lower"),
    "disk_bytes": ("bytes", "lower"),
    "optimization_seconds": ("s", "lower"),
}
_DEPLOYMENT_METRICS = (
    "load_time_seconds",
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "latency_seconds",
    "peak_ram_bytes",
    "peak_vram_bytes",
    "disk_bytes",
)


class PhysicalParetoError(ValueError):
    """Raised when physical Pareto evidence is incomplete or incomparable."""


class PhysicalParetoOutcome(StrEnum):
    """Terminal result retained for every study matrix cell."""

    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhysicalParetoError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise PhysicalParetoError(f"{label} must be a lowercase SHA-256")
    return result


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalParetoError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PhysicalParetoError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class PhysicalParetoKey:
    """Matched identity shared by one family/target/limit/seed cell."""

    source_artifact_digest: str
    family: str
    format: ArtifactFormat
    method: str
    compression_target: str
    quality_loss_limit: float
    corpus_id: str
    evaluator_id: str
    runtime_id: str
    checkpoint_revision: str
    tool_revision: str
    seed: int

    def __post_init__(self) -> None:
        _digest(self.source_artifact_digest, "source artifact digest")
        for label, value in (
            ("model family", self.family),
            ("method", self.method),
            ("compression target", self.compression_target),
            ("corpus ID", self.corpus_id),
            ("evaluator ID", self.evaluator_id),
            ("runtime ID", self.runtime_id),
            ("checkpoint revision", self.checkpoint_revision),
            ("tool revision", self.tool_revision),
        ):
            _text(value, label)
        if self.quality_loss_limit < 0 or not math.isfinite(self.quality_loss_limit):
            raise PhysicalParetoError("quality-loss limit must be finite and non-negative")
        if self.seed < 0:
            raise PhysicalParetoError("seed must be unsigned")

    def to_record(self) -> dict[str, object]:
        return {
            "source_artifact_digest": self.source_artifact_digest,
            "family": self.family,
            "format": self.format.value,
            "method": self.method,
            "compression_target": self.compression_target,
            "quality_loss_limit": self.quality_loss_limit,
            "corpus_id": self.corpus_id,
            "evaluator_id": self.evaluator_id,
            "runtime_id": self.runtime_id,
            "checkpoint_revision": self.checkpoint_revision,
            "tool_revision": self.tool_revision,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class PhysicalParetoMetric:
    """A metric with an interval used for conservative dominance."""

    name: str
    value: float
    lower: float
    upper: float
    repetitions: int
    source: str

    def __post_init__(self) -> None:
        if self.name not in _METRIC_SPECS:
            raise PhysicalParetoError(f"unknown physical Pareto metric: {self.name}")
        for value, label in (
            (self.value, "metric value"),
            (self.lower, "metric lower bound"),
            (self.upper, "metric upper bound"),
        ):
            _finite(value, label)
            if value < 0:
                raise PhysicalParetoError(f"{label} must be non-negative")
        if not self.lower <= self.value <= self.upper:
            raise PhysicalParetoError("metric interval must contain its value")
        if self.repetitions <= 0:
            raise PhysicalParetoError("metric repetitions must be positive")
        _text(self.source, "metric source")

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
            "value": self.value,
            "lower": self.lower,
            "upper": self.upper,
            "repetitions": self.repetitions,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class PhysicalParetoCell:
    """One immutable result bound to a physical artifact and runtime record."""

    key: PhysicalParetoKey
    outcome: PhysicalParetoOutcome
    source_artifact_size_bytes: int | None
    physical_outcome: PhysicalArtifactOutcome | None
    deployment: DeploymentBenchmarkRecord | None
    metrics: tuple[PhysicalParetoMetric, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        names = tuple(item.name for item in self.metrics)
        if names != tuple(sorted(set(names))):
            raise PhysicalParetoError("physical Pareto metrics must be unique and sorted")
        if self.outcome is PhysicalParetoOutcome.MEASURED:
            if self.reason is not None:
                raise PhysicalParetoError("measured cells cannot carry a reason")
            if self.physical_outcome is None or self.deployment is None:
                raise PhysicalParetoError("measured cells require artifact and deployment evidence")
            if self.deployment.outcome is not DeploymentOutcome.MEASURED:
                raise PhysicalParetoError("measured cells require a measured deployment record")
            artifact = self.physical_outcome.artifact
            if artifact.state is not ArtifactState.COMPLETE:
                raise PhysicalParetoError("measured cells require a complete physical artifact")
            if artifact.digest != self.deployment.artifact.digest:
                raise PhysicalParetoError("physical and deployment artifact digests do not match")
            if artifact.size_bytes != self.deployment.artifact.size_bytes:
                raise PhysicalParetoError("physical and deployment artifact sizes do not match")
            expected_format = (
                DeploymentFormat.HF
                if artifact.format is ArtifactFormat.HUGGINGFACE
                else DeploymentFormat.GGUF
            )
            if self.deployment.artifact.format is not expected_format:
                raise PhysicalParetoError("physical and deployment artifact formats do not match")
            if (
                self.source_artifact_size_bytes is None
                or self.source_artifact_size_bytes <= artifact.size_bytes
            ):
                raise PhysicalParetoError(
                    "measured artifacts must be physically smaller than source"
                )
            if self.key.source_artifact_digest == artifact.digest:
                raise PhysicalParetoError("source and output artifact digests must differ")
            required = set(_METRIC_SPECS)
            if set(names) != required:
                raise PhysicalParetoError("measured cells require every frontier metric")
            active = next(item for item in self.metrics if item.name == "active_parameters")
            if active.value != self.physical_outcome.architecture.active_parameters:
                raise PhysicalParetoError(
                    "active parameter metric does not reconcile artifact outcome"
                )
            artifact_metric = next(
                item for item in self.metrics if item.name == "artifact_size_bytes"
            )
            source_metric = next(
                item for item in self.metrics if item.name == "source_artifact_size_bytes"
            )
            if (
                artifact_metric.value != artifact.size_bytes
                or source_metric.value != self.source_artifact_size_bytes
            ):
                raise PhysicalParetoError(
                    "artifact byte metrics do not reconcile physical evidence"
                )
        else:
            if not self.reason or not self.reason.strip():
                raise PhysicalParetoError("non-measured cells require a retained reason")
            if self.metrics or self.physical_outcome is not None or self.deployment is not None:
                raise PhysicalParetoError(
                    "non-measured cells cannot publish partial measured evidence"
                )

    @property
    def cell_id(self) -> str:
        identity = {
            "schema_version": PHYSICAL_PARETO_SCHEMA_VERSION,
            "key": self.key.to_record(),
        }
        return "physical_pareto_" + hashlib.sha256(_canonical(identity).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": PHYSICAL_PARETO_SCHEMA_VERSION,
            "cell_id": self.cell_id,
            "key": self.key.to_record(),
            "outcome": self.outcome.value,
            "source_artifact_size_bytes": self.source_artifact_size_bytes,
            "physical_outcome": None
            if self.physical_outcome is None
            else self.physical_outcome.to_record(),
            "deployment": None if self.deployment is None else self.deployment.to_record(),
            "metrics": [item.to_record() for item in self.metrics],
            "reason": self.reason,
        }


def _metric_from_deployment(record: DeploymentBenchmarkRecord, name: str) -> PhysicalParetoMetric:
    summary = next(item for item in record.metrics if item.name == name)
    return PhysicalParetoMetric(
        name,
        summary.median,
        max(0.0, summary.median - summary.dispersion),
        summary.median + summary.dispersion,
        len(summary.samples),
        f"deployment:{record.record_id}",
    )


def build_physical_pareto_cell(
    key: PhysicalParetoKey,
    physical_outcome: PhysicalArtifactOutcome,
    deployment: DeploymentBenchmarkRecord,
    *,
    source_artifact_size_bytes: int,
    quality_loss: float,
    quality_loss_low: float,
    quality_loss_high: float,
    quality_repetitions: int,
    optimization_seconds: float,
    optimization_low: float | None = None,
    optimization_high: float | None = None,
    optimization_repetitions: int = 1,
) -> PhysicalParetoCell:
    """Bind surgery, quality, deployment, and optimization evidence into one cell."""

    optimization_low = optimization_seconds if optimization_low is None else optimization_low
    optimization_high = optimization_seconds if optimization_high is None else optimization_high
    metrics = [
        PhysicalParetoMetric(
            "source_artifact_size_bytes",
            float(source_artifact_size_bytes),
            float(source_artifact_size_bytes),
            float(source_artifact_size_bytes),
            1,
            "physical-source-manifest",
        ),
        PhysicalParetoMetric(
            "artifact_size_bytes",
            float(deployment.artifact.size_bytes),
            float(deployment.artifact.size_bytes),
            float(deployment.artifact.size_bytes),
            1,
            f"deployment:{deployment.record_id}",
        ),
        PhysicalParetoMetric(
            "active_parameters",
            float(physical_outcome.architecture.active_parameters),
            float(physical_outcome.architecture.active_parameters),
            float(physical_outcome.architecture.active_parameters),
            1,
            f"physical:{physical_outcome.outcome_id}",
        ),
        PhysicalParetoMetric(
            "quality_loss",
            quality_loss,
            quality_loss_low,
            quality_loss_high,
            quality_repetitions,
            "held-out-quality-bootstrap",
        ),
    ]
    metrics.extend(_metric_from_deployment(deployment, name) for name in _DEPLOYMENT_METRICS)
    metrics.append(
        PhysicalParetoMetric(
            "optimization_seconds",
            optimization_seconds,
            optimization_low,
            optimization_high,
            optimization_repetitions,
            "optimization-budget-record",
        )
    )
    return PhysicalParetoCell(
        key,
        PhysicalParetoOutcome.MEASURED,
        source_artifact_size_bytes,
        physical_outcome,
        deployment,
        tuple(sorted(metrics, key=lambda item: item.name)),
    )


def conservatively_dominates(left: PhysicalParetoCell, right: PhysicalParetoCell) -> bool:
    """Return true only when interval worst cases separate on every metric."""

    if (
        left.outcome is not PhysicalParetoOutcome.MEASURED
        or right.outcome is not PhysicalParetoOutcome.MEASURED
    ):
        return False
    left_metrics = {item.name: item for item in left.metrics}
    right_metrics = {item.name: item for item in right.metrics}
    strict = False
    for name, left_metric in left_metrics.items():
        right_metric = right_metrics[name]
        if left_metric.direction == "lower":
            if left_metric.upper > right_metric.lower:
                return False
            strict |= left_metric.upper < right_metric.lower
        else:
            if left_metric.lower < right_metric.upper:
                return False
            strict |= left_metric.lower > right_metric.upper
    return strict


def compute_physical_frontier(cells: tuple[PhysicalParetoCell, ...]) -> tuple[str, ...]:
    measured = tuple(
        cell
        for cell in cells
        if cell.outcome is PhysicalParetoOutcome.MEASURED
        and next(item for item in cell.metrics if item.name == "quality_loss").upper
        <= cell.key.quality_loss_limit
    )
    frontier = tuple(
        cell
        for cell in measured
        if not any(
            other.cell_id != cell.cell_id and conservatively_dominates(other, cell)
            for other in measured
        )
    )
    return tuple(sorted(cell.cell_id for cell in frontier))


@dataclass(frozen=True, slots=True)
class PhysicalParetoStudy:
    """Complete v1.2 matrix with retained negative cells and a safe frontier."""

    families: tuple[str, ...]
    methods: tuple[str, ...]
    compression_targets: tuple[str, ...]
    quality_loss_limits: tuple[float, ...]
    seeds: tuple[int, ...]
    protocol_revision: str
    cells: tuple[PhysicalParetoCell, ...]
    frontier_cell_ids: tuple[str, ...]
    superiority_claim: str | None
    schema_version: int = PHYSICAL_PARETO_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PHYSICAL_PARETO_SCHEMA_VERSION:
            raise PhysicalParetoError("unsupported physical Pareto schema version")
        if len(self.families) < 2 or len(set(self.families)) != len(self.families):
            raise PhysicalParetoError("study requires two unique model families")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise PhysicalParetoError("study methods must be unique")
        if len(self.compression_targets) < 5 or len(set(self.compression_targets)) != len(
            self.compression_targets
        ):
            raise PhysicalParetoError("study requires five unique compression targets")
        if len(self.quality_loss_limits) < 3 or self.quality_loss_limits != tuple(
            sorted(set(self.quality_loss_limits))
        ):
            raise PhysicalParetoError("study requires three sorted unique quality-loss limits")
        if len(self.seeds) < 3 or self.seeds != tuple(sorted(set(self.seeds))):
            raise PhysicalParetoError("study requires three sorted unique seeds")
        _text(self.protocol_revision, "protocol revision")
        expected = {
            (family, method, target, limit, seed)
            for family, method, target, limit, seed in product(
                self.families,
                self.methods,
                self.compression_targets,
                self.quality_loss_limits,
                self.seeds,
            )
        }
        actual = {
            (
                cell.key.family,
                cell.key.method,
                cell.key.compression_target,
                cell.key.quality_loss_limit,
                cell.key.seed,
            )
            for cell in self.cells
        }
        if actual != expected or len(actual) != len(self.cells):
            raise PhysicalParetoError(
                "study must retain every family/method/target/limit/seed cell"
            )
        ids = {cell.cell_id for cell in self.cells}
        if any(item not in ids for item in self.frontier_cell_ids):
            raise PhysicalParetoError("frontier references an unknown cell")
        if any(
            self._cell(item).outcome is not PhysicalParetoOutcome.MEASURED
            for item in self.frontier_cell_ids
        ):
            raise PhysicalParetoError("frontier may reference measured cells only")
        if self.superiority_claim is not None:
            _text(self.superiority_claim, "superiority claim")
            if not self.frontier_cell_ids:
                raise PhysicalParetoError("a superiority claim requires a measured frontier")

    def _cell(self, cell_id: str) -> PhysicalParetoCell:
        return next(cell for cell in self.cells if cell.cell_id == cell_id)

    @property
    def study_id(self) -> str:
        return (
            "physical_pareto_study_"
            + hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "families": list(self.families),
            "methods": list(self.methods),
            "compression_targets": list(self.compression_targets),
            "quality_loss_limits": list(self.quality_loss_limits),
            "seeds": list(self.seeds),
            "protocol_revision": self.protocol_revision,
            "cells": [cell.to_record() for cell in self.cells],
            "frontier_cell_ids": list(self.frontier_cell_ids),
            "superiority_claim": self.superiority_claim,
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["study_id"] = self.study_id
        return record


def build_physical_pareto_study(
    *,
    families: tuple[str, ...],
    methods: tuple[str, ...],
    compression_targets: tuple[str, ...],
    quality_loss_limits: tuple[float, ...],
    seeds: tuple[int, ...],
    protocol_revision: str,
    cells: tuple[PhysicalParetoCell, ...],
    superiority_claim: str | None = None,
) -> PhysicalParetoStudy:
    """Build a study and derive its conservative measured frontier."""

    return PhysicalParetoStudy(
        families,
        methods,
        compression_targets,
        quality_loss_limits,
        seeds,
        protocol_revision,
        cells,
        compute_physical_frontier(cells),
        superiority_claim,
    )


def build_default_physical_pareto_study() -> PhysicalParetoStudy:
    """Return the preregistered matrix with unavailable tools retained explicitly."""

    families = ("llama", "qwen")
    methods = ("modelsurgeon", "quantization_only", "wanda", "sparsegpt", "structured_baselines")
    targets = ("checkpoint_bf16", "safetensors_int8", "gguf_q8_0", "gguf_q6_k", "gguf_q4_k_m")
    limits = (0.01, 0.03, 0.05)
    seeds = (11, 23, 47)
    source_digests = {"llama": "a" * 64, "qwen": "b" * 64}
    revision = "modelsurgeon-v1.2-physical-pareto-v1"
    cells = tuple(
        PhysicalParetoCell(
            PhysicalParetoKey(
                source_digests[family],
                family,
                ArtifactFormat.GGUF if target.startswith("gguf_") else ArtifactFormat.HUGGINGFACE,
                method,
                target,
                limit,
                "wikitext-2-raw-v1-test",
                "heldout-quality-v1",
                "deployment-runtime-v1",
                f"{family}-revision-pinned-v1",
                revision,
                seed,
            ),
            PhysicalParetoOutcome.UNSUPPORTED,
            None,
            None,
            None,
            (),
            "no licensed, reproducible artifact and runtime bundle is available for this cell",
        )
        for family, method, target, limit, seed in product(
            families, methods, targets, limits, seeds
        )
    )
    return build_physical_pareto_study(
        families=families,
        methods=methods,
        compression_targets=targets,
        quality_loss_limits=limits,
        seeds=seeds,
        protocol_revision=revision,
        cells=cells,
    )


DEFAULT_PHYSICAL_PARETO_STUDY = build_default_physical_pareto_study()


def render_physical_pareto_study(
    study: PhysicalParetoStudy = DEFAULT_PHYSICAL_PARETO_STUDY,
    *,
    format: str = "json",
) -> str:
    """Render deterministic JSON or a bounded human-readable summary."""

    if format == "json":
        return _canonical(study.to_record()) + "\n"
    if format == "markdown":
        measured = sum(cell.outcome is PhysicalParetoOutcome.MEASURED for cell in study.cells)
        return (
            "\n".join(
                (
                    f"# Physical Pareto study {study.study_id}",
                    "",
                    f"- Cells: {measured}/{len(study.cells)} measured",
                    f"- Frontier cells: {len(study.frontier_cell_ids)}",
                    "- Superiority claim: "
                    + (
                        study.superiority_claim
                        or "none; no complete measured frontier claim is made."
                    ),
                )
            )
            + "\n"
        )
    raise PhysicalParetoError("physical Pareto format must be json or markdown")

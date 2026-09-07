"""Versioned, provenance-complete evidence records for deployable benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from modelsurgeon.experiments.identity import canonical_identity_json

BENCHMARK_EVIDENCE_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BenchmarkSchemaError(ValueError):
    """Raised when benchmark evidence is incomplete, ambiguous, or unsafe."""


class BenchmarkOutcome(StrEnum):
    """The terminal state of a benchmark cell."""

    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class BenchmarkMetricState(StrEnum):
    """Whether one metric has a numeric observation."""

    MEASURED = "measured"
    UNAVAILABLE = "unavailable"


class BenchmarkArtifactState(StrEnum):
    """Whether the physical output artifact exists for this cell."""

    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"
    MISSING = "missing"


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkSchemaError(f"{label} is required")


def _finite_nonnegative(value: float, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BenchmarkSchemaError(f"{label} must be numeric")
    if not math.isfinite(float(value)) or value < 0:
        raise BenchmarkSchemaError(f"{label} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BenchmarkModel:
    """Immutable model identity used by a benchmark cell."""

    identifier: str
    revision: str
    family: str
    format: str
    parameter_count: int | None = None
    license: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("model identifier", self.identifier),
            ("model revision", self.revision),
            ("model family", self.family),
            ("model format", self.format),
        ):
            _require_text(value, label)
        if self.parameter_count is not None and (
            isinstance(self.parameter_count, bool) or self.parameter_count <= 0
        ):
            raise BenchmarkSchemaError("model parameter_count must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "family": self.family,
            "format": self.format,
            "parameter_count": self.parameter_count,
            "license": self.license,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkCorpus:
    """Dataset/task/split identity and licensing decision."""

    identifier: str
    revision: str
    split: str
    task: str
    license: str
    manifest_id: str

    def __post_init__(self) -> None:
        for label, value in (
            ("corpus identifier", self.identifier),
            ("corpus revision", self.revision),
            ("corpus split", self.split),
            ("corpus task", self.task),
            ("corpus license", self.license),
            ("corpus manifest_id", self.manifest_id),
        ):
            _require_text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "split": self.split,
            "task": self.task,
            "license": self.license,
            "manifest_id": self.manifest_id,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkMethod:
    """Competitor or reference method implementation identity."""

    name: str
    revision: str
    variant: str

    def __post_init__(self) -> None:
        for label, value in (
            ("method name", self.name),
            ("method revision", self.revision),
            ("method variant", self.variant),
        ):
            _require_text(value, label)

    def to_record(self) -> dict[str, str]:
        return {"name": self.name, "revision": self.revision, "variant": self.variant}


@dataclass(frozen=True, slots=True)
class BenchmarkHardware:
    """Hardware and runtime identity; local paths must not be used as identity."""

    platform: str
    cpu: str
    accelerator: str
    runtime: str
    details: Mapping[str, object]

    def __post_init__(self) -> None:
        for label, value in (
            ("hardware platform", self.platform),
            ("hardware cpu", self.cpu),
            ("hardware accelerator", self.accelerator),
            ("hardware runtime", self.runtime),
        ):
            _require_text(value, label)
        if not isinstance(self.details, Mapping):
            raise BenchmarkSchemaError("hardware details must be a mapping")
        try:
            canonical_identity_json(self.details)
        except (TypeError, ValueError) as error:
            raise BenchmarkSchemaError("hardware details must be canonical JSON") from error

    def to_record(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "cpu": self.cpu,
            "accelerator": self.accelerator,
            "runtime": self.runtime,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkBudgetLimit:
    """One bounded resource or quality budget with an explicit unit."""

    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        _require_text(self.name, "budget name")
        _require_text(self.unit, "budget unit")
        _finite_nonnegative(self.value, "budget value")

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class BenchmarkBudget:
    """Equal-budget constraints applied to optimization and evaluation."""

    limits: tuple[BenchmarkBudgetLimit, ...]
    objective: str

    def __post_init__(self) -> None:
        _require_text(self.objective, "budget objective")
        names = tuple(item.name for item in self.limits)
        if not names or names != tuple(sorted(set(names))):
            raise BenchmarkSchemaError("budget limits must be non-empty and canonically sorted")

    def to_record(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "limits": [item.to_record() for item in self.limits],
        }


@dataclass(frozen=True, slots=True)
class BenchmarkMetric:
    """A metric that never uses a sentinel number for missing evidence."""

    name: str
    unit: str
    state: BenchmarkMetricState = BenchmarkMetricState.MEASURED
    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "metric name")
        _require_text(self.unit, "metric unit")
        if self.state is BenchmarkMetricState.MEASURED:
            if self.value is None or not math.isfinite(self.value):
                raise BenchmarkSchemaError("measured metrics require a finite value")
            if self.reason is not None:
                raise BenchmarkSchemaError("measured metrics cannot carry an unavailable reason")
            if self.lower_bound is not None and not math.isfinite(self.lower_bound):
                raise BenchmarkSchemaError("metric lower_bound must be finite")
            if self.upper_bound is not None and not math.isfinite(self.upper_bound):
                raise BenchmarkSchemaError("metric upper_bound must be finite")
            if self.lower_bound is not None and self.lower_bound > self.value:
                raise BenchmarkSchemaError("metric lower_bound exceeds value")
            if self.upper_bound is not None and self.upper_bound < self.value:
                raise BenchmarkSchemaError("metric upper_bound is below value")
            return
        if self.value is not None or self.lower_bound is not None or self.upper_bound is not None:
            raise BenchmarkSchemaError("unavailable metrics cannot carry numeric values")
        _require_text(self.reason or "", "metric unavailable reason")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "state": self.state.value,
            "value": self.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkArtifact:
    """Physical output lineage, including an explicit non-publication state."""

    identifier: str
    revision: str
    format: str
    state: BenchmarkArtifactState
    digest: str | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("artifact identifier", self.identifier),
            ("artifact revision", self.revision),
            ("artifact format", self.format),
        ):
            _require_text(value, label)
        if self.state is BenchmarkArtifactState.AVAILABLE:
            if self.digest is None or not _SHA256.fullmatch(self.digest):
                raise BenchmarkSchemaError("available artifacts require a lowercase SHA-256 digest")
            if self.size_bytes is None or self.size_bytes <= 0:
                raise BenchmarkSchemaError("available artifacts require a positive size")
        elif self.digest is not None or self.size_bytes is not None:
            raise BenchmarkSchemaError("unavailable artifacts cannot carry publication metadata")

    def to_record(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "format": self.format,
            "state": self.state.value,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkProvenance:
    """Code, evaluator, configuration, and lock identities for reproduction."""

    tool_revision: str
    evaluator_revision: str
    config_digest: str
    source_commit: str
    dependency_lock_digest: str
    command: str

    def __post_init__(self) -> None:
        for label, value in (
            ("tool revision", self.tool_revision),
            ("evaluator revision", self.evaluator_revision),
            ("configuration digest", self.config_digest),
            ("source commit", self.source_commit),
            ("dependency lock digest", self.dependency_lock_digest),
            ("benchmark command", self.command),
        ):
            _require_text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "tool_revision": self.tool_revision,
            "evaluator_revision": self.evaluator_revision,
            "config_digest": self.config_digest,
            "source_commit": self.source_commit,
            "dependency_lock_digest": self.dependency_lock_digest,
            "command": self.command,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkConfidenceInterval:
    """Uncertainty retained alongside the metric and its unit."""

    metric_name: str
    level: float
    lower_bound: float
    upper_bound: float

    def __post_init__(self) -> None:
        _require_text(self.metric_name, "confidence metric_name")
        if not 0 < self.level < 1:
            raise BenchmarkSchemaError("confidence level must be between zero and one")
        for label, value in (
            ("confidence lower_bound", self.lower_bound),
            ("confidence upper_bound", self.upper_bound),
        ):
            if not math.isfinite(value):
                raise BenchmarkSchemaError(f"{label} must be finite")
        if self.lower_bound > self.upper_bound:
            raise BenchmarkSchemaError("confidence lower_bound exceeds upper_bound")

    def to_record(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "level": self.level,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReliability:
    """Repeated-run and failure accounting for a benchmark cell."""

    repetitions: int
    successful_repetitions: int
    intervals: tuple[BenchmarkConfidenceInterval, ...] = ()

    def __post_init__(self) -> None:
        if self.repetitions <= 0 or not 0 <= self.successful_repetitions <= self.repetitions:
            raise BenchmarkSchemaError("reliability repetitions are inconsistent")
        names = tuple(item.metric_name for item in self.intervals)
        if len(names) != len(set(names)):
            raise BenchmarkSchemaError("reliability intervals must use unique metric names")

    def to_record(self) -> dict[str, object]:
        return {
            "repetitions": self.repetitions,
            "successful_repetitions": self.successful_repetitions,
            "intervals": [item.to_record() for item in self.intervals],
        }


def _metric_records(metrics: Sequence[BenchmarkMetric]) -> list[dict[str, object]]:
    names = tuple(metric.name for metric in metrics)
    if len(names) != len(set(names)):
        raise BenchmarkSchemaError("metric names must be unique within each metric group")
    return [metric.to_record() for metric in metrics]


@dataclass(frozen=True, slots=True)
class BenchmarkEvidenceRecord:
    """One fully identified benchmark cell and its measured or terminal result."""

    model: BenchmarkModel
    corpus: BenchmarkCorpus
    method: BenchmarkMethod
    hardware: BenchmarkHardware
    budget: BenchmarkBudget
    seeds: tuple[int, ...]
    provenance: BenchmarkProvenance
    artifact: BenchmarkArtifact
    outcome: BenchmarkOutcome
    quality_metrics: tuple[BenchmarkMetric, ...] = ()
    runtime_metrics: tuple[BenchmarkMetric, ...] = ()
    optimization_cost_metrics: tuple[BenchmarkMetric, ...] = ()
    reliability: BenchmarkReliability | None = None
    outcome_reason: str | None = None
    schema_version: int = BENCHMARK_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_EVIDENCE_SCHEMA_VERSION:
            raise BenchmarkSchemaError("unsupported benchmark evidence schema version")
        if not self.seeds or self.seeds != tuple(sorted(set(self.seeds))):
            raise BenchmarkSchemaError("benchmark seeds must be non-empty, unique, and sorted")
        if any(isinstance(seed, bool) or seed < 0 for seed in self.seeds):
            raise BenchmarkSchemaError("benchmark seeds must be unsigned integers")
        if self.outcome is BenchmarkOutcome.MEASURED:
            if self.outcome_reason is not None:
                raise BenchmarkSchemaError("measured outcomes cannot carry a failure reason")
            if self.artifact.state is not BenchmarkArtifactState.AVAILABLE:
                raise BenchmarkSchemaError("measured outcomes require an available artifact")
        else:
            _require_text(self.outcome_reason or "", "benchmark outcome reason")
        all_metrics = (
            *self.quality_metrics,
            *self.runtime_metrics,
            *self.optimization_cost_metrics,
        )
        if self.outcome is BenchmarkOutcome.UNSUPPORTED and any(
            item.state is BenchmarkMetricState.MEASURED for item in all_metrics
        ):
            raise BenchmarkSchemaError("unsupported outcomes cannot claim measured metrics")
        if self.reliability is not None:
            known_names = {item.name for item in all_metrics}
            if any(item.metric_name not in known_names for item in self.reliability.intervals):
                raise BenchmarkSchemaError("reliability interval references an unknown metric")

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_record(),
            "corpus": self.corpus.to_record(),
            "method": self.method.to_record(),
            "hardware": self.hardware.to_record(),
            "budget": self.budget.to_record(),
            "seeds": list(self.seeds),
            "provenance": self.provenance.to_record(),
            "artifact": self.artifact.to_record(),
        }

    @property
    def cell_id(self) -> str:
        digest = hashlib.sha256(
            canonical_identity_json(self._identity_record()).encode()
        ).hexdigest()
        return f"benchmark_{digest}"

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record.update(
            {
                "cell_id": self.cell_id,
                "outcome": self.outcome.value,
                "outcome_reason": self.outcome_reason,
                "quality_metrics": _metric_records(self.quality_metrics),
                "runtime_metrics": _metric_records(self.runtime_metrics),
                "optimization_cost_metrics": _metric_records(self.optimization_cost_metrics),
                "reliability": None if self.reliability is None else self.reliability.to_record(),
            }
        )
        return record


def migrate_benchmark_record(raw: Mapping[str, object]) -> dict[str, object]:
    """Migrate the flat v0 fixture shape to the versioned v1 record shape."""

    if not isinstance(raw, Mapping):
        raise BenchmarkSchemaError("benchmark record must be an object")
    version = raw.get("schema_version")
    if version == BENCHMARK_EVIDENCE_SCHEMA_VERSION:
        return cast(dict[str, object], json.loads(canonical_identity_json(raw)))
    if version != 0:
        raise BenchmarkSchemaError(f"unsupported benchmark migration version: {version!r}")
    if "status" not in raw or "metrics" not in raw:
        raise BenchmarkSchemaError("v0 benchmark records require status and metrics")
    migrated = dict(raw)
    migrated["schema_version"] = BENCHMARK_EVIDENCE_SCHEMA_VERSION
    migrated["outcome"] = migrated.pop("status")
    migrated["quality_metrics"] = migrated.pop("metrics")
    migrated.setdefault("runtime_metrics", [])
    migrated.setdefault("optimization_cost_metrics", [])
    migrated.setdefault("reliability", None)
    migrated.setdefault("outcome_reason", None)
    migrated.pop("cell_id", None)
    return cast(dict[str, object], json.loads(canonical_identity_json(migrated)))


def render_benchmark_report(record: BenchmarkEvidenceRecord, *, format: str = "json") -> str:
    """Render stable JSON or human-readable Markdown without dropping units."""

    if format == "json":
        return canonical_identity_json(record.to_record()) + "\n"
    if format != "markdown":
        raise BenchmarkSchemaError(f"unsupported benchmark report format: {format!r}")
    lines = [
        f"# Benchmark cell {record.cell_id}",
        "",
        f"- Outcome: **{record.outcome.value}**",
        f"- Model: {record.model.identifier} @ {record.model.revision}",
        f"- Corpus: {record.corpus.identifier} / {record.corpus.split} @ {record.corpus.revision}",
        f"- Method: {record.method.name} / {record.method.variant} @ {record.method.revision}",
        f"- Tool: {record.provenance.tool_revision}; "
        f"evaluator: {record.provenance.evaluator_revision}",
        f"- Artifact: {record.artifact.state.value} / {record.artifact.revision}",
        "",
    ]
    if record.outcome_reason:
        lines.extend([f"Reason: {record.outcome_reason}", ""])
    for title, metrics in (
        ("Quality", record.quality_metrics),
        ("Runtime", record.runtime_metrics),
        ("Optimization cost", record.optimization_cost_metrics),
    ):
        lines.extend(
            [
                f"## {title}",
                "",
                "| Metric | Value | Unit | Uncertainty |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for metric in metrics:
            if metric.state is BenchmarkMetricState.MEASURED:
                value = str(metric.value)
                uncertainty = (
                    ""
                    if metric.lower_bound is None and metric.upper_bound is None
                    else f"[{metric.lower_bound}, {metric.upper_bound}]"
                )
            else:
                value, uncertainty = "—", metric.reason or "unavailable"
            lines.append(f"| {metric.name} | {value} | {metric.unit} | {uncertainty} |")
        lines.append("")
    return "\n".join(lines)

"""Fail-closed release-candidate contract for the v2 benchmark.

This module deliberately records and validates benchmark evidence; it does not
run models, download checkpoints, or turn a missing measurement into a result.
The contract is the boundary between an executed campaign and a publishable
reference package.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from modelsurgeon.evaluation.public_audit import (
    FindingSeverity,
    PublicAuditReport,
)
from modelsurgeon.experiments.identity import canonical_identity_json

V2_BENCHMARK_SCHEMA_VERSION: Final[int] = 1
V2_REQUIRED_METRICS: Final[tuple[str, ...]] = (
    "artifact_size_bytes",
    "cost_units",
    "decode_tokens_per_second",
    "optimization_seconds",
    "prompt_tokens_per_second",
    "quality",
    "ram_bytes",
    "vram_bytes",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LINEAGE_STAGES = ("mutation", "repair", "quantization", "runtime", "rollback")


class V2BenchmarkError(ValueError):
    """Raised when v2 benchmark evidence is malformed or unsafe to publish."""


class V2CellOutcome(StrEnum):
    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ArtifactAvailability(StrEnum):
    AVAILABLE = "available"
    NOT_PERMITTED = "not_permitted"
    NOT_AVAILABLE = "not_available"
    FAILED = "failed"


class V2PublicationOutcome(StrEnum):
    PUBLISH = "publish"
    WITHHOLD = "withhold"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V2BenchmarkError(f"{label} is required")
    return value


def _canonical_items(values: Sequence[str], label: str, *, minimum: int = 1) -> tuple[str, ...]:
    result = tuple(values)
    if (
        len(result) < minimum
        or result != tuple(sorted(set(result)))
        or any(not item.strip() for item in result)
    ):
        raise V2BenchmarkError(f"{label} must be non-empty, unique, and sorted")
    return result


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise V2BenchmarkError(f"{label} must be a lowercase SHA-256")
    return result


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V2BenchmarkError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise V2BenchmarkError(f"{label} must be finite")
    return result


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise V2BenchmarkError(f"{label} must be a positive integer")
    return value


def _canonical_digest(value: object, namespace: str) -> str:
    return f"{namespace}_{hashlib.sha256(canonical_identity_json(value).encode()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class V2ModelSpec:
    """Exact model checkpoint identity preregistered for one benchmark cell."""

    family: str
    size: str
    identifier: str
    revision: str
    license: str

    def __post_init__(self) -> None:
        for label, value in (
            ("model family", self.family),
            ("model size", self.size),
            ("model identifier", self.identifier),
            ("model revision", self.revision),
            ("model license", self.license),
        ):
            _text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "family": self.family,
            "size": self.size,
            "identifier": self.identifier,
            "revision": self.revision,
            "license": self.license,
        }


@dataclass(frozen=True, slots=True)
class V2DatasetSpec:
    """Pinned task/split identity; publication requires a held-out split."""

    identifier: str
    revision: str
    task: str
    split: str
    license: str
    held_out: bool = True

    def __post_init__(self) -> None:
        for label, value in (
            ("dataset identifier", self.identifier),
            ("dataset revision", self.revision),
            ("dataset task", self.task),
            ("dataset split", self.split),
            ("dataset license", self.license),
        ):
            _text(value, label)

    def to_record(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "task": self.task,
            "split": self.split,
            "license": self.license,
            "held_out": self.held_out,
        }


@dataclass(frozen=True, slots=True)
class V2MethodSpec:
    """Implementation identity for the autonomous arm or a comparison arm."""

    name: str
    revision: str
    kind: str
    license: str | None = None

    def __post_init__(self) -> None:
        _text(self.name, "method name")
        _text(self.revision, "method revision")
        if self.kind not in {"autonomous", "baseline", "ablation", "competitor", "manual"}:
            raise V2BenchmarkError("method kind is unsupported")
        if self.license is not None:
            _text(self.license, "method license")

    def to_record(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "revision": self.revision,
            "kind": self.kind,
            "license": self.license,
        }


@dataclass(frozen=True, slots=True)
class V2HardwareProfile:
    """Stable hardware/runtime identity with a bounded memory ceiling."""

    name: str
    cpu: str
    accelerator: str
    runtime_revision: str
    memory_gib: float

    def __post_init__(self) -> None:
        for label, value in (
            ("hardware profile name", self.name),
            ("hardware cpu", self.cpu),
            ("hardware accelerator", self.accelerator),
            ("runtime revision", self.runtime_revision),
        ):
            _text(value, label)
        if _finite(self.memory_gib, "hardware memory_gib") <= 0:
            raise V2BenchmarkError("hardware memory_gib must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "cpu": self.cpu,
            "accelerator": self.accelerator,
            "runtime_revision": self.runtime_revision,
            "memory_gib": self.memory_gib,
        }


@dataclass(frozen=True, slots=True)
class V2Budget:
    """Equal optimization/evaluation/resource limits for every protocol cell."""

    optimization_seconds: float
    evaluation_tokens: int
    max_memory_gib: float
    max_artifact_bytes: int

    def __post_init__(self) -> None:
        if _finite(self.optimization_seconds, "optimization_seconds") <= 0:
            raise V2BenchmarkError("optimization_seconds must be positive")
        _positive(self.evaluation_tokens, "evaluation_tokens")
        if _finite(self.max_memory_gib, "max_memory_gib") <= 0:
            raise V2BenchmarkError("max_memory_gib must be positive")
        _positive(self.max_artifact_bytes, "max_artifact_bytes")

    def to_record(self) -> dict[str, object]:
        return {
            "optimization_seconds": self.optimization_seconds,
            "evaluation_tokens": self.evaluation_tokens,
            "max_memory_gib": self.max_memory_gib,
            "max_artifact_bytes": self.max_artifact_bytes,
        }


@dataclass(frozen=True, slots=True)
class V2BenchmarkProtocol:
    """Preregistered v2 matrix and decision boundary."""

    revision: str
    models: tuple[V2ModelSpec, ...]
    datasets: tuple[V2DatasetSpec, ...]
    methods: tuple[V2MethodSpec, ...]
    hardware: tuple[V2HardwareProfile, ...]
    budget: V2Budget
    seeds: tuple[int, ...]
    repetitions: int = 3
    metrics: tuple[str, ...] = V2_REQUIRED_METRICS
    confidence_level: float = 0.95
    decision_thresholds: Mapping[str, float] = field(default_factory=dict)
    max_cells: int = 10_000

    def __post_init__(self) -> None:
        _text(self.revision, "protocol revision")
        if (
            len(self.models) < 4
            or len({item.family for item in self.models}) < 2
            or len({item.size for item in self.models}) < 2
        ):
            raise V2BenchmarkError("protocol requires multiple families and model sizes")
        if len({(item.identifier, item.revision) for item in self.models}) != len(self.models):
            raise V2BenchmarkError("protocol model checkpoints must be unique")
        if len(
            {(item.identifier, item.revision, item.task, item.split) for item in self.datasets}
        ) != len(self.datasets):
            raise V2BenchmarkError("protocol dataset splits must be unique")
        if len({(item.name, item.revision) for item in self.methods}) != len(self.methods):
            raise V2BenchmarkError("protocol methods must be unique")
        if len({item.name for item in self.hardware}) != len(self.hardware):
            raise V2BenchmarkError("protocol hardware profiles must be unique")
        if not self.datasets or not any(item.held_out for item in self.datasets):
            raise V2BenchmarkError("protocol requires a held-out dataset split")
        if (
            not self.methods
            or not any(item.kind == "autonomous" for item in self.methods)
            or not any(item.kind in {"baseline", "competitor"} for item in self.methods)
        ):
            raise V2BenchmarkError("protocol requires autonomous and baseline methods")
        if not self.hardware:
            raise V2BenchmarkError("protocol requires one hardware profile")
        if len(self.seeds) < 3 or self.seeds != tuple(sorted(set(self.seeds))):
            raise V2BenchmarkError("protocol requires at least three sorted seeds")
        if any(isinstance(seed, bool) or seed < 0 for seed in self.seeds):
            raise V2BenchmarkError("protocol seeds must be unsigned integers")
        if self.repetitions < 3:
            raise V2BenchmarkError("protocol requires at least three repetitions")
        confidence = _finite(self.confidence_level, "confidence level")
        if not 0.5 < confidence < 1:
            raise V2BenchmarkError("confidence level must be between 0.5 and 1")
        _canonical_items(self.metrics, "protocol metrics")
        if not set(V2_REQUIRED_METRICS).issubset(self.metrics):
            raise V2BenchmarkError("protocol metrics omit a required frontier metric")
        thresholds = {} if self.decision_thresholds is None else dict(self.decision_thresholds)
        if any(not isinstance(key, str) or not key.strip() for key in thresholds):
            raise V2BenchmarkError("decision threshold names must be non-empty")
        for name, value in thresholds.items():
            _finite(value, f"decision threshold {name}")
        if self.max_cells <= 0:
            raise V2BenchmarkError("max_cells must be positive")
        if self.expected_cell_count > self.max_cells:
            raise V2BenchmarkError("protocol matrix exceeds its bounded cell limit")

    @property
    def expected_cell_count(self) -> int:
        return (
            len(self.models)
            * len(self.datasets)
            * len(self.methods)
            * len(self.hardware)
            * len(self.seeds)
        )

    @property
    def protocol_id(self) -> str:
        return _canonical_digest(self.to_record(), "v2_protocol")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": V2_BENCHMARK_SCHEMA_VERSION,
            "revision": self.revision,
            "models": [item.to_record() for item in self.models],
            "datasets": [item.to_record() for item in self.datasets],
            "methods": [item.to_record() for item in self.methods],
            "hardware": [item.to_record() for item in self.hardware],
            "budget": self.budget.to_record(),
            "seeds": list(self.seeds),
            "repetitions": self.repetitions,
            "confidence_level": self.confidence_level,
            "metrics": list(self.metrics),
            "decision_thresholds": dict(self.decision_thresholds),
            "max_cells": self.max_cells,
        }


@dataclass(frozen=True, slots=True)
class V2Metric:
    name: str
    unit: str
    value: float | None
    confidence_low: float | None = None
    confidence_high: float | None = None
    repetitions: int = 0
    confidence_level: float = 0.95
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.name, "metric name")
        _text(self.unit, "metric unit")
        if self.value is None:
            if self.confidence_low is not None or self.confidence_high is not None:
                raise V2BenchmarkError("unavailable metrics cannot carry confidence bounds")
            _text(self.reason or "", "unavailable metric reason")
            return
        value = _finite(self.value, "metric value")
        confidence = _finite(self.confidence_level, "metric confidence level")
        if not 0.5 < confidence < 1:
            raise V2BenchmarkError("metric confidence level must be between 0.5 and 1")
        if self.confidence_low is None or self.confidence_high is None:
            raise V2BenchmarkError("measured metrics require confidence bounds")
        low = _finite(self.confidence_low, "metric confidence_low")
        high = _finite(self.confidence_high, "metric confidence_high")
        if not low <= value <= high:
            raise V2BenchmarkError("confidence bounds must contain metric value")
        if self.repetitions < 3:
            raise V2BenchmarkError("measured metrics require three repetitions")
        if self.reason is not None:
            raise V2BenchmarkError("measured metrics cannot carry a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "value": self.value,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "repetitions": self.repetitions,
            "confidence_level": self.confidence_level,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ArtifactLineageStage:
    name: str
    revision: str
    input_digest: str
    output_digest: str
    config_digest: str

    def __post_init__(self) -> None:
        if self.name not in _LINEAGE_STAGES:
            raise V2BenchmarkError("artifact lineage has an unknown stage")
        _text(self.revision, f"{self.name} revision")
        for label, value in (
            ("input digest", self.input_digest),
            ("output digest", self.output_digest),
            ("config digest", self.config_digest),
        ):
            _digest(value, f"{self.name} {label}")

    def to_record(self) -> dict[str, str]:
        return {
            "name": self.name,
            "revision": self.revision,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "config_digest": self.config_digest,
        }


@dataclass(frozen=True, slots=True)
class ReferenceArtifactLineage:
    """Complete mutation, repair, quantization, runtime, and rollback chain."""

    stages: tuple[ArtifactLineageStage, ...]
    license_id: str
    license_allows_publication: bool

    def __post_init__(self) -> None:
        names = tuple(item.name for item in self.stages)
        if names != _LINEAGE_STAGES:
            raise V2BenchmarkError("artifact lineage must contain all ordered stages")
        _text(self.license_id, "artifact license_id")
        if any(
            current.input_digest != previous.output_digest
            for previous, current in zip(self.stages, self.stages[1:], strict=False)
        ):
            raise V2BenchmarkError("artifact lineage stage digests are discontinuous")
        if self.stages[-1].output_digest != self.stages[0].input_digest:
            raise V2BenchmarkError("rollback lineage must return to the source digest")

    def to_record(self) -> dict[str, object]:
        return {
            "stages": [item.to_record() for item in self.stages],
            "license_id": self.license_id,
            "license_allows_publication": self.license_allows_publication,
        }


@dataclass(frozen=True, slots=True)
class ReferenceArtifact:
    identifier: str
    revision: str
    format: str
    state: ArtifactAvailability
    digest: str | None
    size_bytes: int | None
    lineage: ReferenceArtifactLineage
    reason: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("artifact identifier", self.identifier),
            ("artifact revision", self.revision),
            ("artifact format", self.format),
        ):
            _text(value, label)
        if self.state is ArtifactAvailability.AVAILABLE:
            if self.digest is None:
                raise V2BenchmarkError("available artifacts require a digest")
            _digest(self.digest, "artifact digest")
            if self.size_bytes is None or self.size_bytes <= 0:
                raise V2BenchmarkError("available artifacts require positive size")
            if not self.lineage.license_allows_publication:
                raise V2BenchmarkError("license-prohibited artifact cannot be available")
            if self.lineage.stages[-2].output_digest != self.digest:
                raise V2BenchmarkError("artifact digest does not match runtime lineage")
            if self.reason is not None:
                raise V2BenchmarkError("available artifacts cannot carry a reason")
        else:
            if self.digest is not None or self.size_bytes is not None:
                raise V2BenchmarkError("unavailable artifacts cannot carry public metadata")
            _text(self.reason or "", "unavailable artifact reason")

    def to_record(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "format": self.format,
            "state": self.state.value,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "lineage": self.lineage.to_record(),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class V2BenchmarkCell:
    model: V2ModelSpec
    dataset: V2DatasetSpec
    method: V2MethodSpec
    hardware: V2HardwareProfile
    budget: V2Budget
    seed: int
    outcome: V2CellOutcome
    artifact: ReferenceArtifact
    metrics: tuple[V2Metric, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise V2BenchmarkError("cell seed must be unsigned")
        names = tuple(item.name for item in self.metrics)
        if names != tuple(sorted(set(names))):
            raise V2BenchmarkError("cell metrics must be unique and sorted")
        if self.outcome is V2CellOutcome.MEASURED:
            if self.reason is not None or self.artifact.state is not ArtifactAvailability.AVAILABLE:
                raise V2BenchmarkError("measured cells require a publishable artifact")
            if any(item.value is None for item in self.metrics):
                raise V2BenchmarkError("measured cells require complete metrics")
        else:
            if self.artifact.state is ArtifactAvailability.AVAILABLE:
                raise V2BenchmarkError("terminal cells cannot claim an available artifact")
            _text(self.reason or "", "terminal cell reason")

    @property
    def cell_id(self) -> str:
        return _canonical_digest(
            {
                "schema_version": V2_BENCHMARK_SCHEMA_VERSION,
                "model": self.model.to_record(),
                "dataset": self.dataset.to_record(),
                "method": self.method.to_record(),
                "hardware": self.hardware.to_record(),
                "budget": self.budget.to_record(),
                "seed": self.seed,
            },
            "v2_cell",
        )

    def to_record(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "model": self.model.to_record(),
            "dataset": self.dataset.to_record(),
            "method": self.method.to_record(),
            "hardware": self.hardware.to_record(),
            "budget": self.budget.to_record(),
            "seed": self.seed,
            "outcome": self.outcome.value,
            "artifact": self.artifact.to_record(),
            "metrics": [item.to_record() for item in self.metrics],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class V2CompetitivenessClaim:
    claim_id: str
    metric: str
    candidate_cell_ids: tuple[str, ...]
    baseline_cell_ids: tuple[str, ...]
    direction: str
    margin: float = 0.0
    published: bool = True

    def __post_init__(self) -> None:
        _text(self.claim_id, "claim ID")
        _text(self.metric, "claim metric")
        _canonical_items(self.candidate_cell_ids, "candidate claim cells")
        _canonical_items(self.baseline_cell_ids, "baseline claim cells")
        if self.direction not in {"higher", "lower"}:
            raise V2BenchmarkError("claim direction must be higher or lower")
        if _finite(self.margin, "claim margin") < 0:
            raise V2BenchmarkError("claim margin cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "metric": self.metric,
            "candidate_cell_ids": list(self.candidate_cell_ids),
            "baseline_cell_ids": list(self.baseline_cell_ids),
            "direction": self.direction,
            "margin": self.margin,
            "published": self.published,
        }


@dataclass(frozen=True, slots=True)
class V2AuditRecord:
    auditor_revision: str
    audit_digest: str
    outcome: str
    replay_digest: str
    replay_passed: bool
    critical_findings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.auditor_revision, "auditor revision")
        _digest(self.audit_digest, "audit digest")
        _digest(self.replay_digest, "replay digest")
        if self.outcome not in {"clean", "findings", "unknown"}:
            raise V2BenchmarkError("audit outcome is invalid")
        _canonical_items(self.critical_findings, "critical findings", minimum=0)
        _canonical_items(self.limitations, "audit limitations", minimum=0)

    @classmethod
    def from_public_audit(
        cls,
        report: PublicAuditReport,
        *,
        auditor_revision: str,
        replay_digest: str,
        replay_passed: bool,
        limitations: tuple[str, ...] = (),
    ) -> V2AuditRecord:
        critical = tuple(
            sorted(
                item.code for item in report.findings if item.severity is FindingSeverity.BLOCKING
            )
        )
        return cls(
            auditor_revision,
            _digest(
                hashlib.sha256(canonical_identity_json(report.to_record()).encode()).hexdigest(),
                "audit digest",
            ),
            report.outcome.value,
            replay_digest,
            replay_passed,
            critical,
            limitations,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "auditor_revision": self.auditor_revision,
            "audit_digest": self.audit_digest,
            "outcome": self.outcome,
            "replay_digest": self.replay_digest,
            "replay_passed": self.replay_passed,
            "critical_findings": list(self.critical_findings),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class V2PublicationDecision:
    outcome: V2PublicationOutcome
    reasons: tuple[str, ...]
    supported_claim_ids: tuple[str, ...]

    @property
    def decision_id(self) -> str:
        return _canonical_digest(
            {
                "schema_version": V2_BENCHMARK_SCHEMA_VERSION,
                "outcome": self.outcome.value,
                "reasons": list(self.reasons),
                "supported_claim_ids": list(self.supported_claim_ids),
            },
            "v2_publication",
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": V2_BENCHMARK_SCHEMA_VERSION,
            "outcome": self.outcome.value,
            "reasons": list(self.reasons),
            "supported_claim_ids": list(self.supported_claim_ids),
            "decision_id": self.decision_id,
        }


@dataclass(frozen=True, slots=True)
class V2BenchmarkBundle:
    """Complete, deterministic evidence package input to publication review."""

    protocol: V2BenchmarkProtocol
    cells: tuple[V2BenchmarkCell, ...]
    claims: tuple[V2CompetitivenessClaim, ...]
    audit: V2AuditRecord | None

    def __post_init__(self) -> None:
        cell_ids = tuple(item.cell_id for item in self.cells)
        if cell_ids != tuple(sorted(set(cell_ids))):
            raise V2BenchmarkError("bundle cells must be unique and sorted")
        claim_ids = tuple(item.claim_id for item in self.claims)
        if claim_ids != tuple(sorted(set(claim_ids))):
            raise V2BenchmarkError("bundle claims must be unique and sorted")

    @property
    def bundle_id(self) -> str:
        return _canonical_digest(self._identity_record(), "v2_bundle")

    @property
    def decision(self) -> V2PublicationDecision:
        return evaluate_v2_publication(self.protocol, self.cells, self.claims, self.audit)

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": V2_BENCHMARK_SCHEMA_VERSION,
            "protocol": self.protocol.to_record(),
            "cells": [item.to_record() for item in self.cells],
            "claims": [item.to_record() for item in self.claims],
            "audit": None if self.audit is None else self.audit.to_record(),
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self._identity_record(),
            "bundle_id": self.bundle_id,
            "decision": self.decision.to_record(),
        }


def build_v2_audit_record(
    report: PublicAuditReport,
    *,
    auditor_revision: str,
    replay_digest: str,
    replay_passed: bool,
    limitations: tuple[str, ...] = (),
) -> V2AuditRecord:
    """Bind the existing public auditor and an independent replay to this release."""

    return V2AuditRecord.from_public_audit(
        report,
        auditor_revision=auditor_revision,
        replay_digest=replay_digest,
        replay_passed=replay_passed,
        limitations=limitations,
    )


def build_v2_benchmark_bundle(
    protocol: V2BenchmarkProtocol,
    cells: Sequence[V2BenchmarkCell],
    claims: Sequence[V2CompetitivenessClaim],
    audit: V2AuditRecord | None,
) -> V2BenchmarkBundle:
    """Canonicalize retained cells and claims into one immutable bundle."""

    return V2BenchmarkBundle(
        protocol,
        tuple(sorted(cells, key=lambda item: item.cell_id)),
        tuple(sorted(claims, key=lambda item: item.claim_id)),
        audit,
    )


def _aggregate_interval(
    cells: Sequence[V2BenchmarkCell], metric: str
) -> tuple[float, float] | None:
    intervals: list[tuple[float, float]] = []
    for cell in cells:
        found = next((item for item in cell.metrics if item.name == metric), None)
        if (
            found is None
            or found.value is None
            or found.confidence_low is None
            or found.confidence_high is None
        ):
            return None
        intervals.append((found.confidence_low, found.confidence_high))
    return (
        (
            math.fsum(item[0] for item in intervals) / len(intervals),
            math.fsum(item[1] for item in intervals) / len(intervals),
        )
        if intervals
        else None
    )


def evaluate_v2_publication(
    protocol: V2BenchmarkProtocol,
    cells: Sequence[V2BenchmarkCell],
    claims: Sequence[V2CompetitivenessClaim],
    audit: V2AuditRecord | None,
) -> V2PublicationDecision:
    """Return a deterministic publish/withhold decision without changing evidence."""

    reasons: set[str] = set()
    by_id: dict[str, V2BenchmarkCell] = {}
    for cell in cells:
        if cell.cell_id in by_id:
            reasons.add(f"duplicate cell identity: {cell.cell_id}")
        by_id[cell.cell_id] = cell
    expected = {
        V2BenchmarkCell(
            model,
            dataset,
            method,
            hardware,
            protocol.budget,
            seed,
            V2CellOutcome.UNKNOWN,
            _placeholder_artifact(),
            (),
            "identity only",
        ).cell_id
        for model in protocol.models
        for dataset in protocol.datasets
        for method in protocol.methods
        for hardware in protocol.hardware
        for seed in protocol.seeds
    }
    missing = sorted(expected - set(by_id))
    if missing:
        reasons.add(f"incomplete matrix: {len(missing)} cells are missing")
    unexpected = sorted(set(by_id) - expected)
    if unexpected:
        reasons.add("matrix contains cells outside the preregistered protocol")
    for cell in cells:
        names = {item.name for item in cell.metrics}
        if cell.outcome is V2CellOutcome.MEASURED and set(protocol.metrics) != names:
            reasons.add(f"measured cell has incomplete metric set: {cell.cell_id}")
        if cell.outcome is V2CellOutcome.MEASURED and any(
            item.repetitions < protocol.repetitions
            or item.confidence_level != protocol.confidence_level
            for item in cell.metrics
        ):
            reasons.add(
                f"measured cell does not meet protocol repetition/confidence policy: {cell.cell_id}"
            )
        if (
            cell.outcome
            in {
                V2CellOutcome.NEGATIVE_RESULT,
                V2CellOutcome.FAILED,
                V2CellOutcome.UNKNOWN,
                V2CellOutcome.UNSUPPORTED,
            }
            and not cell.reason
        ):
            reasons.add(f"terminal cell lacks retained reason: {cell.cell_id}")
    if not any(cell.outcome is V2CellOutcome.NEGATIVE_RESULT for cell in cells):
        reasons.add("negative or inconclusive cells are not retained")
    if audit is None:
        reasons.add("independent audit and replay are missing")
    elif audit.outcome != "clean" or not audit.replay_passed or audit.critical_findings:
        reasons.add("independent audit or replay has unresolved critical evidence defects")

    supported_claim_ids: list[str] = []
    for claim in claims:
        if not claim.published:
            continue
        candidates = [by_id.get(item) for item in claim.candidate_cell_ids]
        baselines = [by_id.get(item) for item in claim.baseline_cell_ids]
        if any(item is None for item in (*candidates, *baselines)):
            reasons.add(f"claim references a missing cell: {claim.claim_id}")
            continue
        candidate_cells = [item for item in candidates if item is not None]
        baseline_cells = [item for item in baselines if item is not None]
        if any(
            item.outcome is not V2CellOutcome.MEASURED
            or item.artifact.state is not ArtifactAvailability.AVAILABLE
            for item in (*candidate_cells, *baseline_cells)
        ):
            reasons.add(f"claim is not tied to measured deployable cells: {claim.claim_id}")
            continue
        candidate_interval = _aggregate_interval(candidate_cells, claim.metric)
        baseline_interval = _aggregate_interval(baseline_cells, claim.metric)
        if candidate_interval is None or baseline_interval is None:
            reasons.add(f"claim lacks confidence-bounded metric evidence: {claim.claim_id}")
            continue
        supported = (
            candidate_interval[0] >= baseline_interval[1] + claim.margin
            if claim.direction == "higher"
            else candidate_interval[1] <= baseline_interval[0] - claim.margin
        )
        if not supported:
            reasons.add(f"confidence intervals do not support claim: {claim.claim_id}")
            continue
        supported_claim_ids.append(claim.claim_id)
    outcome = V2PublicationOutcome.PUBLISH if not reasons else V2PublicationOutcome.WITHHOLD
    return V2PublicationDecision(
        outcome, tuple(sorted(reasons)), tuple(sorted(supported_claim_ids))
    )


def _placeholder_artifact() -> ReferenceArtifact:
    digest = "0" * 64
    stages = tuple(
        ArtifactLineageStage(stage, "identity", digest, digest, digest) for stage in _LINEAGE_STAGES
    )
    return ReferenceArtifact(
        "identity",
        "identity",
        "identity",
        ArtifactAvailability.NOT_AVAILABLE,
        None,
        None,
        ReferenceArtifactLineage(stages, "not-published", False),
        "identity only",
    )


def render_v2_protocol(protocol: V2BenchmarkProtocol, *, format: str = "json") -> str:
    """Render only the preregistered contract; no benchmark result is implied."""

    if format == "json":
        return (
            canonical_identity_json(
                {"protocol_id": protocol.protocol_id, "protocol": protocol.to_record()}
            )
            + "\n"
        )
    if format == "markdown":
        return "\n".join(
            [
                f"# v2 benchmark protocol `{protocol.protocol_id}`",
                "",
                "This is a preregistration contract. It contains no live benchmark result.",
                "",
                f"- Matrix cells: {protocol.expected_cell_count}",
                f"- Models: {len(protocol.models)} exact checkpoints across "
                f"{len({item.family for item in protocol.models})} families",
                f"- Seeds: {', '.join(str(item) for item in protocol.seeds)}",
                f"- Metrics: {', '.join(protocol.metrics)}",
                "",
            ]
        )
    raise V2BenchmarkError(f"unsupported v2 protocol format: {format}")


__all__ = [
    "V2_BENCHMARK_SCHEMA_VERSION",
    "V2_REQUIRED_METRICS",
    "ArtifactAvailability",
    "ArtifactLineageStage",
    "ReferenceArtifact",
    "ReferenceArtifactLineage",
    "V2AuditRecord",
    "V2BenchmarkBundle",
    "V2BenchmarkCell",
    "V2BenchmarkError",
    "V2BenchmarkProtocol",
    "V2Budget",
    "V2CellOutcome",
    "V2CompetitivenessClaim",
    "V2DatasetSpec",
    "V2HardwareProfile",
    "V2MethodSpec",
    "V2Metric",
    "V2ModelSpec",
    "V2PublicationDecision",
    "V2PublicationOutcome",
    "build_v2_audit_record",
    "build_v2_benchmark_bundle",
    "evaluate_v2_publication",
    "render_v2_protocol",
]

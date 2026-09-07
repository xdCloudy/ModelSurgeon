"""Versioned numerical and artifact stability evidence across execution platforms."""

from __future__ import annotations

import hashlib
import json
import math
import platform as platform_module
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

PLATFORM_STABILITY_SCHEMA_VERSION = 1
PLATFORM_TOLERANCE_REGISTRY_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PlatformStabilityError(ValueError):
    """Raised when stability evidence is incomplete or ambiguous."""


class StabilityStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class DriftOutcome(StrEnum):
    WITHIN_TOLERANCE = "within_tolerance"
    EXPECTED_VARIATION = "expected_variation"
    SEMANTIC_DRIFT = "semantic_drift"
    ARTIFACT_DRIFT = "artifact_drift"
    UNKNOWN = "unknown"


class ObservationState(StrEnum):
    MEASURED = "measured"
    UNAVAILABLE = "unavailable"


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PlatformStabilityError(f"{label} is required")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class MetricTolerance:
    """One metric's expected and semantic-drift tolerance provenance."""

    metric: str
    unit: str
    expected_absolute: float
    expected_relative: float
    semantic_absolute: float
    semantic_relative: float
    definition_revision: str = "metric-regression-v1"
    changelog_marker: str = "metric-schema-v1"

    def __post_init__(self) -> None:
        _require_text(self.metric, "metric name")
        _require_text(self.unit, "metric unit")
        _require_text(self.definition_revision, "metric definition revision")
        _require_text(self.changelog_marker, "metric changelog marker")
        values = (
            self.expected_absolute,
            self.expected_relative,
            self.semantic_absolute,
            self.semantic_relative,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise PlatformStabilityError("metric tolerances must be finite and non-negative")
        if self.semantic_absolute < self.expected_absolute:
            raise PlatformStabilityError("semantic absolute tolerance cannot be tighter")
        if self.semantic_relative < self.expected_relative:
            raise PlatformStabilityError("semantic relative tolerance cannot be tighter")

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "expected_absolute": self.expected_absolute,
            "expected_relative": self.expected_relative,
            "semantic_absolute": self.semantic_absolute,
            "semantic_relative": self.semantic_relative,
            "definition_revision": self.definition_revision,
            "changelog_marker": self.changelog_marker,
        }


@dataclass(frozen=True, slots=True)
class ToleranceRegistry:
    """Immutable metric definitions used by a stability comparison."""

    metrics: tuple[MetricTolerance, ...]
    version: int = PLATFORM_TOLERANCE_REGISTRY_VERSION

    def __post_init__(self) -> None:
        if self.version != PLATFORM_TOLERANCE_REGISTRY_VERSION:
            raise PlatformStabilityError("unsupported tolerance registry version")
        names = tuple(item.metric for item in self.metrics)
        if not names or names != tuple(sorted(set(names))):
            raise PlatformStabilityError("tolerance metrics must be non-empty and sorted")

    @property
    def registry_id(self) -> str:
        return hashlib.sha256(_canonical(self.to_record()).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {"version": self.version, "metrics": [item.to_record() for item in self.metrics]}


@dataclass(frozen=True, slots=True)
class PlatformIdentity:
    """Exact environment evidence retained with a comparison."""

    platform: str
    runtime: str
    dtype: str
    threads: int
    accelerator: str
    upstream_revision: str
    config_digest: str
    supported: bool = True
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (
            ("platform", self.platform),
            ("runtime", self.runtime),
            ("dtype", self.dtype),
            ("accelerator", self.accelerator),
            ("upstream revision", self.upstream_revision),
            ("configuration digest", self.config_digest),
        ):
            _require_text(value, label)
        if self.threads <= 0:
            raise PlatformStabilityError("thread count must be positive")
        if not isinstance(self.details, Mapping):
            raise PlatformStabilityError("platform details must be a mapping")

    def to_record(self) -> dict[str, object]:
        return {
            "platform": self.platform,
            "runtime": self.runtime,
            "dtype": self.dtype,
            "threads": self.threads,
            "accelerator": self.accelerator,
            "upstream_revision": self.upstream_revision,
            "config_digest": self.config_digest,
            "supported": self.supported,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class MetricObservation:
    name: str
    unit: str
    value: float | None
    state: ObservationState = ObservationState.MEASURED
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "observation name")
        _require_text(self.unit, "observation unit")
        if self.state is ObservationState.MEASURED:
            if self.value is None or not math.isfinite(self.value) or self.reason is not None:
                raise PlatformStabilityError("measured observations require one finite value")
        elif self.value is not None or not self.reason:
            raise PlatformStabilityError("unavailable observations require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "value": self.value,
            "state": self.state.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class MetricDrift:
    metric: str
    unit: str
    reference: float | None
    candidate: float | None
    delta: float | None
    allowed_absolute: float | None
    allowed_relative: float | None
    outcome: DriftOutcome
    reason: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "reference": self.reference,
            "candidate": self.candidate,
            "delta": self.delta,
            "allowed_absolute": self.allowed_absolute,
            "allowed_relative": self.allowed_relative,
            "outcome": self.outcome.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.digest) or self.size_bytes <= 0:
            raise PlatformStabilityError("artifact identity requires a SHA-256 and positive size")

    def to_record(self) -> dict[str, object]:
        return {"digest": self.digest, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class PlatformStabilityRecord:
    reference: PlatformIdentity
    candidate: PlatformIdentity
    status: StabilityStatus
    drift: DriftOutcome
    metrics: tuple[MetricDrift, ...]
    registry_id: str
    artifact_reference: ArtifactIdentity | None = None
    artifact_candidate: ArtifactIdentity | None = None
    status_reason: str | None = None
    schema_version: int = PLATFORM_STABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PLATFORM_STABILITY_SCHEMA_VERSION:
            raise PlatformStabilityError("unsupported platform stability schema version")
        _require_text(self.registry_id, "tolerance registry id")
        if self.status is StabilityStatus.SUPPORTED and self.status_reason is not None:
            raise PlatformStabilityError("supported records cannot carry a status reason")
        if self.status is not StabilityStatus.SUPPORTED and not self.status_reason:
            raise PlatformStabilityError("non-supported records require a status reason")

    @property
    def record_id(self) -> str:
        return (
            "platform_stability_"
            + hashlib.sha256(_canonical(self.to_record(False)).encode()).hexdigest()
        )

    def to_record(self, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "reference": self.reference.to_record(),
            "candidate": self.candidate.to_record(),
            "status": self.status.value,
            "drift": self.drift.value,
            "metrics": [item.to_record() for item in self.metrics],
            "registry_id": self.registry_id,
            "artifact_reference": None
            if self.artifact_reference is None
            else self.artifact_reference.to_record(),
            "artifact_candidate": None
            if self.artifact_candidate is None
            else self.artifact_candidate.to_record(),
            "status_reason": self.status_reason,
        }
        if include_id:
            record["record_id"] = self.record_id
        return record


def capture_platform_identity(
    *,
    dtype: str = "float32",
    threads: int = 1,
    accelerator: str = "cpu",
    upstream_revision: str = "unknown",
    config_digest: str = "unknown",
    supported: bool = True,
) -> PlatformIdentity:
    """Capture portable environment evidence without probing optional runtimes."""
    return PlatformIdentity(
        platform=sys.platform,
        runtime=f"python-{platform_module.python_version()}",
        dtype=dtype,
        threads=threads,
        accelerator=accelerator,
        upstream_revision=upstream_revision,
        config_digest=config_digest,
        supported=supported,
        details={"platform_release": platform_module.release()},
    )


def compare_platform_stability(
    reference: PlatformIdentity,
    candidate: PlatformIdentity,
    reference_metrics: tuple[MetricObservation, ...],
    candidate_metrics: tuple[MetricObservation, ...],
    registry: ToleranceRegistry,
    *,
    reference_artifact: ArtifactIdentity | None = None,
    candidate_artifact: ArtifactIdentity | None = None,
) -> PlatformStabilityRecord:
    """Compare a pair of runs while retaining unsupported and unknown cells."""
    ref_by_name = {item.name: item for item in reference_metrics}
    cand_by_name = {item.name: item for item in candidate_metrics}
    if len(ref_by_name) != len(reference_metrics) or len(cand_by_name) != len(candidate_metrics):
        raise PlatformStabilityError("metric observations must be unique by name")
    drifts: list[MetricDrift] = []
    for tolerance in registry.metrics:
        ref = ref_by_name.get(tolerance.metric)
        cand = cand_by_name.get(tolerance.metric)
        if (
            ref is None
            or cand is None
            or ref.state is not ObservationState.MEASURED
            or cand.state is not ObservationState.MEASURED
        ):
            drifts.append(
                MetricDrift(
                    tolerance.metric,
                    tolerance.unit,
                    None,
                    None,
                    None,
                    None,
                    None,
                    DriftOutcome.UNKNOWN,
                    "metric unavailable",
                )
            )
            continue
        if ref.unit != tolerance.unit or cand.unit != tolerance.unit:
            drifts.append(
                MetricDrift(
                    tolerance.metric,
                    tolerance.unit,
                    ref.value,
                    cand.value,
                    None,
                    None,
                    None,
                    DriftOutcome.UNKNOWN,
                    "metric unit mismatch",
                )
            )
            continue
        assert ref.value is not None and cand.value is not None
        delta = abs(cand.value - ref.value)
        relative = delta / max(abs(ref.value), 1e-12)
        if delta <= tolerance.expected_absolute or relative <= tolerance.expected_relative:
            outcome = DriftOutcome.WITHIN_TOLERANCE
        elif delta <= tolerance.semantic_absolute or relative <= tolerance.semantic_relative:
            outcome = DriftOutcome.EXPECTED_VARIATION
        else:
            outcome = DriftOutcome.SEMANTIC_DRIFT
        drifts.append(
            MetricDrift(
                tolerance.metric,
                tolerance.unit,
                ref.value,
                cand.value,
                delta,
                tolerance.semantic_absolute,
                tolerance.semantic_relative,
                outcome,
            )
        )
    artifact_drift = (
        reference_artifact is not None
        and candidate_artifact is not None
        and reference_artifact != candidate_artifact
    )
    if artifact_drift:
        overall_drift = DriftOutcome.ARTIFACT_DRIFT
    elif any(item.outcome is DriftOutcome.SEMANTIC_DRIFT for item in drifts):
        overall_drift = DriftOutcome.SEMANTIC_DRIFT
    elif any(item.outcome is DriftOutcome.EXPECTED_VARIATION for item in drifts):
        overall_drift = DriftOutcome.EXPECTED_VARIATION
    elif any(item.outcome is DriftOutcome.UNKNOWN for item in drifts):
        overall_drift = DriftOutcome.UNKNOWN
    else:
        overall_drift = DriftOutcome.WITHIN_TOLERANCE
    if not candidate.supported:
        status, reason = StabilityStatus.UNSUPPORTED, "candidate platform is unsupported"
    elif overall_drift is DriftOutcome.UNKNOWN:
        status, reason = (
            StabilityStatus.UNKNOWN,
            "comparison has unavailable or incompatible evidence",
        )
    elif overall_drift in {DriftOutcome.SEMANTIC_DRIFT, DriftOutcome.ARTIFACT_DRIFT}:
        status, reason = StabilityStatus.FAILED, f"comparison failed with {overall_drift.value}"
    else:
        status, reason = StabilityStatus.SUPPORTED, None
    return PlatformStabilityRecord(
        reference,
        candidate,
        status,
        overall_drift,
        tuple(drifts),
        registry.registry_id,
        reference_artifact,
        candidate_artifact,
        reason,
    )


def migrate_platform_stability_record(raw: Mapping[str, object]) -> dict[str, object]:
    """Migrate v0 evidence additively and reject unknown versions."""
    if not isinstance(raw, Mapping):
        raise PlatformStabilityError("stability record must be an object")
    version = raw.get("schema_version", 0)
    if version == PLATFORM_STABILITY_SCHEMA_VERSION:
        return cast(dict[str, object], json.loads(_canonical(raw)))
    if version != 0:
        raise PlatformStabilityError(
            f"unsupported platform stability migration version: {version!r}"
        )
    migrated = dict(raw)
    migrated["schema_version"] = PLATFORM_STABILITY_SCHEMA_VERSION
    migrated.setdefault("status_reason", "migrated legacy stability record")
    migrated.setdefault("drift", DriftOutcome.UNKNOWN.value)
    migrated["migration"] = {"from_schema_version": 0, "to_schema_version": 1, "lossless": True}
    return cast(dict[str, object], json.loads(_canonical(migrated)))

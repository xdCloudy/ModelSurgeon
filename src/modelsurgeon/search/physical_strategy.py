"""Measured final repair/quantization/export strategy selection."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum

PHYSICAL_STRATEGY_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUIRED_PROVENANCE = frozenset(
    {"repair", "teacher", "dataset", "quantization", "artifact", "deployment"}
)


class PhysicalStrategyError(ValueError):
    """Raised when final strategy evidence is incomplete or ambiguous."""


class PhysicalStrategyOutcome(StrEnum):
    SELECTED = "selected"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class PhysicalStrategyKind(StrEnum):
    NO_REPAIR = "no_repair"
    QUANTIZATION_ONLY = "quantization_only"
    REPAIR = "repair"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhysicalStrategyError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise PhysicalStrategyError(f"{label} must be a lowercase SHA-256")
    return result


def _nonnegative(value: float | int, label: str) -> None:
    if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
        raise PhysicalStrategyError(f"{label} must be finite and non-negative")


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise PhysicalStrategyError(f"{label} must be positive")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class PhysicalStrategyCandidate:
    candidate_id: str
    kind: PhysicalStrategyKind
    measured: bool
    compatible: bool
    constraints_passed: bool
    quality: float | None
    deployment_cost: float | None
    optimization_cost: float | None
    artifact_bytes: int | None
    source_artifact_digest: str
    artifact_digest: str | None
    provenance: tuple[str, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.candidate_id, "physical strategy candidate ID")
        for value, label in (
            (self.quality, "quality"),
            (self.deployment_cost, "deployment cost"),
            (self.optimization_cost, "optimization cost"),
        ):
            if value is not None:
                _nonnegative(value, label)
        if self.artifact_bytes is not None:
            _positive(self.artifact_bytes, "artifact bytes")
        _digest(self.source_artifact_digest, "source artifact digest")
        if self.artifact_digest is not None:
            _digest(self.artifact_digest, "artifact digest")
        if self.provenance != tuple(sorted(set(self.provenance))):
            raise PhysicalStrategyError("strategy provenance must be sorted and unique")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind.value,
            "measured": self.measured,
            "compatible": self.compatible,
            "constraints_passed": self.constraints_passed,
            "quality": self.quality,
            "deployment_cost": self.deployment_cost,
            "optimization_cost": self.optimization_cost,
            "artifact_bytes": self.artifact_bytes,
            "source_artifact_digest": self.source_artifact_digest,
            "artifact_digest": self.artifact_digest,
            "provenance": list(self.provenance),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PhysicalStrategyRequest:
    source_artifact_digest: str
    min_quality: float
    max_deployment_cost: float
    max_optimization_cost: float
    max_artifact_bytes: int
    required_controls: tuple[PhysicalStrategyKind, ...] = (
        PhysicalStrategyKind.NO_REPAIR,
        PhysicalStrategyKind.QUANTIZATION_ONLY,
    )

    def __post_init__(self) -> None:
        _digest(self.source_artifact_digest, "request source artifact digest")
        if not 0 <= self.min_quality <= 1:
            raise PhysicalStrategyError("minimum quality must be within [0, 1]")
        _nonnegative(self.max_deployment_cost, "maximum deployment cost")
        _nonnegative(self.max_optimization_cost, "maximum optimization cost")
        _positive(self.max_artifact_bytes, "maximum artifact bytes")
        if (
            not self.required_controls
            or len(set(self.required_controls)) != len(self.required_controls)
        ):
            raise PhysicalStrategyError("required controls must be unique and non-empty")


@dataclass(frozen=True, slots=True)
class PhysicalStrategyDecision:
    outcome: PhysicalStrategyOutcome
    selected: PhysicalStrategyCandidate | None
    measured_controls: tuple[PhysicalStrategyKind, ...]
    excluded: tuple[tuple[str, str], ...]
    reason: str
    decision_id: str

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": PHYSICAL_STRATEGY_SCHEMA_VERSION,
            "outcome": self.outcome.value,
            "selected": None if self.selected is None else self.selected.to_record(),
            "measured_controls": [item.value for item in self.measured_controls],
            "excluded": [{"candidate_id": key, "reason": value} for key, value in self.excluded],
            "reason": self.reason,
            "decision_id": self.decision_id,
        }


def select_physical_strategy(
    request: PhysicalStrategyRequest,
    candidates: tuple[PhysicalStrategyCandidate, ...],
) -> PhysicalStrategyDecision:
    """Select only measured, compatible, budgeted, fully proven final artifacts."""

    if not candidates:
        raise PhysicalStrategyError("physical strategy selection requires candidates")
    by_kind = {
        kind: tuple(item for item in candidates if item.kind is kind)
        for kind in PhysicalStrategyKind
    }
    exclusions: list[tuple[str, str]] = []
    eligible: list[PhysicalStrategyCandidate] = []
    for candidate in candidates:
        if not candidate.compatible:
            exclusions.append((candidate.candidate_id, "strategy is incompatible"))
            continue
        if not candidate.measured or candidate.quality is None:
            exclusions.append((candidate.candidate_id, "predicted or incomplete evidence"))
            continue
        if not candidate.constraints_passed:
            exclusions.append((candidate.candidate_id, "measured hard constraints failed"))
            continue
        if candidate.source_artifact_digest != request.source_artifact_digest:
            exclusions.append((candidate.candidate_id, "source artifact identity changed"))
            continue
        if (
            candidate.artifact_digest is None
            or candidate.artifact_digest == request.source_artifact_digest
        ):
            exclusions.append(
                (candidate.candidate_id, "final artifact is not a distinct immutable child")
            )
            continue
        if candidate.deployment_cost is None or candidate.optimization_cost is None:
            exclusions.append((candidate.candidate_id, "measured cost evidence is incomplete"))
            continue
        if candidate.artifact_bytes is None:
            exclusions.append((candidate.candidate_id, "artifact size evidence is incomplete"))
            continue
        if candidate.quality < request.min_quality:
            exclusions.append((candidate.candidate_id, "quality constraint failed"))
            continue
        if candidate.deployment_cost > request.max_deployment_cost:
            exclusions.append((candidate.candidate_id, "deployment budget exceeded"))
            continue
        if candidate.optimization_cost > request.max_optimization_cost:
            exclusions.append((candidate.candidate_id, "optimization budget exceeded"))
            continue
        if candidate.artifact_bytes > request.max_artifact_bytes:
            exclusions.append((candidate.candidate_id, "artifact budget exceeded"))
            continue
        if not _REQUIRED_PROVENANCE.issubset(
            {item.split("=", 1)[0] for item in candidate.provenance}
        ):
            exclusions.append(
                (
                    candidate.candidate_id,
                    "repair/teacher/data/quantization/artifact/deployment provenance incomplete",
                )
            )
            continue
        eligible.append(candidate)
    measured_controls = tuple(
        kind for kind in request.required_controls if any(item.kind is kind for item in eligible)
    )
    missing_controls = tuple(
        kind for kind in request.required_controls if kind not in measured_controls
    )
    if missing_controls:
        missing_candidates = [
            item for kind in missing_controls for item in by_kind[kind]
        ]
        if any(not item.compatible for item in missing_candidates):
            outcome = PhysicalStrategyOutcome.UNSUPPORTED
            reason = "required control is incompatible with the deployment"
        elif not missing_candidates or any(
            not item.measured or item.quality is None for item in missing_candidates
        ):
            outcome = PhysicalStrategyOutcome.UNKNOWN
            reason = "required controls lack measured evidence"
        else:
            outcome = PhysicalStrategyOutcome.FAILED
            reason = "required controls failed measured constraints or budgets"
        return _decision(request, outcome, None, measured_controls, exclusions, reason)
    if not eligible:
        return _decision(
            request,
            PhysicalStrategyOutcome.FAILED,
            None,
            measured_controls,
            exclusions,
            "no measured final strategy satisfies constraints and budgets",
        )
    selected = min(
        eligible,
        key=_ranking_key,
    )
    return _decision(
        request,
        PhysicalStrategyOutcome.SELECTED,
        selected,
        measured_controls,
        exclusions,
        "selected measured strategy by quality, deployment cost, "
        "optimization cost, and artifact size",
    )


def _ranking_key(
    item: PhysicalStrategyCandidate,
) -> tuple[float, float, float, int, str]:
    assert item.quality is not None
    assert item.deployment_cost is not None
    assert item.optimization_cost is not None
    assert item.artifact_bytes is not None
    return (
        -item.quality,
        item.deployment_cost,
        item.optimization_cost,
        item.artifact_bytes,
        item.candidate_id,
    )
def _decision(
    request: PhysicalStrategyRequest,
    outcome: PhysicalStrategyOutcome,
    selected: PhysicalStrategyCandidate | None,
    measured_controls: tuple[PhysicalStrategyKind, ...],
    excluded: list[tuple[str, str]],
    reason: str,
) -> PhysicalStrategyDecision:
    record = {
        "request": {
            "source_artifact_digest": request.source_artifact_digest,
            "min_quality": request.min_quality,
            "max_deployment_cost": request.max_deployment_cost,
            "max_optimization_cost": request.max_optimization_cost,
            "max_artifact_bytes": request.max_artifact_bytes,
        },
        "outcome": outcome.value,
        "selected": None if selected is None else selected.to_record(),
        "reason": reason,
    }
    decision_id = f"physical_decision_{hashlib.sha256(_canonical(record).encode()).hexdigest()}"
    return PhysicalStrategyDecision(
        outcome,
        selected,
        measured_controls,
        tuple(sorted(excluded)),
        reason,
        decision_id,
    )


__all__ = [
    "PHYSICAL_STRATEGY_SCHEMA_VERSION",
    "PhysicalStrategyCandidate",
    "PhysicalStrategyDecision",
    "PhysicalStrategyError",
    "PhysicalStrategyKind",
    "PhysicalStrategyOutcome",
    "PhysicalStrategyRequest",
    "select_physical_strategy",
]

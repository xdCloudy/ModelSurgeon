"""Hardware-aware candidate identity, conservative feasibility, and ranking."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.config import ObjectiveDirection

HARDWARE_OBJECTIVE_SCHEMA_VERSION: Final[int] = 1


class HardwareObjectiveError(ValueError):
    """Raised when hardware evidence or deployment objectives are ambiguous."""


class HardwareMetric(StrEnum):
    """Metrics that are directly relevant to a deployment placement."""

    LATENCY_SECONDS = "latency_seconds"
    PEAK_RAM_BYTES = "peak_ram_bytes"
    PEAK_VRAM_BYTES = "peak_vram_bytes"
    DISK_BYTES = "disk_bytes"
    THROUGHPUT_TOKENS_PER_SECOND = "throughput_tokens_per_second"
    PARAMETER_COUNT = "parameter_count"


class EvidenceStatus(StrEnum):
    MEASURED = "measured"
    PREDICTED = "predicted"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class ConstraintDirection(StrEnum):
    MAXIMUM = "maximum"
    MINIMUM = "minimum"


class HardwareDecisionStatus(StrEnum):
    PROMOTABLE = "promotable"
    PREDICTED_ONLY = "predicted_only"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HardwareObjectiveError(f"{label} is required")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HardwareObjectiveError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HardwareObjectiveError(f"{label} must be finite")
    return result


def _nonnegative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0:
        raise HardwareObjectiveError(f"{label} must be non-negative")
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class OffloadBoundary:
    """A measured or proposed CPU/GPU placement boundary."""

    profile_id: str
    runtime: str
    cpu_threads: int
    gpu_available: bool
    gpu_offload_layers: int = 0
    offload_fraction: float = 0.0

    def __post_init__(self) -> None:
        _text(self.profile_id, "hardware profile ID")
        _text(self.runtime, "runtime")
        if self.cpu_threads <= 0:
            raise HardwareObjectiveError("CPU threads must be positive")
        if self.gpu_offload_layers < 0:
            raise HardwareObjectiveError("GPU offload layers must be non-negative")
        fraction = _finite(self.offload_fraction, "offload fraction")
        if not 0 <= fraction <= 1:
            raise HardwareObjectiveError("offload fraction must be within [0, 1]")
        if not self.gpu_available and (self.gpu_offload_layers or fraction):
            raise HardwareObjectiveError("CPU-only profiles cannot carry GPU offload")

    @property
    def boundary_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"placement_{digest}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": HARDWARE_OBJECTIVE_SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "runtime": self.runtime,
            "cpu_threads": self.cpu_threads,
            "gpu_available": self.gpu_available,
            "gpu_offload_layers": self.gpu_offload_layers,
            "offload_fraction": self.offload_fraction,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._identity_record(), "boundary_id": self.boundary_id}


@dataclass(frozen=True, slots=True)
class HardwareEvidence:
    """One value with an uncertainty interval and explicit provenance status."""

    metric: HardwareMetric
    value: float
    status: EvidenceStatus
    uncertainty: float = 0.0
    source_id: str = ""

    def __post_init__(self) -> None:
        _nonnegative(self.value, "hardware evidence value")
        _nonnegative(self.uncertainty, "hardware evidence uncertainty")
        if self.status in {EvidenceStatus.MEASURED, EvidenceStatus.PREDICTED}:
            _text(self.source_id, "hardware evidence source ID")
        elif self.source_id and not self.source_id.strip():
            raise HardwareObjectiveError("hardware evidence source ID cannot be blank")

    @property
    def conservative_upper_bound(self) -> float:
        return self.value + self.uncertainty

    @property
    def conservative_lower_bound(self) -> float:
        return max(0.0, self.value - self.uncertainty)

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "value": self.value,
            "status": self.status.value,
            "uncertainty": self.uncertainty,
            "source_id": self.source_id,
        }


@dataclass(frozen=True, slots=True)
class HardwareConstraint:
    """A hard deployment constraint evaluated with conservative bounds."""

    metric: HardwareMetric
    direction: ConstraintDirection
    threshold: float
    measured_required: bool = True

    def __post_init__(self) -> None:
        _nonnegative(self.threshold, "hardware constraint threshold")

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "direction": self.direction.value,
            "threshold": self.threshold,
            "measured_required": self.measured_required,
        }


@dataclass(frozen=True, slots=True)
class DeploymentObjective:
    """A weighted deployment objective with an explicit normalization reference."""

    metric: HardwareMetric
    direction: ObjectiveDirection
    reference: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        _nonnegative(self.reference, "objective reference")
        if self.reference == 0:
            raise HardwareObjectiveError("objective reference must be non-zero")
        weight = _finite(self.weight, "objective weight")
        if weight <= 0:
            raise HardwareObjectiveError("objective weight must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "direction": self.direction.value,
            "reference": self.reference,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class HardwareCandidate:
    """Candidate identity including structure, hardware profile, and placement."""

    structure_identity: str
    parent_checkpoint_id: str
    placement: OffloadBoundary
    predicted_evidence: tuple[HardwareEvidence, ...] = ()
    measured_evidence: tuple[HardwareEvidence, ...] = ()
    unknown_evidence: tuple[HardwareEvidence, ...] = ()

    def __post_init__(self) -> None:
        _text(self.structure_identity, "structure identity")
        if not self.parent_checkpoint_id.startswith("checkpoint_"):
            raise HardwareObjectiveError(
                "hardware candidates require an accepted parent checkpoint"
            )
        for label, evidence in (
            ("predicted", self.predicted_evidence),
            ("measured", self.measured_evidence),
            ("unknown", self.unknown_evidence),
        ):
            metrics = tuple(item.metric for item in evidence)
            if metrics != tuple(sorted(set(metrics), key=lambda item: item.value)):
                raise HardwareObjectiveError(
                    f"{label} hardware evidence must be unique and canonical"
                )
            if label == "predicted" and any(
                item.status is not EvidenceStatus.PREDICTED for item in evidence
            ):
                raise HardwareObjectiveError("predicted evidence must have predicted status")
            if label == "measured" and any(
                item.status is not EvidenceStatus.MEASURED for item in evidence
            ):
                raise HardwareObjectiveError("measured evidence must have measured status")
            if label == "unknown" and any(
                item.status is not EvidenceStatus.UNKNOWN for item in evidence
            ):
                raise HardwareObjectiveError("unknown evidence must have unknown status")

    @property
    def candidate_id(self) -> str:
        payload = {
            "schema_version": HARDWARE_OBJECTIVE_SCHEMA_VERSION,
            "structure_identity": self.structure_identity,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "placement": self.placement.to_record(),
        }
        return f"candidate_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": HARDWARE_OBJECTIVE_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "structure_identity": self.structure_identity,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "placement": self.placement.to_record(),
            "predicted_evidence": [item.to_record() for item in self.predicted_evidence],
            "measured_evidence": [item.to_record() for item in self.measured_evidence],
            "unknown_evidence": [item.to_record() for item in self.unknown_evidence],
        }


@dataclass(frozen=True, slots=True)
class HardwareConstraintResult:
    constraint: HardwareConstraint
    evidence: HardwareEvidence | None
    conservative_value: float | None
    passed: bool
    reason: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "constraint": self.constraint.to_record(),
            "evidence": None if self.evidence is None else self.evidence.to_record(),
            "conservative_value": self.conservative_value,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class HardwareCandidateAssessment:
    candidate_id: str
    status: HardwareDecisionStatus
    constraint_results: tuple[HardwareConstraintResult, ...]
    objective_score: float | None
    reason: str

    @property
    def promotable(self) -> bool:
        return self.status is HardwareDecisionStatus.PROMOTABLE

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": HARDWARE_OBJECTIVE_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "promotable": self.promotable,
            "constraint_results": [item.to_record() for item in self.constraint_results],
            "objective_score": self.objective_score,
            "reason": self.reason,
        }


def _evidence_by_metric(
    evidence: tuple[HardwareEvidence, ...],
) -> dict[HardwareMetric, HardwareEvidence]:
    return {item.metric: item for item in evidence}


def _conservative_value(
    evidence: HardwareEvidence, direction: ConstraintDirection
) -> float:
    return (
        evidence.conservative_upper_bound
        if direction is ConstraintDirection.MAXIMUM
        else evidence.conservative_lower_bound
    )


def _score(
    objectives: tuple[DeploymentObjective, ...],
    evidence: dict[HardwareMetric, HardwareEvidence],
) -> float | None:
    contributions: list[float] = []
    for objective in objectives:
        item = evidence.get(objective.metric)
        if item is None or item.status is not EvidenceStatus.MEASURED:
            return None
        value = item.value / objective.reference
        signed = value if objective.direction is ObjectiveDirection.MAXIMIZE else -value
        contributions.append(objective.weight * signed)
    if not objectives:
        return None
    total_weight = math.fsum(item.weight for item in objectives)
    return math.fsum(contributions) / total_weight


def assess_hardware_candidate(
    candidate: HardwareCandidate,
    constraints: tuple[HardwareConstraint, ...],
    objectives: tuple[DeploymentObjective, ...] = (),
) -> HardwareCandidateAssessment:
    """Assess one placement, refusing promotion without measured hard evidence."""

    if len({item.metric for item in constraints}) != len(constraints):
        raise HardwareObjectiveError("hardware constraints must be unique by metric")
    if len({item.metric for item in objectives}) != len(objectives):
        raise HardwareObjectiveError("deployment objectives must be unique by metric")
    predicted = _evidence_by_metric(candidate.predicted_evidence)
    measured = _evidence_by_metric(candidate.measured_evidence)
    unknown = _evidence_by_metric(candidate.unknown_evidence)
    results: list[HardwareConstraintResult] = []
    for constraint in constraints:
        item = (
            measured.get(constraint.metric)
            or predicted.get(constraint.metric)
            or unknown.get(constraint.metric)
        )
        if item is None or item.status in {EvidenceStatus.UNKNOWN, EvidenceStatus.UNSUPPORTED}:
            missing_reason = (
                "missing_evidence"
                if item is None
                else f"{item.status.value}_evidence"
            )
            results.append(
                HardwareConstraintResult(constraint, item, None, False, missing_reason)
            )
            continue
        conservative = _conservative_value(item, constraint.direction)
        passed = (
            conservative <= constraint.threshold
            if constraint.direction is ConstraintDirection.MAXIMUM
            else conservative >= constraint.threshold
        )
        reason: str | None = None if passed else "threshold_violation"
        if passed and constraint.measured_required and item.status is not EvidenceStatus.MEASURED:
            passed = False
            reason = "measured_hard_constraint_required"
        results.append(HardwareConstraintResult(constraint, item, conservative, passed, reason))
    objective_score = _score(objectives, measured)
    if any(
        item.evidence is not None and item.evidence.status is EvidenceStatus.UNSUPPORTED
        for item in results
    ):
        status = HardwareDecisionStatus.UNSUPPORTED
        reason = "unsupported_hardware_evidence"
    elif any(
        item.evidence is not None and item.evidence.status is EvidenceStatus.UNKNOWN
        for item in results
    ):
        status = HardwareDecisionStatus.UNKNOWN
        reason = "unknown_hardware_evidence"
    elif any(item.reason == "threshold_violation" for item in results):
        status = HardwareDecisionStatus.INFEASIBLE
        reason = "hard_constraint_violation"
    elif any(
        item.reason in {"missing_evidence", "measured_hard_constraint_required"}
        for item in results
    ):
        status = HardwareDecisionStatus.PREDICTED_ONLY
        reason = "measured_hard_constraint_evidence_required"
    elif objectives and objective_score is None:
        status = HardwareDecisionStatus.PREDICTED_ONLY
        reason = "measured_objective_evidence_required"
    else:
        status = HardwareDecisionStatus.PROMOTABLE
        reason = "measured_constraints_and_objectives_pass"
    return HardwareCandidateAssessment(
        candidate.candidate_id, status, tuple(results), objective_score, reason
    )


@dataclass(frozen=True, slots=True)
class HardwareSearchReport:
    """Durable report retaining placement, assessment, and lineage context."""

    search_id: str
    profile: OffloadBoundary
    candidates: tuple[HardwareCandidate, ...]
    assessments: tuple[HardwareCandidateAssessment, ...]
    schema_version: int = HARDWARE_OBJECTIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.search_id, "hardware search ID")
        if self.schema_version != HARDWARE_OBJECTIVE_SCHEMA_VERSION:
            raise HardwareObjectiveError("unsupported hardware search report schema")
        ids = tuple(item.candidate_id for item in self.candidates)
        if len(ids) != len(set(ids)):
            raise HardwareObjectiveError("hardware report candidates must be unique")
        assessment_ids = tuple(item.candidate_id for item in self.assessments)
        if set(assessment_ids) != set(ids):
            raise HardwareObjectiveError("hardware report assessments must match candidates")
        if any(item.placement != self.profile for item in self.candidates):
            raise HardwareObjectiveError("hardware report profile must match every candidate")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "search_id": self.search_id,
            "profile": self.profile.to_record(),
            "candidates": [item.to_record() for item in self.candidates],
            "assessments": [item.to_record() for item in self.assessments],
        }


@dataclass(frozen=True, slots=True)
class HardwareSearchResume:
    """Resume identity retaining the exact placement and candidate lineage."""

    search_id: str
    generation: int
    profile: OffloadBoundary
    frontier_candidate_ids: tuple[str, ...]
    selected_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.search_id, "hardware search ID")
        if self.generation < 0:
            raise HardwareObjectiveError("hardware resume generation cannot be negative")
        for label, values in (
            ("frontier", self.frontier_candidate_ids),
            ("selected", self.selected_candidate_ids),
        ):
            if any(not value.startswith("candidate_") for value in values):
                raise HardwareObjectiveError(f"{label} candidate IDs must be canonical")
            if len(values) != len(set(values)):
                raise HardwareObjectiveError(f"{label} candidate IDs must be unique")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": HARDWARE_OBJECTIVE_SCHEMA_VERSION,
            "search_id": self.search_id,
            "generation": self.generation,
            "profile": self.profile.to_record(),
            "frontier_candidate_ids": list(self.frontier_candidate_ids),
            "selected_candidate_ids": list(self.selected_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class HardwareLineageRecord:
    """Accepted or rejected candidate lineage with placement retained."""

    candidate: HardwareCandidate
    parent_checkpoint_id: str
    accepted_checkpoint_id: str | None
    assessment: HardwareCandidateAssessment

    def __post_init__(self) -> None:
        if self.parent_checkpoint_id != self.candidate.parent_checkpoint_id:
            raise HardwareObjectiveError("lineage parent does not match candidate identity")
        if self.accepted_checkpoint_id is not None and not self.accepted_checkpoint_id.startswith(
            "checkpoint_"
        ):
            raise HardwareObjectiveError("accepted checkpoint IDs must be canonical")
        if self.accepted_checkpoint_id is not None and not self.assessment.promotable:
            raise HardwareObjectiveError(
                "non-promotable candidates cannot have accepted checkpoints"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": HARDWARE_OBJECTIVE_SCHEMA_VERSION,
            "candidate": self.candidate.to_record(),
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "accepted_checkpoint_id": self.accepted_checkpoint_id,
            "assessment": self.assessment.to_record(),
        }


def rank_hardware_candidates(
    candidates: tuple[HardwareCandidate, ...],
    constraints: tuple[HardwareConstraint, ...],
    objectives: tuple[DeploymentObjective, ...] = (),
) -> tuple[tuple[HardwareCandidate, HardwareCandidateAssessment], ...]:
    """Rank measured, feasible candidates while retaining predicted candidates for evaluation."""

    assessed = tuple(
        (candidate, assess_hardware_candidate(candidate, constraints, objectives))
        for candidate in candidates
    )
    return tuple(
        sorted(
            assessed,
            key=lambda item: (
                not item[1].promotable,
                -(
                    item[1].objective_score
                    if item[1].objective_score is not None
                    else float("-inf")
                ),
                item[0].candidate_id,
            ),
        )
    )

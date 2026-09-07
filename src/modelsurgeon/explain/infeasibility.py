"""Measured feasibility and fail-closed infeasibility explanations.

This module is deliberately a read-only projection over typed evidence.  It
never promotes predictions to measurements, changes a hard constraint, or
discards a negative terminal result.  Candidates are canonicalized before
evaluation so the same archive always produces the same explanation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import cast

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    HardConstraint,
    MetricObservation,
    ObjectiveContract,
)

FEASIBILITY_EXPLANATION_SCHEMA_VERSION = 1


class FeasibilityExplanationError(ValueError):
    """Raised when canonical evidence cannot support a safe explanation."""


class FeasibilityOutcome(StrEnum):
    """The evidence-backed state of a contract's hard constraints."""

    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    MISSING_EVIDENCE = "missing_evidence"
    PREDICTED_ONLY = "predicted_only"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"
    INCONCLUSIVE = "inconclusive"


class CandidateEvidenceStatus(StrEnum):
    """Evidence status; only ``MEASURED`` may evaluate hard constraints."""

    MEASURED = "measured"
    PREDICTED = "predicted"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"
    INCONCLUSIVE = "inconclusive"


class CandidateDisposition(StrEnum):
    """Terminal disposition retained independently of measurement status."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    UNKNOWN = "unknown"
    INCONCLUSIVE = "inconclusive"


class NextActionKind(StrEnum):
    """Safe, explicit follow-up actions emitted by an explanation."""

    NO_ACTION = "no_action"
    COLLECT_MEASUREMENTS = "collect_measurements"
    MEASURE_PREDICTED_CANDIDATES = "measure_predicted_candidates"
    RETRY_FAILED_EVIDENCE = "retry_failed_evidence"
    COMPLETE_INCONCLUSIVE_EVALUATION = "complete_inconclusive_evaluation"
    RESOLVE_UNSUPPORTED_CAPABILITY = "resolve_unsupported_capability"
    REVIEW_MEASURED_NEAR_MISSES = "review_measured_near_misses"
    PROPOSE_EXPLICIT_OBJECTIVE_AMENDMENT = "propose_explicit_objective_amendment"


def _canonical(value: object) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FeasibilityExplanationError("value is not canonical JSON") from error


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FeasibilityExplanationError(f"{label} is required")
    return value


def _identifier(value: object, label: str, prefix: str | None = None) -> str:
    result = _text(value, label)
    if prefix is not None and not result.startswith(prefix):
        raise FeasibilityExplanationError(f"{label} must start with {prefix}")
    return result


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if (
        len(result) != 71
        or not result.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in result[7:])
    ):
        raise FeasibilityExplanationError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeasibilityExplanationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise FeasibilityExplanationError(f"{label} must be finite and non-negative")
    return result


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FeasibilityExplanationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FeasibilityExplanationError(f"{label} must be finite")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FeasibilityExplanationError(f"{label} must be a non-negative integer")
    return value


def _mapping(value: Mapping[str, object], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise FeasibilityExplanationError(f"{label} must be a JSON object")
    try:
        decoded = json.loads(_canonical(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise FeasibilityExplanationError(f"{label} must contain canonical JSON values") from error
    if not isinstance(decoded, dict):
        raise FeasibilityExplanationError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)


def _sorted_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(sorted(set(values)))
    if len(result) != len(values):
        raise FeasibilityExplanationError(f"{label} must be unique")
    if result != tuple(values):
        raise FeasibilityExplanationError(f"{label} must be sorted")
    if any(not value.strip() for value in result):
        raise FeasibilityExplanationError(f"{label} cannot contain blank values")
    return result


@dataclass(frozen=True, slots=True)
class FeasibilityResourceBounds:
    """Hard limits for explanation input and output work."""

    max_candidates: int = 10_000
    max_near_misses: int = 20
    max_next_actions: int = 8
    max_output_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        for value, label in (
            (self.max_candidates, "maximum candidates"),
            (self.max_near_misses, "maximum near misses"),
            (self.max_next_actions, "maximum next actions"),
            (self.max_output_bytes, "maximum output bytes"),
        ):
            if _nonnegative_int(value, label) == 0:
                raise FeasibilityExplanationError(f"{label} must be positive")

    def to_record(self) -> dict[str, int]:
        return {
            "max_candidates": self.max_candidates,
            "max_near_misses": self.max_near_misses,
            "max_next_actions": self.max_next_actions,
            "max_output_bytes": self.max_output_bytes,
        }


_DEFAULT_RESOURCE_BOUNDS = FeasibilityResourceBounds()


@dataclass(frozen=True, slots=True)
class FeasibilityResourceUsage:
    """Measured accounting for the bounded explanation projection."""

    evidence_records: int
    measured_candidates: int
    near_misses: int
    next_actions: int
    output_bytes: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.evidence_records, "evidence record usage"),
            (self.measured_candidates, "measured candidate usage"),
            (self.near_misses, "near-miss usage"),
            (self.next_actions, "next-action usage"),
            (self.output_bytes, "output byte usage"),
        ):
            _nonnegative_int(value, label)

    def to_record(self) -> dict[str, int]:
        return {
            "evidence_records": self.evidence_records,
            "measured_candidates": self.measured_candidates,
            "near_misses": self.near_misses,
            "next_actions": self.next_actions,
            "output_bytes": self.output_bytes,
        }


@dataclass(frozen=True, slots=True)
class FeasibilityProvenance:
    """Immutable identity and approval context for a feasibility claim."""

    spec_identity: str
    source_model_digest: str
    archive_id: str
    approval_id: str | None = None
    approval_provenance: Mapping[str, object] = field(default_factory=dict)
    generator_revision: str = "modelsurgeon-feasibility-v1"
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.spec_identity, "spec identity")
        _digest(self.source_model_digest, "source model digest")
        _identifier(self.archive_id, "evidence archive ID")
        if self.approval_id is not None:
            _identifier(self.approval_id, "approval ID")
        _text(self.generator_revision, "generator revision")
        _sorted_unique(self.evidence_ids, "provenance evidence IDs")
        object.__setattr__(
            self,
            "approval_provenance",
            _mapping(self.approval_provenance, "approval provenance"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "spec_identity": self.spec_identity,
            "source_model_digest": self.source_model_digest,
            "archive_id": self.archive_id,
            "approval_id": self.approval_id,
            "approval_provenance": dict(self.approval_provenance),
            "generator_revision": self.generator_revision,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True, slots=True)
class ResourceObservation:
    """Optional measured resource use retained with one candidate."""

    wall_seconds: float | None = None
    memory_bytes: int | None = None
    output_bytes: int | None = None
    evaluation_count: int | None = None

    def __post_init__(self) -> None:
        if self.wall_seconds is not None:
            _nonnegative(self.wall_seconds, "candidate wall seconds")
        for value, label in (
            (self.memory_bytes, "candidate memory bytes"),
            (self.output_bytes, "candidate output bytes"),
            (self.evaluation_count, "candidate evaluation count"),
        ):
            if value is not None:
                _nonnegative_int(value, label)

    def to_record(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "memory_bytes": self.memory_bytes,
            "output_bytes": self.output_bytes,
            "evaluation_count": self.evaluation_count,
        }


@dataclass(frozen=True, slots=True)
class RetainedEvidenceReference:
    """Compact status/disposition reference for every archive cell."""

    candidate_id: str
    evidence_id: str
    status: CandidateEvidenceStatus
    disposition: CandidateDisposition

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "retained candidate ID", "candidate_")
        _identifier(self.evidence_id, "retained evidence ID", "evidence_")

    def to_record(self) -> dict[str, str]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "status": self.status.value,
            "disposition": self.disposition.value,
        }


@dataclass(frozen=True, slots=True)
class FeasibilityCandidateEvidence:
    """One canonical candidate result, including negative terminal evidence."""

    candidate_id: str
    evidence_id: str
    status: CandidateEvidenceStatus
    source_model_digest: str
    observations: tuple[MetricObservation, ...] = ()
    predicted_observations: tuple[MetricObservation, ...] = ()
    disposition: CandidateDisposition = CandidateDisposition.UNKNOWN
    evaluation_id: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)
    resource_usage: ResourceObservation | None = None

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "candidate ID", "candidate_")
        _identifier(self.evidence_id, "evidence ID", "evidence_")
        _digest(self.source_model_digest, "candidate source model digest")
        if not isinstance(self.status, CandidateEvidenceStatus):
            raise FeasibilityExplanationError("candidate evidence status is invalid")
        if not isinstance(self.disposition, CandidateDisposition):
            raise FeasibilityExplanationError("candidate disposition is invalid")
        if self.evaluation_id is not None:
            _identifier(self.evaluation_id, "evaluation ID", "evaluation_")
        for label, observations in (
            ("measured observations", self.observations),
            ("predicted observations", self.predicted_observations),
        ):
            metrics = tuple(item.metric for item in observations)
            if metrics != tuple(sorted(set(metrics))):
                raise FeasibilityExplanationError(f"{label} must be sorted and unique")
        if self.status is CandidateEvidenceStatus.MEASURED:
            if self.predicted_observations:
                raise FeasibilityExplanationError(
                    "measured evidence cannot carry predicted observations"
                )
        elif self.observations:
            raise FeasibilityExplanationError(
                "non-measured evidence cannot carry measurements"
            )
        object.__setattr__(self, "provenance", _mapping(self.provenance, "candidate provenance"))

    @property
    def measured(self) -> bool:
        return self.status is CandidateEvidenceStatus.MEASURED

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": FEASIBILITY_EXPLANATION_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "status": self.status.value,
            "source_model_digest": self.source_model_digest,
            "observations": [item.to_record() for item in self.observations],
            "predicted_observations": [item.to_record() for item in self.predicted_observations],
            "disposition": self.disposition.value,
            "evaluation_id": self.evaluation_id,
            "provenance": dict(self.provenance),
            "resource_usage": (
                None if self.resource_usage is None else self.resource_usage.to_record()
            ),
        }


@dataclass(frozen=True, slots=True)
class CanonicalEvidenceArchive:
    """Deterministic in-memory projection of candidate evidence."""

    contract_id: str
    candidates: tuple[FeasibilityCandidateEvidence, ...]
    archive_id: str

    def __post_init__(self) -> None:
        _identifier(self.contract_id, "objective contract ID")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise FeasibilityExplanationError("archive candidates must be sorted and unique")
        evidence_ids = tuple(item.evidence_id for item in self.candidates)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise FeasibilityExplanationError("archive evidence IDs must be unique")
        _identifier(self.archive_id, "evidence archive ID")

    @classmethod
    def build(
        cls,
        contract: ObjectiveContract,
        candidates: Sequence[FeasibilityCandidateEvidence],
    ) -> CanonicalEvidenceArchive:
        ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        if len(ordered) > 10_000:
            raise FeasibilityExplanationError("archive exceeds the default candidate bound")
        if any(item.source_model_digest != ordered[0].source_model_digest for item in ordered[1:]):
            raise FeasibilityExplanationError("archive candidates use different source models")
        payload = {
            "schema_version": FEASIBILITY_EXPLANATION_SCHEMA_VERSION,
            "contract_id": contract.contract_id,
            "evidence": [item.to_record() for item in ordered],
        }
        archive_id = "evidence_archive_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()
        return cls(contract.contract_id, ordered, archive_id)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": FEASIBILITY_EXPLANATION_SCHEMA_VERSION,
            "archive_id": self.archive_id,
            "contract_id": self.contract_id,
            "candidates": [item.to_record() for item in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class ConstraintGap:
    """Conservative measured distance from one hard constraint."""

    constraint: HardConstraint
    observed: float
    gap: float
    normalized_gap: float
    uncertain: bool

    def __post_init__(self) -> None:
        _finite(self.observed, "conservative observation")
        _nonnegative(self.gap, "constraint gap")
        _nonnegative(self.normalized_gap, "normalized constraint gap")

    def to_record(self) -> dict[str, object]:
        return {
            "constraint": self.constraint.to_record(),
            "observed": self.observed,
            "gap": self.gap,
            "normalized_gap": self.normalized_gap,
            "uncertain": self.uncertain,
        }


@dataclass(frozen=True, slots=True)
class UnmetHardConstraint:
    """A contract constraint violated by one or more complete measurements."""

    constraint: HardConstraint
    candidate_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    closest_observed: float
    reason: str = "threshold_violation"

    def __post_init__(self) -> None:
        _sorted_unique(self.candidate_ids, "unmet constraint candidate IDs")
        _sorted_unique(self.evidence_ids, "unmet constraint evidence IDs")
        _nonnegative(self.closest_observed, "closest observed constraint value")
        _text(self.reason, "unmet constraint reason")

    def to_record(self) -> dict[str, object]:
        return {
            "constraint": self.constraint.to_record(),
            "candidate_ids": list(self.candidate_ids),
            "evidence_ids": list(self.evidence_ids),
            "closest_observed": self.closest_observed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyRecord:
    """An explicit interval retained in a measured candidate observation."""

    candidate_id: str
    evidence_id: str
    metric: str
    value: float
    lower: float | None
    upper: float | None

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "uncertainty candidate ID", "candidate_")
        _identifier(self.evidence_id, "uncertainty evidence ID", "evidence_")
        _text(self.metric, "uncertainty metric")
        _finite(self.value, "uncertainty value")
        if self.lower is None and self.upper is None:
            raise FeasibilityExplanationError("uncertainty requires a lower or upper bound")
        if self.lower is not None and self.lower > self.value:
            raise FeasibilityExplanationError("uncertainty lower bound exceeds value")
        if self.upper is not None and self.upper < self.value:
            raise FeasibilityExplanationError("uncertainty upper bound is below value")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "metric": self.metric,
            "value": self.value,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True, slots=True)
class ClosestMeasuredCandidate:
    """A measured, complete candidate nearest to satisfying every constraint."""

    candidate_id: str
    evidence_id: str
    distance: float
    gaps: tuple[ConstraintGap, ...]
    disposition: CandidateDisposition
    uncertain_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "closest candidate ID", "candidate_")
        _identifier(self.evidence_id, "closest evidence ID", "evidence_")
        _nonnegative(self.distance, "candidate distance")
        metrics = tuple(item.constraint.metric for item in self.gaps)
        if metrics != tuple(sorted(set(metrics))):
            raise FeasibilityExplanationError("candidate gaps must be sorted and unique")
        _sorted_unique(self.uncertain_metrics, "uncertain candidate metrics")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "evidence_id": self.evidence_id,
            "distance": self.distance,
            "gaps": [item.to_record() for item in self.gaps],
            "disposition": self.disposition.value,
            "uncertain_metrics": list(self.uncertain_metrics),
        }


@dataclass(frozen=True, slots=True)
class FeasibilityNextAction:
    """A typed follow-up that cannot mutate the contract implicitly."""

    kind: NextActionKind
    reason: str
    candidate_ids: tuple[str, ...] = ()
    metric_names: tuple[str, ...] = ()
    requires_approval: bool = False

    def __post_init__(self) -> None:
        _text(self.reason, "next-action reason")
        _sorted_unique(self.candidate_ids, "next-action candidate IDs")
        _sorted_unique(self.metric_names, "next-action metric names")
        if (
            self.kind is NextActionKind.PROPOSE_EXPLICIT_OBJECTIVE_AMENDMENT
            and not self.requires_approval
        ):
            raise FeasibilityExplanationError("objective amendments require explicit approval")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "candidate_ids": list(self.candidate_ids),
            "metric_names": list(self.metric_names),
            "requires_approval": self.requires_approval,
        }


@dataclass(frozen=True, slots=True)
class FeasibilityExplanation:
    """Typed deterministic feasibility or measured-infeasibility result."""

    outcome: FeasibilityOutcome
    contract_id: str
    hard_constraints: tuple[HardConstraint, ...]
    measured_candidate_ids: tuple[str, ...]
    feasible_candidate_ids: tuple[str, ...]
    unmet_constraints: tuple[UnmetHardConstraint, ...]
    closest_candidates: tuple[ClosestMeasuredCandidate, ...]
    retained_evidence: tuple[RetainedEvidenceReference, ...]
    missing_metrics: tuple[str, ...]
    uncertainty: tuple[UncertaintyRecord, ...]
    next_actions: tuple[FeasibilityNextAction, ...]
    provenance: FeasibilityProvenance
    resource_bounds: FeasibilityResourceBounds
    resource_usage: FeasibilityResourceUsage
    reason: str
    schema_version: int = FEASIBILITY_EXPLANATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.contract_id, "objective contract ID")
        if self.schema_version != FEASIBILITY_EXPLANATION_SCHEMA_VERSION:
            raise FeasibilityExplanationError("unsupported feasibility explanation schema")
        constraint_metrics = tuple(item.metric for item in self.hard_constraints)
        if constraint_metrics != tuple(sorted(set(constraint_metrics))):
            raise FeasibilityExplanationError("hard constraints must be sorted and unique")
        _sorted_unique(self.measured_candidate_ids, "measured candidate IDs")
        _sorted_unique(self.feasible_candidate_ids, "feasible candidate IDs")
        if not set(self.feasible_candidate_ids) <= set(self.measured_candidate_ids):
            raise FeasibilityExplanationError("feasible candidates must be measured candidates")
        _sorted_unique(self.missing_metrics, "missing metrics")
        if tuple(item.constraint.metric for item in self.unmet_constraints) != tuple(
            sorted(item.constraint.metric for item in self.unmet_constraints)
        ):
            raise FeasibilityExplanationError("unmet constraints must be sorted")
        retained_ids = tuple(item.candidate_id for item in self.retained_evidence)
        if retained_ids != tuple(sorted(set(retained_ids))):
            raise FeasibilityExplanationError("retained evidence must be sorted and unique")
        if tuple((item.distance, item.candidate_id) for item in self.closest_candidates) != tuple(
            sorted((item.distance, item.candidate_id) for item in self.closest_candidates)
        ):
            raise FeasibilityExplanationError(
                "closest candidates must be deterministically ordered"
            )
        action_keys = tuple(
            (item.kind.value, item.candidate_ids, item.metric_names) for item in self.next_actions
        )
        if action_keys != tuple(sorted(action_keys)):
            raise FeasibilityExplanationError("next actions must be deterministically ordered")
        if (
            self.outcome is FeasibilityOutcome.INFEASIBLE
            and (not self.measured_candidate_ids or not self.unmet_constraints)
        ):
            raise FeasibilityExplanationError(
                "infeasibility requires measured candidates and unmet constraints"
            )
        if self.outcome is FeasibilityOutcome.FEASIBLE and not self.feasible_candidate_ids:
            raise FeasibilityExplanationError("feasible results require a measured candidate")
        _text(self.reason, "feasibility explanation reason")

    @property
    def infeasible(self) -> bool:
        return self.outcome is FeasibilityOutcome.INFEASIBLE

    def _record_without_id(self) -> dict[str, object]:
        return {
            "record_type": "measured_feasibility_explanation",
            "schema_version": self.schema_version,
            "outcome": self.outcome.value,
            "contract_id": self.contract_id,
            "hard_constraints": [item.to_record() for item in self.hard_constraints],
            "measured_candidate_ids": list(self.measured_candidate_ids),
            "feasible_candidate_ids": list(self.feasible_candidate_ids),
            "unmet_constraints": [item.to_record() for item in self.unmet_constraints],
            "closest_candidates": [item.to_record() for item in self.closest_candidates],
            "retained_evidence": [item.to_record() for item in self.retained_evidence],
            "missing_metrics": list(self.missing_metrics),
            "uncertainty": [item.to_record() for item in self.uncertainty],
            "next_actions": [item.to_record() for item in self.next_actions],
            "provenance": self.provenance.to_record(),
            "resource_bounds": self.resource_bounds.to_record(),
            "resource_usage": self.resource_usage.to_record(),
            "reason": self.reason,
        }

    @property
    def result_id(self) -> str:
        return "feasibility_result_" + hashlib.sha256(
            _canonical(self._record_without_id()).encode()
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {**self._record_without_id(), "result_id": self.result_id}

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


# Friendly aliases for callers that use the archive/result vocabulary.
FeasibilityResult = FeasibilityExplanation
FeasibilityEvidenceStatus = CandidateEvidenceStatus
MeasurementStatus = CandidateEvidenceStatus
FeasibilityEvidenceArchive = CanonicalEvidenceArchive


def _constraint_gap(
    constraint: HardConstraint,
    observation: MetricObservation,
) -> tuple[bool, float, ConstraintGap | None]:
    observed = observation.conservative(constraint.direction)
    passed = (
        observed >= constraint.threshold
        if constraint.direction is ConstraintDirection.MINIMUM
        else observed <= constraint.threshold
    )
    if passed:
        return True, observed, None
    gap = (
        constraint.threshold - observed
        if constraint.direction is ConstraintDirection.MINIMUM
        else observed - constraint.threshold
    )
    scale = max(abs(constraint.threshold), 1.0)
    return (
        False,
        observed,
        ConstraintGap(
            constraint,
            observed,
            gap,
            gap / scale,
            observation.lower is not None or observation.upper is not None,
        ),
    )


def _actions(
    outcome: FeasibilityOutcome,
    *,
    predicted_ids: tuple[str, ...],
    failed_ids: tuple[str, ...],
    inconclusive_ids: tuple[str, ...],
    unsupported_ids: tuple[str, ...],
    incomplete_ids: tuple[str, ...],
    missing_metrics: tuple[str, ...],
    near_miss_ids: tuple[str, ...],
) -> tuple[FeasibilityNextAction, ...]:
    actions: list[FeasibilityNextAction] = []
    if outcome is FeasibilityOutcome.FEASIBLE:
        actions.append(
            FeasibilityNextAction(
                NextActionKind.NO_ACTION,
                "a measured candidate satisfies every declared hard constraint",
            )
        )
    elif outcome is FeasibilityOutcome.INFEASIBLE:
        if near_miss_ids:
            actions.append(
                FeasibilityNextAction(
                    NextActionKind.REVIEW_MEASURED_NEAR_MISSES,
                    "review measured candidates without treating any near miss as acceptable",
                    candidate_ids=near_miss_ids,
                )
            )
        actions.append(
            FeasibilityNextAction(
                NextActionKind.PROPOSE_EXPLICIT_OBJECTIVE_AMENDMENT,
                "any constraint change requires a new approved objective/spec amendment",
                requires_approval=True,
            )
        )
    if incomplete_ids or outcome is FeasibilityOutcome.MISSING_EVIDENCE:
        actions.append(
            FeasibilityNextAction(
                NextActionKind.COLLECT_MEASUREMENTS,
                "collect measured evidence for the missing hard-constraint metrics",
                candidate_ids=incomplete_ids,
                metric_names=missing_metrics,
            )
        )
    if predicted_ids:
        actions.append(
            FeasibilityNextAction(
                NextActionKind.MEASURE_PREDICTED_CANDIDATES,
                "predictions nominate candidates but cannot establish feasibility",
                candidate_ids=predicted_ids,
            )
        )
    if failed_ids:
        actions.append(
            FeasibilityNextAction(
                NextActionKind.RETRY_FAILED_EVIDENCE,
                "retry failed evaluations within the declared resource budget",
                candidate_ids=failed_ids,
            )
        )
    if inconclusive_ids:
        actions.append(
            FeasibilityNextAction(
                NextActionKind.COMPLETE_INCONCLUSIVE_EVALUATION,
                "complete or replace inconclusive evidence before making a feasibility claim",
                candidate_ids=inconclusive_ids,
            )
        )
    if unsupported_ids:
        actions.append(
            FeasibilityNextAction(
                NextActionKind.RESOLVE_UNSUPPORTED_CAPABILITY,
                "resolve the unsupported capability or record a supported evaluator",
                candidate_ids=unsupported_ids,
            )
        )
    return tuple(
        sorted(actions, key=lambda item: (item.kind.value, item.candidate_ids, item.metric_names))
    )


def build_feasibility_explanation(
    contract: ObjectiveContract,
    evidence: Sequence[FeasibilityCandidateEvidence] | CanonicalEvidenceArchive,
    *,
    provenance: FeasibilityProvenance,
    resource_bounds: FeasibilityResourceBounds = _DEFAULT_RESOURCE_BOUNDS,
) -> FeasibilityExplanation:
    """Build a deterministic explanation from canonical candidate evidence.

    Missing, unsupported, failed, unknown, inconclusive, and predicted cells
    are retained.  Only complete measured cells can produce ``FEASIBLE`` or
    ``INFEASIBLE``; a hard-constraint violation is never inferred from a
    prediction or from an absent observation.
    """

    if isinstance(evidence, CanonicalEvidenceArchive):
        if evidence.contract_id != contract.contract_id:
            raise FeasibilityExplanationError("archive objective contract does not match")
        archive = evidence
    else:
        if len(evidence) > resource_bounds.max_candidates:
            raise FeasibilityExplanationError("evidence exceeds the declared candidate bound")
        archive = CanonicalEvidenceArchive.build(contract, evidence)
    candidates = archive.candidates
    if len(candidates) > resource_bounds.max_candidates:
        raise FeasibilityExplanationError("archive exceeds the declared candidate bound")
    if any(item.source_model_digest != provenance.source_model_digest for item in candidates):
        raise FeasibilityExplanationError("evidence source does not match provenance")
    evidence_ids = tuple(sorted(item.evidence_id for item in candidates))
    if provenance.archive_id != archive.archive_id:
        raise FeasibilityExplanationError("provenance archive ID does not match evidence")
    if provenance.evidence_ids and provenance.evidence_ids != evidence_ids:
        raise FeasibilityExplanationError("provenance evidence IDs do not match evidence")
    bound_provenance = provenance
    if not provenance.evidence_ids:
        bound_provenance = replace(provenance, evidence_ids=evidence_ids)

    measured = tuple(item for item in candidates if item.measured)
    measured_ids = tuple(item.candidate_id for item in measured)
    predicted_ids = tuple(
        item.candidate_id for item in candidates if item.status is CandidateEvidenceStatus.PREDICTED
    )
    failed_ids = tuple(
        item.candidate_id for item in candidates if item.status is CandidateEvidenceStatus.FAILED
    )
    inconclusive_ids = tuple(
        item.candidate_id
        for item in candidates
        if item.status is CandidateEvidenceStatus.INCONCLUSIVE
    )
    unsupported_ids = tuple(
        item.candidate_id
        for item in candidates
        if item.status is CandidateEvidenceStatus.UNSUPPORTED
    )

    complete_failures: list[tuple[FeasibilityCandidateEvidence, tuple[ConstraintGap, ...]]] = []
    feasible_ids: list[str] = []
    incomplete_ids: list[str] = []
    missing_metric_names: set[str] = set()
    uncertainty: list[UncertaintyRecord] = []
    constraint_failures: dict[str, list[tuple[str, str, float]]] = {
        constraint.metric: [] for constraint in contract.constraints
    }
    for candidate in measured:
        observations = {item.metric: item for item in candidate.observations}
        missing_for_candidate = [
            constraint.metric
            for constraint in contract.constraints
            if constraint.metric not in observations
        ]
        missing_metric_names.update(missing_for_candidate)
        for observation in candidate.observations:
            if observation.lower is not None or observation.upper is not None:
                uncertainty.append(
                    UncertaintyRecord(
                        candidate.candidate_id,
                        candidate.evidence_id,
                        observation.metric,
                        observation.value,
                        observation.lower,
                        observation.upper,
                    )
                )
        if missing_for_candidate:
            incomplete_ids.append(candidate.candidate_id)
            continue
        gaps: list[ConstraintGap] = []
        for constraint in contract.constraints:
            observation = observations[constraint.metric]
            passed, observed, gap = _constraint_gap(constraint, observation)
            if not passed:
                assert gap is not None
                gaps.append(gap)
                constraint_failures[constraint.metric].append(
                    (candidate.candidate_id, candidate.evidence_id, observed)
                )
        if not gaps:
            feasible_ids.append(candidate.candidate_id)
        else:
            complete_failures.append((candidate, tuple(gaps)))

    if feasible_ids:
        outcome = FeasibilityOutcome.FEASIBLE
        reason = "a complete measured candidate satisfies every hard constraint"
    elif complete_failures and not incomplete_ids:
        outcome = FeasibilityOutcome.INFEASIBLE
        reason = "all complete measured candidates violate at least one hard constraint"
    elif measured:
        outcome = FeasibilityOutcome.MISSING_EVIDENCE
        reason = "measured candidates exist, but at least one hard constraint is unmeasured"
    elif predicted_ids:
        outcome = FeasibilityOutcome.PREDICTED_ONLY
        reason = "only predicted candidate evidence is available; no feasibility claim is made"
    elif unsupported_ids:
        outcome = FeasibilityOutcome.UNSUPPORTED
        reason = "candidate evaluation is unsupported; infeasibility is not established"
    elif failed_ids:
        outcome = FeasibilityOutcome.FAILED
        reason = "candidate evaluation failed; infeasibility is not established"
    elif inconclusive_ids:
        outcome = FeasibilityOutcome.INCONCLUSIVE
        reason = "candidate evidence is inconclusive; infeasibility is not established"
    else:
        outcome = FeasibilityOutcome.MISSING_EVIDENCE
        reason = "no candidate evidence is available for the declared hard constraints"

    unmet = tuple(
        UnmetHardConstraint(
            constraint,
            tuple(sorted(item[0] for item in failures)),
            tuple(sorted(item[1] for item in failures)),
            (
                max(item[2] for item in failures)
                if constraint.direction is ConstraintDirection.MINIMUM
                else min(item[2] for item in failures)
            ),
        )
        for constraint in contract.constraints
        if (failures := constraint_failures[constraint.metric])
    )
    near_misses = tuple(
        sorted(
            (
                ClosestMeasuredCandidate(
                    candidate.candidate_id,
                    candidate.evidence_id,
                    math.fsum(item.normalized_gap for item in gaps),
                    gaps,
                    candidate.disposition,
                    tuple(sorted(item.constraint.metric for item in gaps if item.uncertain)),
                )
                for candidate, gaps in complete_failures
            ),
            key=lambda item: (item.distance, item.candidate_id),
        )[: resource_bounds.max_near_misses]
    )
    missing = tuple(sorted(missing_metric_names))
    if not measured:
        missing = tuple(sorted(constraint.metric for constraint in contract.constraints))
    next_actions = _actions(
        outcome,
        predicted_ids=predicted_ids,
        failed_ids=failed_ids,
        inconclusive_ids=inconclusive_ids,
        unsupported_ids=unsupported_ids,
        incomplete_ids=tuple(sorted(incomplete_ids)),
        missing_metrics=missing,
        near_miss_ids=tuple(sorted(item.candidate_id for item in near_misses)),
    )[: resource_bounds.max_next_actions]
    usage = FeasibilityResourceUsage(
        len(candidates),
        len(measured),
        len(near_misses),
        len(next_actions),
        0,
    )
    result = FeasibilityExplanation(
        outcome,
        contract.contract_id,
        contract.constraints,
        measured_ids,
        tuple(sorted(feasible_ids)),
        unmet,
        near_misses,
        tuple(
            RetainedEvidenceReference(
                item.candidate_id,
                item.evidence_id,
                item.status,
                item.disposition,
            )
            for item in candidates
        ),
        missing,
        tuple(sorted(uncertainty, key=lambda item: (item.candidate_id, item.metric)),),
        next_actions,
        bound_provenance,
        resource_bounds,
        usage,
        reason,
    )
    for _ in range(3):
        output_bytes = len(result.canonical_json().encode("utf-8"))
        if output_bytes == result.resource_usage.output_bytes:
            break
        result = replace(
            result,
            resource_usage=replace(result.resource_usage, output_bytes=output_bytes),
        )
    if result.resource_usage.output_bytes > resource_bounds.max_output_bytes:
        raise FeasibilityExplanationError("explanation exceeds the declared output bound")
    return result


def explain_feasibility(
    contract: ObjectiveContract,
    evidence: Sequence[FeasibilityCandidateEvidence] | CanonicalEvidenceArchive,
    *,
    provenance: FeasibilityProvenance,
    resource_bounds: FeasibilityResourceBounds = _DEFAULT_RESOURCE_BOUNDS,
) -> FeasibilityExplanation:
    """Alias for callers that prefer a verb-led API."""

    return build_feasibility_explanation(
        contract, evidence, provenance=provenance, resource_bounds=resource_bounds
    )


def explain_infeasibility(
    contract: ObjectiveContract,
    evidence: Sequence[FeasibilityCandidateEvidence] | CanonicalEvidenceArchive,
    *,
    provenance: FeasibilityProvenance,
    resource_bounds: FeasibilityResourceBounds = _DEFAULT_RESOURCE_BOUNDS,
) -> FeasibilityExplanation:
    """Compatibility alias for the measured-infeasibility use case."""

    return build_feasibility_explanation(
        contract, evidence, provenance=provenance, resource_bounds=resource_bounds
    )


__all__ = [
    "FEASIBILITY_EXPLANATION_SCHEMA_VERSION",
    "CandidateDisposition",
    "CandidateEvidenceStatus",
    "CanonicalEvidenceArchive",
    "ClosestMeasuredCandidate",
    "ConstraintGap",
    "FeasibilityCandidateEvidence",
    "FeasibilityEvidenceArchive",
    "FeasibilityEvidenceStatus",
    "FeasibilityExplanation",
    "FeasibilityExplanationError",
    "FeasibilityNextAction",
    "FeasibilityOutcome",
    "FeasibilityProvenance",
    "FeasibilityResourceBounds",
    "FeasibilityResourceUsage",
    "FeasibilityResult",
    "MeasurementStatus",
    "NextActionKind",
    "ResourceObservation",
    "RetainedEvidenceReference",
    "UncertaintyRecord",
    "UnmetHardConstraint",
    "build_feasibility_explanation",
    "explain_feasibility",
    "explain_infeasibility",
]

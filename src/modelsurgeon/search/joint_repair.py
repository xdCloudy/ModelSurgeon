"""Bounded joint architecture, repair, and quantization search contracts.

The policy in this module deliberately separates prediction from promotion.  A
recoverability or cost predictor may nominate a candidate for measurement, but
only held-out evidence for a published artifact can enter the measured
frontier.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final

from modelsurgeon.surgery.repair_outcome import RepairCost, RepairMethod

JOINT_REPAIR_SEARCH_SCHEMA_VERSION: Final[int] = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class JointRepairSearchError(ValueError):
    """Raised when a joint-search record would be unsafe or non-reproducible."""


class JointRepairEvidenceStatus(StrEnum):
    PREDICTED = "predicted"
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class JointRepairOutcome(StrEnum):
    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class JointRepairDecisionStatus(StrEnum):
    SELECTED = "selected"
    NO_MEASURED_ARTIFACT = "no_measured_artifact"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JointRepairSearchError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise JointRepairSearchError(f"{label} must be a lowercase SHA-256")
    return result


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JointRepairSearchError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise JointRepairSearchError(f"{label} must be finite")
    return result


def _non_negative(value: object, label: str) -> float:
    result = _finite(value, label)
    if result < 0:
        raise JointRepairSearchError(f"{label} cannot be negative")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise JointRepairSearchError(f"{label} must be a positive integer")
    return value


def _probability(value: object, label: str) -> float:
    result = _finite(value, label)
    if not 0 <= result <= 1:
        raise JointRepairSearchError(f"{label} must be within [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class JointRepairBudget:
    """Hard resources shared by one architecture/repair/quantization arm."""

    max_search_evaluations: int
    max_repair_steps: int
    max_repair_tokens: int
    max_repair_seconds: float
    max_quantization_seconds: float
    max_artifact_bytes: int

    def __post_init__(self) -> None:
        for label, value in (
            ("search evaluations", self.max_search_evaluations),
            ("repair steps", self.max_repair_steps),
            ("repair tokens", self.max_repair_tokens),
            ("artifact bytes", self.max_artifact_bytes),
        ):
            _positive_int(value, label)
        for label, numeric_value in (
            ("repair seconds", self.max_repair_seconds),
            ("quantization seconds", self.max_quantization_seconds),
        ):
            if _finite(numeric_value, label) <= 0:
                raise JointRepairSearchError(f"{label} must be positive")

    def allows(
        self,
        cost: RepairCost,
        *,
        quantization_seconds: float,
        quantization_bytes: int,
    ) -> bool:
        """Apply every action budget as a hard gate, including no-repair arms."""

        return (
            cost.steps <= self.max_repair_steps
            and cost.tokens <= self.max_repair_tokens
            and cost.training_seconds <= self.max_repair_seconds
            and quantization_seconds <= self.max_quantization_seconds
            and cost.artifact_bytes_written + quantization_bytes <= self.max_artifact_bytes
        )

    def to_record(self) -> dict[str, object]:
        return {
            "max_search_evaluations": self.max_search_evaluations,
            "max_repair_steps": self.max_repair_steps,
            "max_repair_tokens": self.max_repair_tokens,
            "max_repair_seconds": self.max_repair_seconds,
            "max_quantization_seconds": self.max_quantization_seconds,
            "max_artifact_bytes": self.max_artifact_bytes,
        }


@dataclass(frozen=True, slots=True)
class JointRepairAction:
    """One legal method, repair budget, and quantization order."""

    method: RepairMethod
    budget_name: str
    quantization_codec: str
    quantization_order: str
    compatible: bool = True
    compatibility_reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.budget_name, "joint repair budget name")
        _text(self.quantization_codec, "joint quantization codec")
        _text(self.quantization_order, "joint quantization order")
        if self.compatible and self.compatibility_reason is not None:
            raise JointRepairSearchError("compatible actions cannot carry a rejection reason")
        if not self.compatible and not self.compatibility_reason:
            raise JointRepairSearchError("incompatible actions require a compatibility reason")

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.method.value,
            self.budget_name,
            self.quantization_codec,
            self.quantization_order,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "method": self.method.value,
            "budget_name": self.budget_name,
            "quantization_codec": self.quantization_codec,
            "quantization_order": self.quantization_order,
            "compatible": self.compatible,
            "compatibility_reason": self.compatibility_reason,
        }


@dataclass(frozen=True, slots=True)
class JointRepairPrediction:
    """Conservative prediction used for acquisition, never for promotion."""

    status: JointRepairEvidenceStatus
    final_quality_low: float | None
    final_quality: float | None
    final_quality_high: float | None
    repair_benefit_low: float | None
    repair_benefit: float | None
    repair_benefit_high: float | None
    final_deployment_cost_upper: float | None
    success_probability: float | None
    predictor_ids: tuple[str, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        ids = tuple(_text(item, "predictor ID") for item in self.predictor_ids)
        if ids != tuple(sorted(set(ids))):
            raise JointRepairSearchError("predictor IDs must be unique and canonical")
        numeric = (
            self.final_quality_low,
            self.final_quality,
            self.final_quality_high,
            self.repair_benefit_low,
            self.repair_benefit,
            self.repair_benefit_high,
            self.final_deployment_cost_upper,
        )
        if self.status is JointRepairEvidenceStatus.PREDICTED:
            if any(value is None for value in numeric) or self.success_probability is None:
                raise JointRepairSearchError("predictions require complete quality and cost bounds")
            assert self.final_quality_low is not None
            assert self.final_quality is not None
            assert self.final_quality_high is not None
            assert self.repair_benefit_low is not None
            assert self.repair_benefit is not None
            assert self.repair_benefit_high is not None
            assert self.final_deployment_cost_upper is not None
            if not (
                self.final_quality_low <= self.final_quality <= self.final_quality_high
                and self.repair_benefit_low <= self.repair_benefit <= self.repair_benefit_high
            ):
                raise JointRepairSearchError("prediction intervals are not ordered")
            _non_negative(self.final_deployment_cost_upper, "predicted deployment cost")
            _probability(self.success_probability, "predicted success probability")
            if self.reason is not None:
                raise JointRepairSearchError("predictions cannot carry a terminal reason")
            if not ids:
                raise JointRepairSearchError("predictions require predictor lineage")
        else:
            if any(value is not None for value in numeric) or self.success_probability is not None:
                raise JointRepairSearchError("terminal predictions cannot carry numeric values")
            if not self.reason:
                raise JointRepairSearchError("terminal predictions require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "final_quality_low": self.final_quality_low,
            "final_quality": self.final_quality,
            "final_quality_high": self.final_quality_high,
            "repair_benefit_low": self.repair_benefit_low,
            "repair_benefit": self.repair_benefit,
            "repair_benefit_high": self.repair_benefit_high,
            "final_deployment_cost_upper": self.final_deployment_cost_upper,
            "success_probability": self.success_probability,
            "predictor_ids": list(self.predictor_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class JointRepairEvidence:
    """Measured or terminal execution evidence with complete artifact lineage."""

    candidate_id: str
    outcome: JointRepairOutcome
    final_quality: float | None
    final_deployment_cost: float | None
    repair_cost: RepairCost | None
    quantization_seconds: float | None
    quantization_bytes: int | None
    parent_artifact_digest: str | None
    source_artifact_digest: str | None
    final_artifact_digest: str | None
    heldout_group_ids: tuple[str, ...]
    rollback_performed: bool
    accepted: bool
    provenance: tuple[str, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.candidate_id, "joint evidence candidate ID")
        if self.outcome in {JointRepairOutcome.MEASURED, JointRepairOutcome.NEGATIVE_RESULT}:
            if (
                self.final_quality is None
                or self.final_deployment_cost is None
                or self.repair_cost is None
                or self.quantization_seconds is None
                or self.quantization_bytes is None
                or self.parent_artifact_digest is None
                or self.source_artifact_digest is None
                or self.final_artifact_digest is None
            ):
                raise JointRepairSearchError(
                    "measured joint evidence requires complete execution facts"
                )
            _finite(self.final_quality, "final quality")
            _non_negative(self.final_deployment_cost, "final deployment cost")
            _non_negative(self.quantization_seconds, "quantization seconds")
            _positive_int(self.quantization_bytes, "quantization bytes")
            _digest(self.parent_artifact_digest, "parent artifact digest")
            _digest(self.source_artifact_digest, "source artifact digest")
            _digest(self.final_artifact_digest, "final artifact digest")
            if not self.heldout_group_ids or self.heldout_group_ids != tuple(
                sorted(set(self.heldout_group_ids))
            ):
                raise JointRepairSearchError("measured evidence requires canonical held-out groups")
            if not self.provenance or self.provenance != tuple(sorted(set(self.provenance))):
                raise JointRepairSearchError("measured evidence requires canonical provenance")
            if self.outcome is JointRepairOutcome.MEASURED:
                if not self.accepted or self.rollback_performed or self.reason is not None:
                    raise JointRepairSearchError("accepted measured evidence cannot be rolled back")
            elif self.accepted or not self.rollback_performed or not self.reason:
                raise JointRepairSearchError(
                    "negative joint evidence must be rejected, rolled back, and explained"
                )
        else:
            values = (
                self.final_quality,
                self.final_deployment_cost,
                self.repair_cost,
                self.quantization_seconds,
                self.quantization_bytes,
                self.parent_artifact_digest,
                self.source_artifact_digest,
                self.final_artifact_digest,
            )
            if any(value is not None for value in values) or self.heldout_group_ids:
                raise JointRepairSearchError(
                    "terminal joint evidence cannot publish partial metrics"
                )
            if self.accepted or not self.reason:
                raise JointRepairSearchError("terminal joint evidence requires a reason")
            if self.provenance != tuple(sorted(set(self.provenance))):
                raise JointRepairSearchError("terminal provenance must be canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": JOINT_REPAIR_SEARCH_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "outcome": self.outcome.value,
            "final_quality": self.final_quality,
            "final_deployment_cost": self.final_deployment_cost,
            "repair_cost": None if self.repair_cost is None else self.repair_cost.to_record(),
            "quantization_seconds": self.quantization_seconds,
            "quantization_bytes": self.quantization_bytes,
            "parent_artifact_digest": self.parent_artifact_digest,
            "source_artifact_digest": self.source_artifact_digest,
            "final_artifact_digest": self.final_artifact_digest,
            "heldout_group_ids": list(self.heldout_group_ids),
            "rollback_performed": self.rollback_performed,
            "accepted": self.accepted,
            "provenance": list(self.provenance),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class JointRepairCandidate:
    candidate_id: str
    parent_state_id: str
    source_artifact_digest: str
    model_family: str
    initial_damage: float
    initial_quality: float
    initial_deployment_cost: float
    action: JointRepairAction
    budget: JointRepairBudget
    prediction: JointRepairPrediction
    evidence: JointRepairEvidence | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("joint_candidate_"):
            raise JointRepairSearchError("joint candidates require canonical IDs")
        if not self.parent_state_id.startswith("state_"):
            raise JointRepairSearchError("joint candidates require a complete parent state ID")
        _digest(self.source_artifact_digest, "candidate source artifact digest")
        _text(self.model_family, "candidate model family")
        damage = _finite(self.initial_damage, "initial damage")
        if not 0 <= damage <= 1:
            raise JointRepairSearchError("initial damage must be within [0, 1]")
        _finite(self.initial_quality, "initial quality")
        _non_negative(self.initial_deployment_cost, "initial deployment cost")
        if self.evidence is not None and self.evidence.candidate_id != self.candidate_id:
            raise JointRepairSearchError("joint evidence references a different candidate")

    @property
    def identity(self) -> str:
        payload = {
            "parent_state_id": self.parent_state_id,
            "source_artifact_digest": self.source_artifact_digest,
            "model_family": self.model_family,
            "initial_damage": self.initial_damage,
            "action": self.action.to_record(),
        }
        return "joint_candidate_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

    @property
    def measured(self) -> bool:
        return self.evidence is not None and self.evidence.outcome is JointRepairOutcome.MEASURED

    def with_evidence(self, evidence: JointRepairEvidence) -> JointRepairCandidate:
        if evidence.candidate_id != self.candidate_id:
            raise JointRepairSearchError("evidence candidate ID does not match")
        return replace(self, evidence=evidence)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": JOINT_REPAIR_SEARCH_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "identity": self.identity,
            "parent_state_id": self.parent_state_id,
            "source_artifact_digest": self.source_artifact_digest,
            "model_family": self.model_family,
            "initial_damage": self.initial_damage,
            "initial_quality": self.initial_quality,
            "initial_deployment_cost": self.initial_deployment_cost,
            "action": self.action.to_record(),
            "budget": self.budget.to_record(),
            "prediction": self.prediction.to_record(),
            "evidence": None if self.evidence is None else self.evidence.to_record(),
        }


@dataclass(frozen=True, slots=True)
class JointRepairSearchConfig:
    """Deterministic policy gates for acquisition and measured promotion."""

    max_evaluations: int = 32
    minimum_success_probability: float = 0.5
    minimum_quality_improvement: float = 0.0

    def __post_init__(self) -> None:
        _positive_int(self.max_evaluations, "joint search evaluations")
        _probability(self.minimum_success_probability, "minimum success probability")
        if _finite(self.minimum_quality_improvement, "minimum quality improvement") < 0:
            raise JointRepairSearchError("minimum quality improvement cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "max_evaluations": self.max_evaluations,
            "minimum_success_probability": self.minimum_success_probability,
            "minimum_quality_improvement": self.minimum_quality_improvement,
            "promotion_rule": (
                "predictions nominate measurements; measured held-out artifacts alone promote"
            ),
        }


@dataclass(frozen=True, slots=True)
class JointRepairDecision:
    status: JointRepairDecisionStatus
    selected: JointRepairCandidate | None
    measured_frontier: tuple[str, ...]
    acquisition_candidates: tuple[str, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is JointRepairDecisionStatus.SELECTED and self.selected is None:
            raise JointRepairSearchError("selected decisions require a candidate")
        if self.status is not JointRepairDecisionStatus.SELECTED and self.selected is not None:
            raise JointRepairSearchError("unselected decisions cannot carry a candidate")
        if self.status is not JointRepairDecisionStatus.SELECTED and not self.reason:
            raise JointRepairSearchError("unselected decisions require a reason")
        for label, values in (
            ("measured frontier", self.measured_frontier),
            ("acquisition candidates", self.acquisition_candidates),
        ):
            if values != tuple(sorted(set(values))):
                raise JointRepairSearchError(f"{label} IDs must be canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "selected": None if self.selected is None else self.selected.to_record(),
            "measured_frontier": list(self.measured_frontier),
            "acquisition_candidates": list(self.acquisition_candidates),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class JointRepairStudy:
    candidates: tuple[JointRepairCandidate, ...]
    config: JointRepairSearchConfig = JointRepairSearchConfig()
    study_id: str = ""
    schema_version: int = JOINT_REPAIR_SEARCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JOINT_REPAIR_SEARCH_SCHEMA_VERSION:
            raise JointRepairSearchError("unsupported joint-search schema version")
        if not self.candidates:
            raise JointRepairSearchError("joint-search studies require candidates")
        ids = tuple(item.candidate_id for item in self.candidates)
        if ids != tuple(sorted(set(ids))):
            raise JointRepairSearchError("joint-search candidates must be unique and canonical")
        expected = self._expected_study_id()
        if self.study_id != expected:
            raise JointRepairSearchError(
                "joint-search study identity does not match its candidates"
            )

    def _expected_study_id(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "config": self.config.to_record(),
            "candidates": [item.to_record() for item in self.candidates],
        }
        return "joint_repair_study_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "config": self.config.to_record(),
            "candidates": [item.to_record() for item in self.candidates],
        }

    def predicted_frontier(self) -> tuple[JointRepairCandidate, ...]:
        """Return conservative candidates eligible to be measured, not promoted."""

        eligible = tuple(
            item
            for item in self.candidates
            if item.action.compatible
            and item.prediction.status is JointRepairEvidenceStatus.PREDICTED
            and item.prediction.repair_benefit_low is not None
            and item.prediction.repair_benefit_low >= self.config.minimum_quality_improvement
            and item.prediction.success_probability is not None
            and item.prediction.success_probability >= self.config.minimum_success_probability
        )
        return tuple(
            sorted(
                eligible,
                key=lambda item: (
                    -(item.prediction.final_quality_low or -math.inf),
                    item.prediction.final_deployment_cost_upper or math.inf,
                    -item.initial_damage,
                    item.candidate_id,
                ),
            )
        )

    def measured_frontier(self) -> tuple[JointRepairCandidate, ...]:
        measured = tuple(
            item
            for item in self.candidates
            if item.evidence is not None
            and item.evidence.outcome is JointRepairOutcome.MEASURED
            and item.evidence.accepted
        )
        frontier: list[JointRepairCandidate] = []
        for candidate in measured:
            assert candidate.evidence is not None
            assert candidate.evidence.final_quality is not None
            assert candidate.evidence.final_deployment_cost is not None
            dominated = False
            for other in measured:
                if other is candidate or other.evidence is None:
                    continue
                assert other.evidence.final_quality is not None
                assert other.evidence.final_deployment_cost is not None
                if (
                    other.evidence.final_quality >= candidate.evidence.final_quality
                    and other.evidence.final_deployment_cost
                    <= candidate.evidence.final_deployment_cost
                    and (
                        other.evidence.final_quality > candidate.evidence.final_quality
                        or other.evidence.final_deployment_cost
                        < candidate.evidence.final_deployment_cost
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)
        return tuple(sorted(frontier, key=lambda item: item.candidate_id))

    def decide(self) -> JointRepairDecision:
        acquisition = tuple(item.candidate_id for item in self.predicted_frontier())[
            : self.config.max_evaluations
        ]
        frontier = self.measured_frontier()
        if frontier:
            def selection_key(item: JointRepairCandidate) -> tuple[float, float, float, str]:
                assert item.evidence is not None
                assert item.evidence.final_quality is not None
                assert item.evidence.final_deployment_cost is not None
                return (
                    item.evidence.final_quality,
                    -item.evidence.final_deployment_cost,
                    item.initial_damage,
                    item.candidate_id,
                )

            selected = max(frontier, key=selection_key)
            return JointRepairDecision(
                JointRepairDecisionStatus.SELECTED,
                selected,
                tuple(item.candidate_id for item in frontier),
                acquisition,
            )
        statuses = {item.prediction.status for item in self.candidates}
        if statuses and statuses <= {
            JointRepairEvidenceStatus.UNSUPPORTED,
        }:
            status = JointRepairDecisionStatus.UNSUPPORTED
        elif any(
            item.evidence and item.evidence.outcome is JointRepairOutcome.FAILED
            for item in self.candidates
        ):
            status = JointRepairDecisionStatus.FAILED
        else:
            status = JointRepairDecisionStatus.NO_MEASURED_ARTIFACT
        return JointRepairDecision(
            status,
            None,
            (),
            acquisition,
            "no held-out measured artifact is eligible for promotion",
        )


def build_joint_repair_study(
    candidates: tuple[JointRepairCandidate, ...],
    config: JointRepairSearchConfig | None = None,
) -> JointRepairStudy:
    """Build a content-addressed study from a canonical candidate matrix."""

    if config is None:
        config = JointRepairSearchConfig()
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    payload = {
        "schema_version": JOINT_REPAIR_SEARCH_SCHEMA_VERSION,
        "config": config.to_record(),
        "candidates": [item.to_record() for item in ordered],
    }
    study_id = "joint_repair_study_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return JointRepairStudy(ordered, config, study_id)


def record_joint_repair_evidence(
    candidate: JointRepairCandidate, evidence: JointRepairEvidence
) -> JointRepairCandidate:
    """Attach one immutable result, preserving failed and negative cells."""

    return candidate.with_evidence(evidence)


def joint_candidate_id(
    parent_state_id: str,
    source_artifact_digest: str,
    model_family: str,
    initial_damage: float,
    action: JointRepairAction,
) -> str:
    """Return the deterministic ID used by a joint-search candidate."""

    payload = {
        "parent_state_id": parent_state_id,
        "source_artifact_digest": source_artifact_digest,
        "model_family": model_family,
        "initial_damage": initial_damage,
        "action": action.to_record(),
    }
    return "joint_candidate_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

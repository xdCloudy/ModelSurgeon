"""Fail-closed confidence and abstention decisions for transfer inference."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from .compatibility import CompatibilityDecision, CompatibilityLevel

TRANSFER_CONFIDENCE_SCHEMA_VERSION: Final[int] = 1
TRANSFER_CONFIDENCE_PROTOCOL_REVISION: Final[str] = "transfer-confidence-v1"


class TransferConfidenceError(ValueError):
    """Raised when confidence evidence or abstention policy is invalid."""


class ConfidenceDecisionStatus(StrEnum):
    ACCEPTED = "accepted"
    ABSTAINED = "abstained"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class ConfidenceReasonCode(StrEnum):
    ACCEPTED = "accepted"
    COMPATIBILITY_UNSUPPORTED = "compatibility_unsupported"
    COMPATIBILITY_UNKNOWN = "compatibility_unknown"
    ARCHITECTURE_UNSUPPORTED = "architecture_unsupported"
    MUTATION_UNSUPPORTED = "mutation_unsupported"
    HARDWARE_UNKNOWN = "hardware_unknown"
    SUPPORT_COVERAGE = "support_coverage"
    FEATURE_DISTANCE = "feature_distance"
    UNCERTAINTY = "uncertainty"
    CALIBRATION_DRIFT = "calibration_drift"


class ConfidenceOutcomeStatus(StrEnum):
    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransferConfidenceError(f"{label} is required")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TransferConfidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TransferConfidenceError(f"{label} must be finite")
    return result


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    minimum_support_coverage: float = 0.8
    maximum_feature_distance: float = 3.0
    maximum_uncertainty: float = 0.25
    maximum_calibration_drift: float = 0.1
    target_coverage: float = 0.9
    maximum_violation_rate: float = 0.05
    bootstrap_repetitions: int = 1_000
    confidence: float = 0.95

    def __post_init__(self) -> None:
        for label, value in (
            ("minimum support coverage", self.minimum_support_coverage),
            ("target coverage", self.target_coverage),
            ("maximum violation rate", self.maximum_violation_rate),
        ):
            value = _finite(value, label)
            if not 0.0 <= value <= 1.0:
                raise TransferConfidenceError(f"{label} must be within [0, 1]")
        for label, value in (
            ("maximum feature distance", self.maximum_feature_distance),
            ("maximum uncertainty", self.maximum_uncertainty),
            ("maximum calibration drift", self.maximum_calibration_drift),
        ):
            if _finite(value, label) < 0.0:
                raise TransferConfidenceError(f"{label} must be non-negative")
        if self.bootstrap_repetitions < 100:
            raise TransferConfidenceError("confidence intervals require 100 bootstrap repetitions")
        if not 0.0 < self.confidence < 1.0:
            raise TransferConfidenceError("confidence level must be within (0, 1)")

    def to_record(self) -> dict[str, object]:
        return {
            "minimum_support_coverage": self.minimum_support_coverage,
            "maximum_feature_distance": self.maximum_feature_distance,
            "maximum_uncertainty": self.maximum_uncertainty,
            "maximum_calibration_drift": self.maximum_calibration_drift,
            "target_coverage": self.target_coverage,
            "maximum_violation_rate": self.maximum_violation_rate,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceInput:
    request_id: str
    compatibility: CompatibilityDecision
    prediction: float
    interval_low: float
    interval_high: float
    support_coverage: float
    feature_distance: float
    uncertainty: float
    calibration_drift: float
    architecture_supported: bool
    mutation_supported: bool
    hardware_known: bool

    def __post_init__(self) -> None:
        _text(self.request_id, "confidence request ID")
        if self.compatibility.target_model_id == "":
            raise TransferConfidenceError("confidence target identity is required")
        for label, value in (
            ("prediction", self.prediction),
            ("interval low", self.interval_low),
            ("interval high", self.interval_high),
            ("support coverage", self.support_coverage),
            ("feature distance", self.feature_distance),
            ("uncertainty", self.uncertainty),
            ("calibration drift", self.calibration_drift),
        ):
            _finite(value, label)
        if not 0.0 <= self.support_coverage <= 1.0:
            raise TransferConfidenceError("support coverage must be within [0, 1]")
        if self.interval_low > self.interval_high:
            raise TransferConfidenceError("confidence interval is reversed")
        if any(
            value < 0.0
            for value in (self.feature_distance, self.uncertainty, self.calibration_drift)
        ):
            raise TransferConfidenceError("confidence distances and uncertainty cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "compatibility_decision_id": self.compatibility.decision_id,
            "prediction": self.prediction,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "support_coverage": self.support_coverage,
            "feature_distance": self.feature_distance,
            "uncertainty": self.uncertainty,
            "calibration_drift": self.calibration_drift,
            "architecture_supported": self.architecture_supported,
            "mutation_supported": self.mutation_supported,
            "hardware_known": self.hardware_known,
        }


@dataclass(frozen=True, slots=True)
class ConfidenceReason:
    code: ConfidenceReasonCode
    detail: str

    def __post_init__(self) -> None:
        _text(self.detail, "confidence reason detail")

    def to_record(self) -> dict[str, str]:
        return {"code": self.code.value, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ConfidenceDecision:
    request_id: str
    status: ConfidenceDecisionStatus
    prediction: float | None
    interval_low: float | None
    interval_high: float | None
    reasons: tuple[ConfidenceReason, ...]
    decision_id: str

    def __post_init__(self) -> None:
        _text(self.request_id, "decision request ID")
        if not self.reasons or not self.decision_id:
            raise TransferConfidenceError("confidence decisions require reasons and identity")
        if self.status is ConfidenceDecisionStatus.ACCEPTED:
            if any(
                value is None for value in (self.prediction, self.interval_low, self.interval_high)
            ):
                raise TransferConfidenceError("accepted decisions require prediction evidence")
        elif any(
            value is not None for value in (self.prediction, self.interval_low, self.interval_high)
        ):
            raise TransferConfidenceError("rejected decisions cannot carry scored predictions")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": TRANSFER_CONFIDENCE_SCHEMA_VERSION,
            "protocol_revision": TRANSFER_CONFIDENCE_PROTOCOL_REVISION,
            "request_id": self.request_id,
            "status": self.status.value,
            "prediction": self.prediction,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "reasons": [reason.to_record() for reason in self.reasons],
            "decision_id": self.decision_id,
        }


def decide_transfer_confidence(
    value: ConfidenceInput, *, policy: ConfidencePolicy | None = None
) -> ConfidenceDecision:
    """Preflight hard support/compatibility before considering learned confidence."""

    resolved = policy or ConfidencePolicy()
    reasons: list[ConfidenceReason] = []
    if value.compatibility.level is CompatibilityLevel.UNKNOWN:
        reasons.append(
            ConfidenceReason(
                ConfidenceReasonCode.COMPATIBILITY_UNKNOWN,
                "compatibility evidence is unknown; confidence cannot override it",
            )
        )
        status = ConfidenceDecisionStatus.UNKNOWN
    elif (
        value.compatibility.level is CompatibilityLevel.UNSUPPORTED
        or not value.compatibility.allowed
    ):
        reasons.append(
            ConfidenceReason(
                ConfidenceReasonCode.COMPATIBILITY_UNSUPPORTED,
                "compatibility decision does not allow inference",
            )
        )
        status = ConfidenceDecisionStatus.UNSUPPORTED
    else:
        status = ConfidenceDecisionStatus.ACCEPTED
    if status in {ConfidenceDecisionStatus.ACCEPTED}:
        if not value.architecture_supported:
            reasons.append(
                ConfidenceReason(
                    ConfidenceReasonCode.ARCHITECTURE_UNSUPPORTED,
                    "architecture capability is unsupported",
                )
            )
        if not value.mutation_supported:
            reasons.append(
                ConfidenceReason(
                    ConfidenceReasonCode.MUTATION_UNSUPPORTED,
                    "mutation capability is unsupported",
                )
            )
        if not value.hardware_known:
            reasons.append(
                ConfidenceReason(
                    ConfidenceReasonCode.HARDWARE_UNKNOWN,
                    "hardware profile is unknown",
                )
            )
        if value.support_coverage < resolved.minimum_support_coverage:
            reasons.append(
                ConfidenceReason(
                    ConfidenceReasonCode.SUPPORT_COVERAGE,
                    "support coverage is below the acceptance threshold",
                )
            )
        if value.feature_distance > resolved.maximum_feature_distance:
            reasons.append(
                ConfidenceReason(
                    ConfidenceReasonCode.FEATURE_DISTANCE,
                    "feature distance exceeds the supported range",
                )
            )
        if value.uncertainty > resolved.maximum_uncertainty:
            reasons.append(
                ConfidenceReason(
                    ConfidenceReasonCode.UNCERTAINTY,
                    "uncertainty exceeds the abstention threshold",
                )
            )
        if value.calibration_drift > resolved.maximum_calibration_drift:
            reasons.append(
                ConfidenceReason(
                    ConfidenceReasonCode.CALIBRATION_DRIFT,
                    "calibration drift exceeds the abstention threshold",
                )
            )
        if not value.architecture_supported or not value.mutation_supported:
            status = ConfidenceDecisionStatus.UNSUPPORTED
        elif reasons:
            status = ConfidenceDecisionStatus.ABSTAINED
        else:
            reasons.append(
                ConfidenceReason(ConfidenceReasonCode.ACCEPTED, "all policy gates passed")
            )
    identity = {
        "input": value.to_record(),
        "policy": resolved.to_record(),
        "status": status.value,
        "reasons": [reason.to_record() for reason in reasons],
    }
    return ConfidenceDecision(
        value.request_id,
        status,
        value.prediction if status is ConfidenceDecisionStatus.ACCEPTED else None,
        value.interval_low if status is ConfidenceDecisionStatus.ACCEPTED else None,
        value.interval_high if status is ConfidenceDecisionStatus.ACCEPTED else None,
        tuple(reasons),
        "sha256:" + hashlib.sha256(_canonical(identity).encode()).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class ConfidenceObservation:
    decision: ConfidenceDecision
    actual_violation: bool | None
    target_evaluation_seconds: float
    outcome: ConfidenceOutcomeStatus
    provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _finite(self.target_evaluation_seconds, "target evaluation seconds")
        if self.target_evaluation_seconds < 0:
            raise TransferConfidenceError("target evaluation cost cannot be negative")
        if self.outcome is ConfidenceOutcomeStatus.MEASURED and self.actual_violation is None:
            raise TransferConfidenceError("measured confidence outcomes require a violation label")
        if (
            self.outcome is not ConfidenceOutcomeStatus.MEASURED
            and self.actual_violation is not None
        ):
            raise TransferConfidenceError("non-measured confidence outcomes cannot carry labels")

    def to_record(self) -> dict[str, object]:
        return {
            "decision": self.decision.to_record(),
            "actual_violation": self.actual_violation,
            "target_evaluation_seconds": self.target_evaluation_seconds,
            "outcome": self.outcome.value,
            "provenance": dict(self.provenance or {}),
        }


@dataclass(frozen=True, slots=True)
class SelectiveRiskPoint:
    uncertainty_threshold: float
    accepted_count: int
    labelled_count: int
    coverage: float | None
    violation_rate: float | None
    selective_risk: float | None
    target_evaluation_seconds: float
    confidence_low: float | None = None
    confidence_high: float | None = None

    def __post_init__(self) -> None:
        _finite(self.uncertainty_threshold, "uncertainty threshold")
        if self.accepted_count < 0 or self.labelled_count < 0:
            raise TransferConfidenceError("selective counts cannot be negative")
        if self.labelled_count > self.accepted_count:
            raise TransferConfidenceError("labelled count cannot exceed accepted count")
        _finite(self.target_evaluation_seconds, "selective target evaluation seconds")
        for label, value in (
            ("coverage", self.coverage),
            ("violation rate", self.violation_rate),
            ("selective risk", self.selective_risk),
        ):
            if value is not None:
                value = _finite(value, label)
                if not 0.0 <= value <= 1.0:
                    raise TransferConfidenceError(f"{label} must be within [0, 1]")
        if self.confidence_low is not None or self.confidence_high is not None:
            if self.confidence_low is None or self.confidence_high is None:
                raise TransferConfidenceError("selective intervals require both bounds")
            if self.confidence_low > self.confidence_high:
                raise TransferConfidenceError("selective interval is reversed")

    def to_record(self) -> dict[str, object]:
        return {
            "uncertainty_threshold": self.uncertainty_threshold,
            "accepted_count": self.accepted_count,
            "labelled_count": self.labelled_count,
            "coverage": self.coverage,
            "violation_rate": self.violation_rate,
            "selective_risk": self.selective_risk,
            "target_evaluation_seconds": self.target_evaluation_seconds,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
        }


@dataclass(frozen=True, slots=True)
class TransferConfidenceReport:
    policy: ConfidencePolicy
    observations: tuple[ConfidenceObservation, ...]
    curve: tuple[SelectiveRiskPoint, ...]
    schema_version: int = TRANSFER_CONFIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        request_ids = tuple(item.decision.request_id for item in self.observations)
        if len(request_ids) != len(set(request_ids)):
            raise TransferConfidenceError("confidence observation request IDs must be unique")
        if not self.curve:
            raise TransferConfidenceError("confidence reports require a selective-risk curve")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": TRANSFER_CONFIDENCE_PROTOCOL_REVISION,
            "policy": self.policy.to_record(),
            "observations": [item.to_record() for item in self.observations],
            "selective_risk_curve": [item.to_record() for item in self.curve],
        }


def build_selective_risk_curve(
    observations: Sequence[ConfidenceObservation],
    thresholds: Sequence[float],
    *,
    policy: ConfidencePolicy | None = None,
) -> TransferConfidenceReport:
    """Build deterministic coverage/risk/cost points from labelled decisions."""

    resolved = policy or ConfidencePolicy()
    if not observations:
        raise TransferConfidenceError("selective-risk curves require observations")
    if not thresholds or tuple(thresholds) != tuple(sorted(set(thresholds))):
        raise TransferConfidenceError("selective thresholds must be sorted and unique")
    for threshold in thresholds:
        if _finite(threshold, "selective threshold") < 0:
            raise TransferConfidenceError("selective thresholds cannot be negative")
    points: list[SelectiveRiskPoint] = []
    labelled_total = sum(item.outcome is ConfidenceOutcomeStatus.MEASURED for item in observations)
    for threshold in thresholds:
        accepted = tuple(
            item
            for item in observations
            if item.decision.status is ConfidenceDecisionStatus.ACCEPTED
            and item.decision.interval_high is not None
            and item.decision.interval_low is not None
            and item.decision.interval_high - item.decision.interval_low <= threshold
        )
        labelled = tuple(item for item in accepted if item.actual_violation is not None)
        violations = sum(bool(item.actual_violation) for item in labelled)
        coverage = len(labelled) / labelled_total if labelled_total else None
        violation_rate = violations / len(labelled) if labelled else None
        points.append(
            SelectiveRiskPoint(
                threshold,
                len(accepted),
                len(labelled),
                coverage,
                violation_rate,
                violation_rate,
                math.fsum(item.target_evaluation_seconds for item in accepted),
            )
        )
    return TransferConfidenceReport(resolved, tuple(observations), tuple(points))

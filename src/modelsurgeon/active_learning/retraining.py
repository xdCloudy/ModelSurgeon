"""Versioned retraining triggers and explicit challenger promotion decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

RETRAINING_POLICY_SCHEMA_VERSION: Final[int] = 1


class RetrainingPolicyError(ValueError):
    """Raised when retraining state or promotion evidence is invalid."""


class MetricDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True, slots=True)
class RetrainingTriggerConfig:
    example_count: int
    elapsed_seconds: float
    drift_threshold: float

    def __post_init__(self) -> None:
        if self.example_count <= 0:
            raise RetrainingPolicyError("retraining example trigger must be positive")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in (self.elapsed_seconds, self.drift_threshold)
        ):
            raise RetrainingPolicyError("retraining elapsed/drift triggers must be positive")


@dataclass(frozen=True, slots=True)
class RetrainingTriggerDecision:
    triggered: bool
    reasons: tuple[str, ...]
    new_examples: int
    elapsed_seconds: float
    drift_value: float
    schema_version: int = RETRAINING_POLICY_SCHEMA_VERSION


def evaluate_retraining_trigger(
    new_examples: int,
    elapsed_seconds: float,
    drift_value: float,
    *,
    config: RetrainingTriggerConfig,
) -> RetrainingTriggerDecision:
    if new_examples < 0 or any(
        not math.isfinite(value) or value < 0.0 for value in (elapsed_seconds, drift_value)
    ):
        raise RetrainingPolicyError("retraining trigger observations must be non-negative")
    reasons: list[str] = []
    if new_examples >= config.example_count:
        reasons.append("example-count")
    if elapsed_seconds >= config.elapsed_seconds:
        reasons.append("elapsed-budget")
    if drift_value >= config.drift_threshold:
        reasons.append("drift-signal")
    return RetrainingTriggerDecision(
        bool(reasons), tuple(reasons), new_examples, elapsed_seconds, drift_value
    )


@dataclass(frozen=True, slots=True)
class PromotionCriterion:
    metric: str
    direction: MetricDirection
    minimum_improvement: float = 0.0

    def __post_init__(self) -> None:
        if not self.metric:
            raise RetrainingPolicyError("promotion metric cannot be blank")
        if not math.isfinite(self.minimum_improvement) or self.minimum_improvement < 0.0:
            raise RetrainingPolicyError("promotion improvement must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PromotionMetricDecision:
    metric: str
    incumbent: float | None
    challenger: float | None
    improvement: float | None
    required_improvement: float
    passed: bool


@dataclass(frozen=True, slots=True)
class SurgeonPromotionDecision:
    incumbent_version: str
    challenger_version: str
    active_version: str
    promoted: bool
    reason: str
    metrics: tuple[PromotionMetricDecision, ...]
    schema_version: int = RETRAINING_POLICY_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "incumbent_version": self.incumbent_version,
            "challenger_version": self.challenger_version,
            "active_version": self.active_version,
            "promoted": self.promoted,
            "reason": self.reason,
            "metrics": [
                {
                    "metric": item.metric,
                    "incumbent": item.incumbent,
                    "challenger": item.challenger,
                    "improvement": item.improvement,
                    "required_improvement": item.required_improvement,
                    "passed": item.passed,
                }
                for item in self.metrics
            ],
        }


def evaluate_challenger_promotion(
    incumbent_version: str,
    challenger_version: str,
    incumbent_metrics: Mapping[str, float],
    challenger_metrics: Mapping[str, float],
    criteria: tuple[PromotionCriterion, ...],
    *,
    challenger_succeeded: bool,
) -> SurgeonPromotionDecision:
    """Promote only a successful challenger passing every explicit validation criterion."""

    if not incumbent_version or not challenger_version or incumbent_version == challenger_version:
        raise RetrainingPolicyError("promotion requires distinct incumbent/challenger versions")
    if not criteria or len({item.metric for item in criteria}) != len(criteria):
        raise RetrainingPolicyError("promotion criteria must be non-empty with unique metrics")
    if not challenger_succeeded:
        return SurgeonPromotionDecision(
            incumbent_version,
            challenger_version,
            incumbent_version,
            False,
            "challenger-training-failed",
            (),
        )
    decisions: list[PromotionMetricDecision] = []
    for criterion in criteria:
        incumbent = incumbent_metrics.get(criterion.metric)
        challenger = challenger_metrics.get(criterion.metric)
        if (
            incumbent is None
            or challenger is None
            or not all(math.isfinite(value) for value in (incumbent, challenger))
        ):
            decisions.append(
                PromotionMetricDecision(
                    criterion.metric,
                    incumbent,
                    challenger,
                    None,
                    criterion.minimum_improvement,
                    False,
                )
            )
            continue
        improvement = (
            challenger - incumbent
            if criterion.direction is MetricDirection.MAXIMIZE
            else incumbent - challenger
        )
        decisions.append(
            PromotionMetricDecision(
                criterion.metric,
                incumbent,
                challenger,
                improvement,
                criterion.minimum_improvement,
                improvement >= criterion.minimum_improvement,
            )
        )
    promoted = all(item.passed for item in decisions)
    return SurgeonPromotionDecision(
        incumbent_version,
        challenger_version,
        challenger_version if promoted else incumbent_version,
        promoted,
        "all-promotion-criteria-passed" if promoted else "promotion-criteria-failed",
        tuple(decisions),
    )

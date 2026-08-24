"""Configurable normalized multi-objective reward functions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from modelsurgeon.config import (
    ObjectiveConfig,
    ObjectiveDirection,
    ObjectiveNormalization,
    ObjectiveTermConfig,
    OptimizeMetric,
)

OBJECTIVE_SCHEMA_VERSION = 1


class ObjectiveError(ValueError):
    """Raised when an objective definition or observation is ambiguous."""


@dataclass(frozen=True, slots=True)
class ObjectiveTerm:
    metric: OptimizeMetric
    direction: ObjectiveDirection
    weight: float = 1.0
    normalization: ObjectiveNormalization = ObjectiveNormalization.BASELINE_RATIO
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ObjectiveError("objective weight must be finite and positive")
        bounds = self.minimum is not None or self.maximum is not None
        if self.normalization is ObjectiveNormalization.MIN_MAX:
            if (
                self.minimum is None
                or self.maximum is None
                or not math.isfinite(self.minimum)
                or not math.isfinite(self.maximum)
                or self.minimum >= self.maximum
            ):
                raise ObjectiveError("min-max normalization requires ordered finite bounds")
        elif bounds:
            raise ObjectiveError("bounds are only valid for min-max normalization")

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "direction": self.direction.value,
            "weight": self.weight,
            "normalization": self.normalization.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveObservation:
    metric: OptimizeMetric
    value: float
    baseline_value: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ObjectiveError("objective observations must be finite")
        if self.baseline_value is not None and (
            not math.isfinite(self.baseline_value) or self.baseline_value == 0
        ):
            raise ObjectiveError("objective baselines must be finite and non-zero")


@dataclass(frozen=True, slots=True)
class ObjectiveContribution:
    metric: OptimizeMetric
    normalized_value: float
    weighted_contribution: float

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "normalized_value": self.normalized_value,
            "weighted_contribution": self.weighted_contribution,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveScore:
    reward: float
    contributions: tuple[ObjectiveContribution, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "reward": self.reward,
            "contributions": [item.to_record() for item in self.contributions],
        }


@dataclass(frozen=True, slots=True)
class ObjectiveSet:
    terms: tuple[ObjectiveTerm, ...]

    def __post_init__(self) -> None:
        if not self.terms:
            raise ObjectiveError("at least one soft objective is required")
        metrics = [term.metric for term in self.terms]
        if len(metrics) != len(set(metrics)):
            raise ObjectiveError("soft objective metrics must be unique")
        object.__setattr__(
            self, "terms", tuple(sorted(self.terms, key=lambda term: term.metric.value))
        )

    @property
    def objective_set_id(self) -> str:
        payload = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))
        return f"objectives_{hashlib.sha256(payload.encode()).hexdigest()}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": OBJECTIVE_SCHEMA_VERSION,
            "terms": [term.to_record() for term in self.terms],
        }

    def score(self, observations: tuple[ObjectiveObservation, ...]) -> ObjectiveScore:
        by_metric: dict[OptimizeMetric, ObjectiveObservation] = {}
        for candidate_observation in observations:
            if candidate_observation.metric in by_metric:
                raise ObjectiveError("objective observations must be unique by metric")
            by_metric[candidate_observation.metric] = candidate_observation
        contributions: list[ObjectiveContribution] = []
        for term in self.terms:
            current = by_metric.get(term.metric)
            if current is None:
                raise ObjectiveError(f"missing observation for objective {term.metric.value}")
            if term.normalization is ObjectiveNormalization.IDENTITY:
                normalized = current.value
            elif term.normalization is ObjectiveNormalization.BASELINE_RATIO:
                if current.baseline_value is None:
                    raise ObjectiveError(f"objective {term.metric.value} requires a baseline value")
                normalized = current.value / current.baseline_value
            else:
                assert term.minimum is not None and term.maximum is not None
                normalized = (current.value - term.minimum) / (term.maximum - term.minimum)
            direction = 1.0 if term.direction is ObjectiveDirection.MAXIMIZE else -1.0
            contributions.append(
                ObjectiveContribution(term.metric, normalized, direction * term.weight * normalized)
            )
        total_weight = math.fsum(term.weight for term in self.terms)
        reward = math.fsum(item.weighted_contribution for item in contributions) / total_weight
        return ObjectiveScore(reward, tuple(contributions))


def _legacy_term(metric: OptimizeMetric) -> ObjectiveTerm:
    direction = (
        ObjectiveDirection.MAXIMIZE
        if metric is OptimizeMetric.QUALITY
        else ObjectiveDirection.MINIMIZE
    )
    return ObjectiveTerm(metric, direction)


def _from_term_config(config: ObjectiveTermConfig) -> ObjectiveTerm:
    return ObjectiveTerm(
        config.metric,
        config.direction,
        config.weight,
        config.normalization,
        config.minimum,
        config.maximum,
    )


def objectives_from_config(config: ObjectiveConfig) -> ObjectiveSet:
    """Materialize canonical soft objectives, retaining legacy optimize compatibility."""

    terms = (
        tuple(_legacy_term(metric) for metric in config.optimize)
        if config.terms is None
        else tuple(_from_term_config(term) for term in config.terms)
    )
    return ObjectiveSet(terms)

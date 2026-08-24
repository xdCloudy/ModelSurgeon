"""Composable, deterministic hard constraints for optimization search."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.config import ConstraintConfig

CONSTRAINT_SCHEMA_VERSION = 1


class ConstraintError(ValueError):
    """Raised when a hard constraint or observation is ambiguous."""


class ConstraintMetric(StrEnum):
    QUALITY_RETENTION = "quality_retention"
    PERPLEXITY_DELTA = "perplexity_delta"
    LATENCY_GAIN = "latency_gain"
    PEAK_RAM = "peak_ram"
    PEAK_VRAM = "peak_vram"
    DISK = "disk"


class BaselineReference(StrEnum):
    IMMUTABLE_SOURCE = "immutable_source"
    PARENT_CANDIDATE = "parent_candidate"
    ABSOLUTE = "absolute"


_METRIC_CONTRACT: dict[ConstraintMetric, tuple[str, str]] = {
    ConstraintMetric.QUALITY_RETENTION: ("ratio", "minimum"),
    ConstraintMetric.PERPLEXITY_DELTA: ("perplexity_points", "maximum"),
    ConstraintMetric.LATENCY_GAIN: ("ratio", "minimum"),
    ConstraintMetric.PEAK_RAM: ("bytes", "maximum"),
    ConstraintMetric.PEAK_VRAM: ("bytes", "maximum"),
    ConstraintMetric.DISK: ("bytes", "maximum"),
}
_ABSOLUTE_METRICS = frozenset(
    {ConstraintMetric.PEAK_RAM, ConstraintMetric.PEAK_VRAM, ConstraintMetric.DISK}
)


@dataclass(frozen=True, slots=True)
class OptimizationConstraint:
    metric: ConstraintMetric
    threshold: float
    baseline: BaselineReference
    schema_version: int = CONSTRAINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONSTRAINT_SCHEMA_VERSION:
            raise ConstraintError("unsupported constraint schema version")
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ConstraintError("constraint thresholds must be finite and non-negative")
        if self.metric is ConstraintMetric.QUALITY_RETENTION and self.threshold > 1:
            raise ConstraintError("quality retention must be a ratio within [0, 1]")
        requires_absolute = self.metric in _ABSOLUTE_METRICS
        if requires_absolute != (self.baseline is BaselineReference.ABSOLUTE):
            raise ConstraintError(
                "resource constraints require absolute baselines and relative metrics "
                "require a model baseline"
            )

    @property
    def unit(self) -> str:
        return _METRIC_CONTRACT[self.metric][0]

    @property
    def comparison(self) -> str:
        return _METRIC_CONTRACT[self.metric][1]

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "metric": self.metric.value,
            "comparison": self.comparison,
            "threshold": self.threshold,
            "unit": self.unit,
            "baseline": self.baseline.value,
        }


@dataclass(frozen=True, slots=True)
class ConstraintObservation:
    metric: ConstraintMetric
    value: float
    baseline: BaselineReference

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ConstraintError("constraint observations must be finite")
        if self.metric in _ABSOLUTE_METRICS and self.value < 0:
            raise ConstraintError("resource observations cannot be negative")


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    constraint: OptimizationConstraint
    observed: float | None
    passed: bool
    reason: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "constraint": self.constraint.to_record(),
            "observed": self.observed,
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ConstraintEvaluation:
    passed: bool
    results: tuple[ConstraintResult, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "results": [result.to_record() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class ConstraintSet:
    constraints: tuple[OptimizationConstraint, ...]

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ConstraintError("at least one hard constraint is required")
        metrics = [constraint.metric for constraint in self.constraints]
        if len(metrics) != len(set(metrics)):
            raise ConstraintError("hard constraint metrics must be unique")
        object.__setattr__(
            self,
            "constraints",
            tuple(sorted(self.constraints, key=lambda item: item.metric.value)),
        )

    @property
    def constraint_set_id(self) -> str:
        payload = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))
        return f"constraints_{hashlib.sha256(payload.encode()).hexdigest()}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": CONSTRAINT_SCHEMA_VERSION,
            "constraints": [constraint.to_record() for constraint in self.constraints],
        }

    def evaluate(
        self,
        observations: tuple[ConstraintObservation, ...],
    ) -> ConstraintEvaluation:
        by_metric: dict[ConstraintMetric, ConstraintObservation] = {}
        for candidate_observation in observations:
            if candidate_observation.metric in by_metric:
                raise ConstraintError("constraint observations must be unique by metric")
            by_metric[candidate_observation.metric] = candidate_observation
        results: list[ConstraintResult] = []
        for constraint in self.constraints:
            current = by_metric.get(constraint.metric)
            if current is None:
                results.append(ConstraintResult(constraint, None, False, "missing_observation"))
                continue
            if current.baseline is not constraint.baseline:
                results.append(
                    ConstraintResult(constraint, current.value, False, "baseline_mismatch")
                )
                continue
            passed = (
                current.value >= constraint.threshold
                if constraint.comparison == "minimum"
                else current.value <= constraint.threshold
            )
            results.append(
                ConstraintResult(
                    constraint,
                    current.value,
                    passed,
                    None if passed else "threshold_violation",
                )
            )
        return ConstraintEvaluation(all(result.passed for result in results), tuple(results))


def constraints_from_config(config: ConstraintConfig) -> ConstraintSet:
    """Materialize canonical hard constraints from resolved application config."""

    constraints = [
        OptimizationConstraint(
            ConstraintMetric.QUALITY_RETENTION,
            config.min_quality_retention_ratio,
            BaselineReference.IMMUTABLE_SOURCE,
        )
    ]
    optional = (
        (
            ConstraintMetric.PERPLEXITY_DELTA,
            config.max_perplexity_delta,
            BaselineReference.IMMUTABLE_SOURCE,
        ),
        (
            ConstraintMetric.LATENCY_GAIN,
            config.min_latency_gain_ratio,
            BaselineReference.IMMUTABLE_SOURCE,
        ),
        (ConstraintMetric.PEAK_RAM, config.max_ram_bytes, BaselineReference.ABSOLUTE),
        (ConstraintMetric.PEAK_VRAM, config.max_vram_bytes, BaselineReference.ABSOLUTE),
        (ConstraintMetric.DISK, config.max_disk_bytes, BaselineReference.ABSOLUTE),
    )
    constraints.extend(
        OptimizationConstraint(metric, float(value), baseline)
        for metric, value, baseline in optional
        if value is not None
    )
    return ConstraintSet(tuple(constraints))

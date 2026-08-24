"""Fixed-budget comparison of tree-surgeon uncertainty methods."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

TREE_UNCERTAINTY_SCHEMA_VERSION: Final[int] = 1


class TreeUncertaintyError(ValueError):
    """Raised when uncertainty evidence violates the bounded comparison contract."""


class TreeUncertaintyMethod(StrEnum):
    ENSEMBLE = "ensemble"
    BOOTSTRAP = "bootstrap"
    QUANTILE = "quantile"


@dataclass(frozen=True, slots=True)
class TreeUncertaintyBudget:
    max_fits_per_method: int = 8
    max_cpu_seconds_per_method: float = 3600.0
    num_threads: int = 4
    max_prediction_values: int = 1_000_000

    def __post_init__(self) -> None:
        if self.max_fits_per_method <= 0:
            raise TreeUncertaintyError("tree uncertainty fit budget must be positive")
        if (
            not math.isfinite(self.max_cpu_seconds_per_method)
            or self.max_cpu_seconds_per_method <= 0.0
        ):
            raise TreeUncertaintyError("tree uncertainty CPU budget must be finite and positive")
        if not 1 <= self.num_threads <= 32:
            raise TreeUncertaintyError("tree uncertainty threads must be within 1..32")
        if self.max_prediction_values <= 0:
            raise TreeUncertaintyError("tree uncertainty prediction budget must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "max_fits_per_method": self.max_fits_per_method,
            "max_cpu_seconds_per_method": self.max_cpu_seconds_per_method,
            "num_threads": self.num_threads,
            "max_prediction_values": self.max_prediction_values,
        }


DEFAULT_TREE_UNCERTAINTY_BUDGET: Final[TreeUncertaintyBudget] = TreeUncertaintyBudget()


@dataclass(frozen=True, slots=True)
class TreePredictionInterval:
    point: float
    lower: float
    upper: float
    uncertainty: float
    schema_version: int = TREE_UNCERTAINTY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TREE_UNCERTAINTY_SCHEMA_VERSION:
            raise TreeUncertaintyError("unsupported tree uncertainty value schema")
        if any(
            not math.isfinite(value)
            for value in (self.point, self.lower, self.upper, self.uncertainty)
        ):
            raise TreeUncertaintyError("tree uncertainty values must be finite")
        if self.lower > self.point or self.point > self.upper:
            raise TreeUncertaintyError("tree uncertainty interval must contain its point estimate")
        if self.uncertainty < 0.0:
            raise TreeUncertaintyError("tree uncertainty cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class TreeMethodEvidence:
    method: TreeUncertaintyMethod
    predictions: tuple[TreePredictionInterval, ...]
    fit_count: int
    cpu_seconds: float
    model_bytes: int
    technique_version: str

    def __post_init__(self) -> None:
        if not self.predictions:
            raise TreeUncertaintyError("tree uncertainty evidence requires predictions")
        if self.fit_count <= 0:
            raise TreeUncertaintyError("tree uncertainty evidence requires a positive fit count")
        if not math.isfinite(self.cpu_seconds) or self.cpu_seconds < 0.0:
            raise TreeUncertaintyError("tree uncertainty CPU cost must be finite and non-negative")
        if self.model_bytes < 0:
            raise TreeUncertaintyError("tree uncertainty model bytes cannot be negative")
        if not self.technique_version:
            raise TreeUncertaintyError("tree uncertainty technique version is required")


@dataclass(frozen=True, slots=True)
class TreeUncertaintyScore:
    method: TreeUncertaintyMethod
    coverage: float
    target_coverage: float
    mean_interval_width: float
    error_rank_spearman: float | None
    fit_count: int
    cpu_seconds: float
    model_bytes: int
    prediction_value_count: int
    technique_version: str
    predictions: tuple[TreePredictionInterval, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "method": self.method.value,
            "coverage": self.coverage,
            "target_coverage": self.target_coverage,
            "coverage_error": abs(self.coverage - self.target_coverage),
            "mean_interval_width": self.mean_interval_width,
            "error_rank_spearman": self.error_rank_spearman,
            "cost": {
                "fit_count": self.fit_count,
                "cpu_seconds": self.cpu_seconds,
                "model_bytes": self.model_bytes,
                "prediction_value_count": self.prediction_value_count,
            },
            "technique_version": self.technique_version,
            "predictions": [prediction.to_record() for prediction in self.predictions],
        }


@dataclass(frozen=True, slots=True)
class TreeUncertaintyStudy:
    selected_method: TreeUncertaintyMethod
    candidates: tuple[TreeUncertaintyScore, ...]
    confidence: float
    budget: TreeUncertaintyBudget
    selection_rule: str = "coverage_then_ranking_then_cpu_bytes_method"
    schema_version: int = TREE_UNCERTAINTY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TREE_UNCERTAINTY_SCHEMA_VERSION:
            raise TreeUncertaintyError("unsupported tree uncertainty study schema")
        if not any(item.method is self.selected_method for item in self.candidates):
            raise TreeUncertaintyError("selected tree uncertainty method has no evidence")

    @property
    def selected(self) -> TreeUncertaintyScore:
        return next(item for item in self.candidates if item.method is self.selected_method)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selected_method": self.selected_method.value,
            "confidence": self.confidence,
            "selection_rule": self.selection_rule,
            "budget": self.budget.to_record(),
            "candidates": [candidate.to_record() for candidate in self.candidates],
        }


def estimate_from_members(
    method: TreeUncertaintyMethod,
    member_predictions: Sequence[Sequence[float]],
    *,
    confidence: float = 0.9,
    max_prediction_values: int = 1_000_000,
) -> tuple[TreePredictionInterval, ...]:
    """Create empirical intervals from ensemble or bootstrap member predictions."""

    if method not in {TreeUncertaintyMethod.ENSEMBLE, TreeUncertaintyMethod.BOOTSTRAP}:
        raise TreeUncertaintyError("member predictions only support ensemble or bootstrap")
    _validate_confidence(confidence)
    members = tuple(tuple(row) for row in member_predictions)
    if len(members) < 2 or not members[0]:
        raise TreeUncertaintyError("member uncertainty requires at least two non-empty fits")
    width = len(members[0])
    if any(len(member) != width for member in members):
        raise TreeUncertaintyError("tree member prediction arrays must align")
    if len(members) * width > max_prediction_values:
        raise TreeUncertaintyError("tree member predictions exceed the consumer-memory budget")
    if any(not math.isfinite(value) for member in members for value in member):
        raise TreeUncertaintyError("tree member predictions must be finite")
    alpha = (1.0 - confidence) / 2.0
    intervals: list[TreePredictionInterval] = []
    for index in range(width):
        values = tuple(member[index] for member in members)
        point = math.fsum(values) / len(values)
        lower = min(point, _percentile(values, alpha))
        upper = max(point, _percentile(values, 1.0 - alpha))
        variance = math.fsum((value - point) ** 2 for value in values) / (len(values) - 1)
        intervals.append(TreePredictionInterval(point, lower, upper, math.sqrt(variance)))
    return tuple(intervals)


def estimate_from_quantiles(
    lower: Sequence[float],
    point: Sequence[float],
    upper: Sequence[float],
    *,
    max_prediction_values: int = 1_000_000,
) -> tuple[TreePredictionInterval, ...]:
    """Create intervals from separately fitted lower, central, and upper quantile trees."""

    if not lower or len(lower) != len(point) or len(point) != len(upper):
        raise TreeUncertaintyError("quantile prediction arrays must be non-empty and aligned")
    if len(lower) * 3 > max_prediction_values:
        raise TreeUncertaintyError("quantile predictions exceed the consumer-memory budget")
    intervals: list[TreePredictionInterval] = []
    for low, center, high in zip(lower, point, upper, strict=True):
        intervals.append(
            TreePredictionInterval(
                point=center,
                lower=low,
                upper=high,
                uncertainty=(high - low) / 2.0,
            )
        )
    return tuple(intervals)


def compare_tree_uncertainty(
    validation_targets: Sequence[float],
    evidence: Sequence[TreeMethodEvidence],
    *,
    confidence: float = 0.9,
    budget: TreeUncertaintyBudget = DEFAULT_TREE_UNCERTAINTY_BUDGET,
) -> TreeUncertaintyStudy:
    """Compare methods on validation coverage, ranking utility, and observed cost."""

    _validate_confidence(confidence)
    if not validation_targets or any(not math.isfinite(value) for value in validation_targets):
        raise TreeUncertaintyError(
            "tree uncertainty validation targets must be non-empty and finite"
        )
    methods = tuple(item.method for item in evidence)
    if len(evidence) != 3 or set(methods) != set(TreeUncertaintyMethod):
        raise TreeUncertaintyError("comparison requires ensemble, bootstrap, and quantile evidence")
    scores: list[TreeUncertaintyScore] = []
    for item in evidence:
        if len(item.predictions) != len(validation_targets):
            raise TreeUncertaintyError(
                "tree uncertainty predictions must align with validation targets"
            )
        if item.fit_count > budget.max_fits_per_method:
            raise TreeUncertaintyError(f"{item.method.value} exceeds the per-method fit budget")
        if item.cpu_seconds > budget.max_cpu_seconds_per_method:
            raise TreeUncertaintyError(f"{item.method.value} exceeds the per-method CPU budget")
        if len(item.predictions) > budget.max_prediction_values:
            raise TreeUncertaintyError(f"{item.method.value} exceeds the prediction-value budget")
        covered = sum(
            prediction.lower <= target <= prediction.upper
            for target, prediction in zip(validation_targets, item.predictions, strict=True)
        )
        errors = tuple(
            abs(target - prediction.point)
            for target, prediction in zip(validation_targets, item.predictions, strict=True)
        )
        uncertainties = tuple(prediction.uncertainty for prediction in item.predictions)
        scores.append(
            TreeUncertaintyScore(
                method=item.method,
                coverage=covered / len(validation_targets),
                target_coverage=confidence,
                mean_interval_width=math.fsum(
                    prediction.upper - prediction.lower for prediction in item.predictions
                )
                / len(item.predictions),
                error_rank_spearman=_spearman(uncertainties, errors),
                fit_count=item.fit_count,
                cpu_seconds=item.cpu_seconds,
                model_bytes=item.model_bytes,
                prediction_value_count=len(item.predictions),
                technique_version=item.technique_version,
                predictions=item.predictions,
            )
        )
    canonical = tuple(sorted(scores, key=lambda item: item.method.value))
    selected = min(
        canonical,
        key=lambda item: (
            abs(item.coverage - confidence),
            math.inf if item.error_rank_spearman is None else -item.error_rank_spearman,
            item.cpu_seconds,
            item.model_bytes,
            item.method.value,
        ),
    )
    return TreeUncertaintyStudy(selected.method, canonical, confidence, budget)


def _validate_confidence(confidence: float) -> None:
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise TreeUncertaintyError("tree uncertainty confidence must be within (0, 1)")


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average = (start + 1 + end) / 2.0
        for offset in range(start, end):
            ranks[ordered[offset][0]] = average
        start = end
    return tuple(ranks)


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = math.fsum(left_ranks) / len(left_ranks)
    right_mean = math.fsum(right_ranks) / len(right_ranks)
    numerator = math.fsum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks, strict=True)
    )
    left_square = math.fsum((value - left_mean) ** 2 for value in left_ranks)
    right_square = math.fsum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_square * right_square)
    return None if denominator == 0.0 else numerator / denominator

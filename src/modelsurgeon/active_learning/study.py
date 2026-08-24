"""Equal-budget active-learning curves, AULC, confidence intervals, and SVG plots."""

from __future__ import annotations

import html
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

ACTIVE_LEARNING_STUDY_SCHEMA_VERSION: Final[int] = 1


class ActiveLearningStudyError(ValueError):
    """Raised when learning curves do not form a comparable equal-budget study."""


class SelectionStrategy(StrEnum):
    ACTIVE = "active"
    RANDOM = "random"
    UTILITY_ONLY = "utility-only"


@dataclass(frozen=True, slots=True)
class LearningCurvePoint:
    experiments: int
    predictive_performance: float

    def __post_init__(self) -> None:
        if self.experiments < 0 or not math.isfinite(self.predictive_performance):
            raise ActiveLearningStudyError("learning-curve points must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class LearningCurve:
    strategy: SelectionStrategy
    seed: int
    points: tuple[LearningCurvePoint, ...]

    def __post_init__(self) -> None:
        experiments = tuple(item.experiments for item in self.points)
        if len(self.points) < 2 or experiments != tuple(sorted(set(experiments))):
            raise ActiveLearningStudyError("learning curves need two canonical unique budgets")

    @property
    def area_under_learning_curve(self) -> float:
        width = self.points[-1].experiments - self.points[0].experiments
        if width <= 0:
            raise ActiveLearningStudyError("learning-curve budget range must be positive")
        area = math.fsum(
            (right.experiments - left.experiments)
            * (left.predictive_performance + right.predictive_performance)
            / 2.0
            for left, right in zip(self.points, self.points[1:], strict=False)
        )
        return area / width

    def to_record(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.value,
            "seed": self.seed,
            "area_under_learning_curve": self.area_under_learning_curve,
            "points": [
                {
                    "experiments": item.experiments,
                    "predictive_performance": item.predictive_performance,
                }
                for item in self.points
            ],
        }


@dataclass(frozen=True, slots=True)
class StrategyStudySummary:
    strategy: SelectionStrategy
    mean_aulc: float
    confidence_low: float
    confidence_high: float
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ActiveLearningStudy:
    curves: tuple[LearningCurve, ...]
    summaries: tuple[StrategyStudySummary, ...]
    confidence: float
    bootstrap_repetitions: int
    negative_result: str | None
    schema_version: int = ACTIVE_LEARNING_STUDY_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "confidence": self.confidence,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "negative_result": self.negative_result,
            "curves": [item.to_record() for item in self.curves],
            "summaries": [
                {
                    "strategy": item.strategy.value,
                    "mean_aulc": item.mean_aulc,
                    "confidence_low": item.confidence_low,
                    "confidence_high": item.confidence_high,
                    "seeds": list(item.seeds),
                }
                for item in self.summaries
            ],
        }


def summarize_active_learning_study(
    curves: Sequence[LearningCurve],
    *,
    confidence: float = 0.95,
    bootstrap_repetitions: int = 1000,
    bootstrap_seed: int = 0,
) -> ActiveLearningStudy:
    if not 0.0 < confidence < 1.0 or bootstrap_repetitions <= 0:
        raise ActiveLearningStudyError("study confidence/bootstrap bounds are invalid")
    grouped: dict[SelectionStrategy, list[LearningCurve]] = {
        strategy: [] for strategy in SelectionStrategy
    }
    for curve in curves:
        grouped[curve.strategy].append(curve)
    if any(not items for items in grouped.values()):
        raise ActiveLearningStudyError("study requires active, random, and utility-only curves")
    grids = {tuple(point.experiments for point in curve.points) for curve in curves}
    seed_sets = {tuple(sorted(item.seed for item in values)) for values in grouped.values()}
    if len(grids) != 1 or len(seed_sets) != 1:
        raise ActiveLearningStudyError("all strategies require identical budgets and seeds")
    summaries: list[StrategyStudySummary] = []
    for strategy in SelectionStrategy:
        values = tuple(item.area_under_learning_curve for item in grouped[strategy])
        low, high = _bootstrap_mean_interval(
            values, confidence, bootstrap_repetitions, bootstrap_seed
        )
        summaries.append(
            StrategyStudySummary(
                strategy,
                math.fsum(values) / len(values),
                low,
                high,
                tuple(sorted(item.seed for item in grouped[strategy])),
            )
        )
    summary_by_strategy = {item.strategy: item for item in summaries}
    active = summary_by_strategy[SelectionStrategy.ACTIVE]
    best_baseline = max(
        summary_by_strategy[SelectionStrategy.RANDOM].mean_aulc,
        summary_by_strategy[SelectionStrategy.UTILITY_ONLY].mean_aulc,
    )
    negative = None
    if active.mean_aulc <= best_baseline:
        negative = "active selection did not exceed the best equal-budget baseline mean AULC"
    return ActiveLearningStudy(
        tuple(sorted(curves, key=lambda item: (item.strategy.value, item.seed))),
        tuple(summaries),
        confidence,
        bootstrap_repetitions,
        negative,
    )


def render_learning_curve_svg(
    study: ActiveLearningStudy, *, width: int = 800, height: int = 480
) -> str:
    """Render mean predictive performance versus experiments as dependency-free SVG."""

    if width < 300 or height < 200:
        raise ActiveLearningStudyError("learning-curve SVG dimensions are too small")
    all_points = [point for curve in study.curves for point in curve.points]
    min_x = min(item.experiments for item in all_points)
    max_x = max(item.experiments for item in all_points)
    min_y = min(item.predictive_performance for item in all_points)
    max_y = max(item.predictive_performance for item in all_points)
    y_span = max(1e-12, max_y - min_y)
    margins = (70, 30, 30, 55)
    plot_width = width - margins[0] - margins[1]
    plot_height = height - margins[2] - margins[3]
    colors = {
        SelectionStrategy.ACTIVE: "#2563eb",
        SelectionStrategy.RANDOM: "#64748b",
        SelectionStrategy.UTILITY_ONLY: "#f97316",
    }
    lines: list[str] = []
    for strategy in SelectionStrategy:
        strategy_curves = [curve for curve in study.curves if curve.strategy is strategy]
        means = []
        for index in range(len(strategy_curves[0].points)):
            values = [curve.points[index].predictive_performance for curve in strategy_curves]
            means.append(math.fsum(values) / len(values))
        coordinates = []
        for point, mean in zip(strategy_curves[0].points, means, strict=True):
            x = margins[0] + (point.experiments - min_x) / (max_x - min_x) * plot_width
            y = margins[2] + (max_y - mean) / y_span * plot_height
            coordinates.append(f"{x:.2f},{y:.2f}")
        points = " ".join(coordinates)
        lines.append(
            f'<polyline fill="none" stroke="{colors[strategy]}" '
            f'stroke-width="3" points="{points}"/>'
        )
    negative = (
        ""
        if study.negative_result is None
        else (
            f'<text x="70" y="{height - 12}" fill="#b91c1c">'
            f"{html.escape(study.negative_result)}</text>"
        )
    )
    return "".join(
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" ',
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<line x1="70" y1="30" x2="70" y2="{height - 55}" stroke="#111"/>',
            f'<line x1="70" y1="{height - 55}" x2="{width - 30}" ',
            f'y2="{height - 55}" stroke="#111"/>',
            "".join(lines),
            negative,
            "</svg>",
        )
    )


def _bootstrap_mean_interval(
    values: Sequence[float], confidence: float, repetitions: int, seed: int
) -> tuple[float, float]:
    randomizer = random.Random(seed)
    means = sorted(
        math.fsum(values[randomizer.randrange(len(values))] for _ in values) / len(values)
        for _ in range(repetitions)
    )
    alpha = (1.0 - confidence) / 2.0
    return (_percentile(means, alpha), _percentile(means, 1.0 - alpha))


def _percentile(values: Sequence[float], quantile: float) -> float:
    position = quantile * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction

"""Bounded research harness for first-order, Fisher, and diagonal-Hessian signals."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum

SENSITIVITY_SPIKE_VERSION = "1"


class SensitivitySpikeError(ValueError):
    """Raised when a sensitivity comparison probe is malformed."""


class SensitivityMethod(StrEnum):
    FIRST_ORDER = "first_order"
    EMPIRICAL_FISHER = "empirical_fisher"
    DIAGONAL_HESSIAN = "diagonal_hessian"


@dataclass(frozen=True, slots=True)
class SensitivityBudget:
    max_workspace_bytes: int = 1 << 20
    max_elapsed_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.max_workspace_bytes <= 0:
            raise SensitivitySpikeError("workspace budget must be positive")
        if not math.isfinite(self.max_elapsed_seconds) or self.max_elapsed_seconds <= 0.0:
            raise SensitivitySpikeError("time budget must be positive and finite")


@dataclass(frozen=True, slots=True)
class SensitivityProbe:
    name: str
    weights: tuple[float, ...]
    gradient_samples: tuple[tuple[float, ...], ...]
    hessian_diagonal: tuple[float, ...]
    exact_removal_deltas: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.weights:
            raise SensitivitySpikeError("probe name and weights are required")
        count = len(self.weights)
        if len(self.gradient_samples) < 2:
            raise SensitivitySpikeError("at least two gradient samples are required")
        if len(self.hessian_diagonal) != count or len(self.exact_removal_deltas) != count:
            raise SensitivitySpikeError("probe vectors must align with weights")
        if any(len(sample) != count for sample in self.gradient_samples):
            raise SensitivitySpikeError("gradient samples must align with weights")
        values = (
            *self.weights,
            *self.hessian_diagonal,
            *self.exact_removal_deltas,
            *(value for sample in self.gradient_samples for value in sample),
        )
        if any(not math.isfinite(value) for value in values):
            raise SensitivitySpikeError("probe values must be finite")
        if any(value < 0.0 for value in self.hessian_diagonal):
            raise SensitivitySpikeError("diagonal Hessian values cannot be negative")


@dataclass(frozen=True, slots=True)
class SensitivityMethodResult:
    method: SensitivityMethod
    probe_name: str
    scores: tuple[float, ...]
    predictive_spearman: float
    split_half_stability: float
    workspace_bytes: int
    operation_units: int
    elapsed_seconds: float
    feasible: bool


@dataclass(frozen=True, slots=True)
class SensitivityAggregate:
    method: SensitivityMethod
    predictive_spearman: float
    ranking_stability: float
    max_workspace_bytes: int
    total_operation_units: int
    total_elapsed_seconds: float
    feasible: bool


@dataclass(frozen=True, slots=True)
class SensitivitySpikeReport:
    version: str
    budget: SensitivityBudget
    probes: tuple[str, ...]
    results: tuple[SensitivityMethodResult, ...]
    aggregates: tuple[SensitivityAggregate, ...]
    recommendation: SensitivityMethod | None
    rationale: str

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "budget": {
                "max_workspace_bytes": self.budget.max_workspace_bytes,
                "max_elapsed_seconds": self.budget.max_elapsed_seconds,
            },
            "probes": list(self.probes),
            "aggregates": [
                {
                    "method": item.method.value,
                    "predictive_spearman": item.predictive_spearman,
                    "ranking_stability": item.ranking_stability,
                    "max_workspace_bytes": item.max_workspace_bytes,
                    "total_operation_units": item.total_operation_units,
                    "total_elapsed_seconds": item.total_elapsed_seconds,
                    "feasible": item.feasible,
                }
                for item in self.aggregates
            ],
            "recommendation": (
                None if self.recommendation is None else self.recommendation.value
            ),
            "rationale": self.rationale,
        }


def _rankdata(values: tuple[float, ...]) -> tuple[float, ...]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[indexed[offset][0]] = average_rank
        cursor = end
    return tuple(ranks)


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise SensitivitySpikeError("correlation vectors must be aligned and non-empty")
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_centered = tuple(value - left_mean for value in left)
    right_centered = tuple(value - right_mean for value in right)
    numerator = math.fsum(
        a * b for a, b in zip(left_centered, right_centered, strict=True)
    )
    left_energy = math.fsum(value * value for value in left_centered)
    right_energy = math.fsum(value * value for value in right_centered)
    if left_energy == 0.0 or right_energy == 0.0:
        return 0.0
    return numerator / math.sqrt(left_energy * right_energy)


def _spearman(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return _pearson(_rankdata(left), _rankdata(right))


def _mean_gradient(samples: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    width = len(samples[0])
    return tuple(
        math.fsum(sample[index] for sample in samples) / len(samples)
        for index in range(width)
    )


def _mean_square_gradient(samples: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    width = len(samples[0])
    return tuple(
        math.fsum(sample[index] ** 2 for sample in samples) / len(samples)
        for index in range(width)
    )


def _scores(
    method: SensitivityMethod,
    probe: SensitivityProbe,
    samples: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    if method is SensitivityMethod.FIRST_ORDER:
        gradient = _mean_gradient(samples)
        return tuple(
            abs(weight * value)
            for weight, value in zip(probe.weights, gradient, strict=True)
        )
    if method is SensitivityMethod.EMPIRICAL_FISHER:
        fisher = _mean_square_gradient(samples)
        return tuple(
            0.5 * weight * weight * value
            for weight, value in zip(probe.weights, fisher, strict=True)
        )
    return tuple(
        0.5 * weight * weight * curvature
        for weight, curvature in zip(
            probe.weights,
            probe.hessian_diagonal,
            strict=True,
        )
    )


def _cost(
    method: SensitivityMethod,
    parameter_count: int,
    sample_count: int,
) -> tuple[int, int]:
    if method is SensitivityMethod.FIRST_ORDER:
        return parameter_count * 8, parameter_count * sample_count * 2
    if method is SensitivityMethod.EMPIRICAL_FISHER:
        return parameter_count * 8, parameter_count * sample_count * 3
    return parameter_count * 16, parameter_count * sample_count * 8


def evaluate_sensitivity_probe(
    probe: SensitivityProbe,
    method: SensitivityMethod,
    budget: SensitivityBudget,
) -> SensitivityMethodResult:
    """Evaluate one method on one tiny probe under fixed resource ceilings."""

    workspace, operations = _cost(
        method,
        len(probe.weights),
        len(probe.gradient_samples),
    )
    started = time.perf_counter()
    scores = _scores(method, probe, probe.gradient_samples)
    even = probe.gradient_samples[::2]
    odd = probe.gradient_samples[1::2]
    even_scores = _scores(method, probe, even)
    odd_scores = _scores(method, probe, odd)
    predictive = _spearman(scores, probe.exact_removal_deltas)
    stability = _spearman(even_scores, odd_scores)
    if method is SensitivityMethod.DIAGONAL_HESSIAN:
        stability = 1.0
    elapsed = time.perf_counter() - started
    feasible = workspace <= budget.max_workspace_bytes and elapsed <= budget.max_elapsed_seconds
    return SensitivityMethodResult(
        method,
        probe.name,
        scores,
        predictive,
        stability,
        workspace,
        operations,
        elapsed,
        feasible,
    )


def evaluate_sensitivity_methods(
    probes: tuple[SensitivityProbe, ...],
    budget: SensitivityBudget | None = None,
    *,
    predictive_tolerance: float = 0.05,
) -> SensitivitySpikeReport:
    """Compare all three bounded methods and choose the cheapest near-best signal."""

    if len(probes) < 2:
        raise SensitivitySpikeError("the research spike requires at least two tiny probes")
    if not math.isfinite(predictive_tolerance) or predictive_tolerance < 0.0:
        raise SensitivitySpikeError("predictive tolerance must be finite and non-negative")
    resolved = budget or SensitivityBudget()
    methods = tuple(SensitivityMethod)
    results = tuple(
        evaluate_sensitivity_probe(probe, method, resolved)
        for probe in probes
        for method in methods
    )
    aggregates: list[SensitivityAggregate] = []
    for method in methods:
        selected = tuple(result for result in results if result.method is method)
        aggregates.append(
            SensitivityAggregate(
                method=method,
                predictive_spearman=math.fsum(
                    item.predictive_spearman for item in selected
                )
                / len(selected),
                ranking_stability=math.fsum(
                    item.split_half_stability for item in selected
                )
                / len(selected),
                max_workspace_bytes=max(item.workspace_bytes for item in selected),
                total_operation_units=sum(item.operation_units for item in selected),
                total_elapsed_seconds=math.fsum(
                    item.elapsed_seconds for item in selected
                ),
                feasible=all(item.feasible for item in selected),
            )
        )
    feasible = tuple(item for item in aggregates if item.feasible)
    if not feasible:
        recommendation = None
        rationale = "reject all candidates: no method satisfied the fixed resource budget"
    else:
        best_signal = max(item.predictive_spearman for item in feasible)
        near_best = tuple(
            item
            for item in feasible
            if item.predictive_spearman >= best_signal - predictive_tolerance
        )
        chosen = min(
            near_best,
            key=lambda item: (
                item.total_operation_units,
                item.max_workspace_bytes,
                -item.ranking_stability,
                item.method.value,
            ),
        )
        recommendation = chosen.method
        rationale = (
            f"recommend {chosen.method.value}: predictive signal is within "
            f"{predictive_tolerance:.3f} of the best feasible method with the lowest "
            "deterministic operation cost"
        )
    return SensitivitySpikeReport(
        SENSITIVITY_SPIKE_VERSION,
        resolved,
        tuple(probe.name for probe in probes),
        results,
        tuple(aggregates),
        recommendation,
        rationale,
    )


def default_tiny_probes() -> tuple[SensitivityProbe, SensitivityProbe]:
    """Return two deterministic quadratic probes used by the bounded research spike."""

    def probe(
        name: str,
        weights: tuple[float, ...],
        hessian: tuple[float, ...],
    ) -> SensitivityProbe:
        root = tuple(math.sqrt(value) for value in hessian)
        negative = tuple(-value for value in root)
        exact = tuple(
            0.5 * weight * weight * curvature
            for weight, curvature in zip(weights, hessian, strict=True)
        )
        return SensitivityProbe(
            name,
            weights,
            (root, negative, root, negative),
            hessian,
            exact,
        )

    return (
        probe("tiny_quadratic_a", (1.0, 2.0, 3.0, 4.0), (4.0, 3.0, 2.0, 1.0)),
        probe("tiny_quadratic_b", (2.0, 1.0, 4.0, 3.0), (1.0, 5.0, 0.5, 2.0)),
    )

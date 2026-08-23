"""Bounded research harness for attention-head redundancy metrics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import StrEnum

HEAD_SIMILARITY_SPIKE_VERSION = "1"


class HeadSimilaritySpikeError(ValueError):
    """Raised when an attention-head similarity research probe is malformed."""


class HeadSimilarityMethod(StrEnum):
    WEIGHT_COSINE = "weight_cosine"
    OUTPUT_CORRELATION = "output_correlation"
    SUBSPACE_PROJECTION = "subspace_projection"


@dataclass(frozen=True, slots=True)
class HeadSimilarityBudget:
    max_workspace_bytes: int = 1 << 20
    max_elapsed_seconds: float = 1.0

    def __post_init__(self) -> None:
        if self.max_workspace_bytes <= 0:
            raise HeadSimilaritySpikeError("head-similarity workspace budget must be positive")
        if not math.isfinite(self.max_elapsed_seconds) or self.max_elapsed_seconds <= 0.0:
            raise HeadSimilaritySpikeError("head-similarity time budget must be positive and finite")


@dataclass(frozen=True, slots=True)
class HeadPairObservation:
    label: str
    left_weight: tuple[float, ...]
    right_weight: tuple[float, ...]
    left_output: tuple[float, ...]
    right_output: tuple[float, ...]
    left_subspace: tuple[tuple[float, ...], ...]
    right_subspace: tuple[tuple[float, ...], ...]
    target_redundancy: float

    def __post_init__(self) -> None:
        if not self.label:
            raise HeadSimilaritySpikeError("head pair label is required")
        if not self.left_weight or len(self.left_weight) != len(self.right_weight):
            raise HeadSimilaritySpikeError("head weight vectors must be aligned and non-empty")
        if len(self.left_output) < 4 or len(self.left_output) != len(self.right_output):
            raise HeadSimilaritySpikeError("head output vectors require four aligned observations")
        if not self.left_subspace or not self.right_subspace:
            raise HeadSimilaritySpikeError("head subspace bases cannot be empty")
        basis_width = len(self.left_subspace[0])
        if basis_width <= 0:
            raise HeadSimilaritySpikeError("head subspace vectors cannot be empty")
        if any(len(vector) != basis_width for vector in self.left_subspace):
            raise HeadSimilaritySpikeError("left head subspace vectors must share a width")
        if any(len(vector) != basis_width for vector in self.right_subspace):
            raise HeadSimilaritySpikeError("head subspace widths must align")
        values = (
            *self.left_weight,
            *self.right_weight,
            *self.left_output,
            *self.right_output,
            *(value for vector in self.left_subspace for value in vector),
            *(value for vector in self.right_subspace for value in vector),
            self.target_redundancy,
        )
        if any(not math.isfinite(value) for value in values):
            raise HeadSimilaritySpikeError("head similarity probe values must be finite")
        if not 0.0 <= self.target_redundancy <= 1.0:
            raise HeadSimilaritySpikeError("target redundancy must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class HeadSimilarityProbe:
    model_name: str
    pairs: tuple[HeadPairObservation, ...]

    def __post_init__(self) -> None:
        if not self.model_name or len(self.pairs) < 3:
            raise HeadSimilaritySpikeError(
                "each head-similarity model probe requires a name and at least three pairs"
            )
        if len({pair.label for pair in self.pairs}) != len(self.pairs):
            raise HeadSimilaritySpikeError("head pair labels must be unique within a model")


@dataclass(frozen=True, slots=True)
class HeadSimilarityResult:
    model_name: str
    method: HeadSimilarityMethod
    scores: tuple[float, ...]
    predictive_spearman: float
    ranking_stability: float
    workspace_bytes: int
    operation_units: int
    elapsed_seconds: float
    feasible: bool


@dataclass(frozen=True, slots=True)
class HeadSimilarityAggregate:
    method: HeadSimilarityMethod
    predictive_spearman: float
    ranking_stability: float
    max_workspace_bytes: int
    total_operation_units: int
    total_elapsed_seconds: float
    feasible: bool


@dataclass(frozen=True, slots=True)
class HeadSimilaritySpikeReport:
    version: str
    models: tuple[str, ...]
    budget: HeadSimilarityBudget
    results: tuple[HeadSimilarityResult, ...]
    aggregates: tuple[HeadSimilarityAggregate, ...]
    recommendation: HeadSimilarityMethod | None
    rationale: str

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "models": list(self.models),
            "budget": {
                "max_workspace_bytes": self.budget.max_workspace_bytes,
                "max_elapsed_seconds": self.budget.max_elapsed_seconds,
            },
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
        rank = (cursor + 1 + end) / 2.0
        for offset in range(cursor, end):
            ranks[indexed[offset][0]] = rank
        cursor = end
    return tuple(ranks)


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise HeadSimilaritySpikeError("correlation vectors must be aligned and non-empty")
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    centered_left = tuple(value - left_mean for value in left)
    centered_right = tuple(value - right_mean for value in right)
    numerator = math.fsum(
        a * b for a, b in zip(centered_left, centered_right, strict=True)
    )
    left_energy = math.fsum(value * value for value in centered_left)
    right_energy = math.fsum(value * value for value in centered_right)
    if left_energy == 0.0 or right_energy == 0.0:
        return 0.0
    return max(-1.0, min(1.0, numerator / math.sqrt(left_energy * right_energy)))


def _spearman(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return _pearson(_rankdata(left), _rankdata(right))


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    left_energy = math.fsum(value * value for value in left)
    right_energy = math.fsum(value * value for value in right)
    if left_energy == 0.0 or right_energy == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / math.sqrt(left_energy * right_energy)))


def _orthonormalize(vectors: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    basis: list[tuple[float, ...]] = []
    for vector in vectors:
        residual = list(vector)
        for existing in basis:
            coefficient = math.fsum(
                a * b for a, b in zip(residual, existing, strict=True)
            )
            residual = [
                value - coefficient * direction
                for value, direction in zip(residual, existing, strict=True)
            ]
        norm = math.sqrt(math.fsum(value * value for value in residual))
        if norm <= 1e-12:
            continue
        basis.append(tuple(value / norm for value in residual))
    return tuple(basis)


def _subspace_similarity(pair: HeadPairObservation) -> float:
    left = _orthonormalize(pair.left_subspace)
    right = _orthonormalize(pair.right_subspace)
    if not left or not right:
        return 0.0
    overlap = math.fsum(
        _cosine(left_vector, right_vector) ** 2
        for left_vector in left
        for right_vector in right
    )
    return max(0.0, min(1.0, overlap / min(len(left), len(right))))


def _score(
    method: HeadSimilarityMethod,
    pair: HeadPairObservation,
    output_indices: tuple[int, ...] | None = None,
) -> float:
    if method is HeadSimilarityMethod.WEIGHT_COSINE:
        return abs(_cosine(pair.left_weight, pair.right_weight))
    if method is HeadSimilarityMethod.SUBSPACE_PROJECTION:
        return _subspace_similarity(pair)
    indices = output_indices or tuple(range(len(pair.left_output)))
    left = tuple(pair.left_output[index] for index in indices)
    right = tuple(pair.right_output[index] for index in indices)
    return abs(_pearson(left, right))


def _cost(method: HeadSimilarityMethod, probe: HeadSimilarityProbe) -> tuple[int, int]:
    max_weight = max(len(pair.left_weight) for pair in probe.pairs)
    max_output = max(len(pair.left_output) for pair in probe.pairs)
    max_basis = max(
        len(pair.left_subspace) * len(pair.left_subspace[0])
        + len(pair.right_subspace) * len(pair.right_subspace[0])
        for pair in probe.pairs
    )
    if method is HeadSimilarityMethod.WEIGHT_COSINE:
        return max_weight * 16, sum(len(pair.left_weight) * 3 for pair in probe.pairs)
    if method is HeadSimilarityMethod.OUTPUT_CORRELATION:
        return 5 * 8, sum(len(pair.left_output) * 5 for pair in probe.pairs)
    return max_basis * 24, sum(
        max_basis * max(1, len(pair.left_subspace) + len(pair.right_subspace))
        for pair in probe.pairs
    )


def evaluate_head_similarity_probe(
    probe: HeadSimilarityProbe,
    method: HeadSimilarityMethod,
    budget: HeadSimilarityBudget,
) -> HeadSimilarityResult:
    """Evaluate one redundancy metric on one tiny model probe."""

    workspace, operations = _cost(method, probe)
    started = time.perf_counter()
    scores = tuple(_score(method, pair) for pair in probe.pairs)
    targets = tuple(pair.target_redundancy for pair in probe.pairs)
    predictive = _spearman(scores, targets)
    if method is HeadSimilarityMethod.OUTPUT_CORRELATION:
        even_scores = tuple(
            _score(method, pair, tuple(range(0, len(pair.left_output), 2)))
            for pair in probe.pairs
        )
        odd_scores = tuple(
            _score(method, pair, tuple(range(1, len(pair.left_output), 2)))
            for pair in probe.pairs
        )
        stability = _spearman(even_scores, odd_scores)
    else:
        stability = 1.0
    elapsed = time.perf_counter() - started
    return HeadSimilarityResult(
        probe.model_name,
        method,
        scores,
        predictive,
        stability,
        workspace,
        operations,
        elapsed,
        workspace <= budget.max_workspace_bytes and elapsed <= budget.max_elapsed_seconds,
    )


def evaluate_head_similarity_methods(
    probes: tuple[HeadSimilarityProbe, ...],
    budget: HeadSimilarityBudget | None = None,
    *,
    predictive_tolerance: float = 0.05,
) -> HeadSimilaritySpikeReport:
    """Compare three bounded metrics and choose the cheapest stable near-best method."""

    if len(probes) < 2:
        raise HeadSimilaritySpikeError("head-similarity spike requires two model probes")
    if not math.isfinite(predictive_tolerance) or predictive_tolerance < 0.0:
        raise HeadSimilaritySpikeError("predictive tolerance must be finite and non-negative")
    resolved = budget or HeadSimilarityBudget()
    methods = tuple(HeadSimilarityMethod)
    results = tuple(
        evaluate_head_similarity_probe(probe, method, resolved)
        for probe in probes
        for method in methods
    )
    aggregates: list[HeadSimilarityAggregate] = []
    for method in methods:
        selected = tuple(item for item in results if item.method is method)
        aggregates.append(
            HeadSimilarityAggregate(
                method,
                math.fsum(item.predictive_spearman for item in selected) / len(selected),
                math.fsum(item.ranking_stability for item in selected) / len(selected),
                max(item.workspace_bytes for item in selected),
                sum(item.operation_units for item in selected),
                math.fsum(item.elapsed_seconds for item in selected),
                all(item.feasible for item in selected),
            )
        )
    feasible = tuple(item for item in aggregates if item.feasible)
    if not feasible:
        recommendation = None
        rationale = "reject all metrics: no candidate satisfied the fixed resource budget"
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
            f"{predictive_tolerance:.3f} of the best feasible metric with the lowest "
            "deterministic cost"
        )
    return HeadSimilaritySpikeReport(
        HEAD_SIMILARITY_SPIKE_VERSION,
        tuple(probe.model_name for probe in probes),
        resolved,
        results,
        tuple(aggregates),
        recommendation,
        rationale,
    )


def default_head_similarity_probes() -> tuple[HeadSimilarityProbe, HeadSimilarityProbe]:
    """Return two deterministic tiny-head fixtures with known redundancy order."""

    high_left = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    high_right = high_left
    medium_right = (1.0, 1.6, 3.2, 3.4, 5.5, 5.7, 7.4, 7.3)
    low_right = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)

    def build(name: str, scale: float) -> HeadSimilarityProbe:
        return HeadSimilarityProbe(
            name,
            (
                HeadPairObservation(
                    "redundant",
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    high_left,
                    tuple(scale * value for value in high_right),
                    ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
                    ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
                    0.95,
                ),
                HeadPairObservation(
                    "related",
                    (1.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0),
                    high_left,
                    tuple(scale * value for value in medium_right),
                    ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
                    ((1.0, 0.0, 0.0, 0.0), (0.0, 0.7, 0.7, 0.0)),
                    0.6,
                ),
                HeadPairObservation(
                    "distinct",
                    (1.0, 1.0, 0.0, 0.0),
                    (1.0, 0.0, 1.0, 0.0),
                    high_left,
                    tuple(scale * value for value in low_right),
                    ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)),
                    ((0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
                    0.1,
                ),
            ),
        )

    return build("tiny_attention_a", 1.0), build("tiny_attention_b", 1.7)

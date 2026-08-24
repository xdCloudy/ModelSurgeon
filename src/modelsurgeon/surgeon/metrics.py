"""Surgeon regression, ranking, classification, and grouped-bootstrap metrics."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

METRICS_SCHEMA_VERSION: Final[int] = 1


class SurgeonMetricError(ValueError):
    """Raised when metric inputs are malformed rather than mathematically undefined."""


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    name: str
    value: float | None
    reason: str | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    bootstrap_repetitions: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise SurgeonMetricError("metric estimates require names")
        if self.value is None:
            if not self.reason:
                raise SurgeonMetricError("undefined metrics require an explicit reason")
            if self.confidence_low is not None or self.confidence_high is not None:
                raise SurgeonMetricError("undefined metrics cannot carry confidence intervals")
        else:
            if not math.isfinite(self.value):
                raise SurgeonMetricError("defined metrics must be finite")
            if self.reason is not None:
                raise SurgeonMetricError("defined metrics cannot carry an undefined reason")
        if (self.confidence_low is None) != (self.confidence_high is None):
            raise SurgeonMetricError("confidence interval bounds must be both present or absent")
        if self.confidence_low is not None:
            assert self.confidence_high is not None
            if (
                not math.isfinite(self.confidence_low)
                or not math.isfinite(self.confidence_high)
                or self.confidence_low > self.confidence_high
            ):
                raise SurgeonMetricError("confidence interval bounds are invalid")
        if self.bootstrap_repetitions < 0:
            raise SurgeonMetricError("bootstrap repetition count cannot be negative")

    @property
    def defined(self) -> bool:
        return self.value is not None

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "defined": self.defined,
            "reason": self.reason,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "bootstrap_repetitions": self.bootstrap_repetitions,
        }


def _validate_equal(*values: Sequence[object]) -> int:
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise SurgeonMetricError("metric input arrays must have equal length")
    count = next(iter(lengths), 0)
    if count == 0:
        raise SurgeonMetricError("metrics require at least one observation")
    return count


def _finite(values: Sequence[float], label: str) -> None:
    if any(not math.isfinite(value) for value in values):
        raise SurgeonMetricError(f"{label} values must be finite")


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    count = _validate_equal(actual, predicted)
    _finite(actual, "actual")
    _finite(predicted, "predicted")
    return (
        math.fsum(
            abs(left - right)
            for left, right in zip(actual, predicted, strict=True)
        )
        / count
    )


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    count = _validate_equal(actual, predicted)
    _finite(actual, "actual")
    _finite(predicted, "predicted")
    return math.sqrt(
        math.fsum((left - right) ** 2 for left, right in zip(actual, predicted, strict=True))
        / count
    )


def r_squared(actual: Sequence[float], predicted: Sequence[float]) -> float | None:
    _validate_equal(actual, predicted)
    _finite(actual, "actual")
    _finite(predicted, "predicted")
    mean = math.fsum(actual) / len(actual)
    total = math.fsum((value - mean) ** 2 for value in actual)
    if total == 0.0:
        return None
    residual = math.fsum(
        (left - right) ** 2 for left, right in zip(actual, predicted, strict=True)
    )
    return 1.0 - residual / total


def roc_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    """Compute tie-aware binary ROC AUC from pairwise order statistics."""

    _validate_equal(labels, probabilities)
    _finite(probabilities, "probability")
    if any(label not in (0, 1) for label in labels):
        raise SurgeonMetricError("classification labels must be 0/1")
    positives = [score for label, score in zip(labels, probabilities, strict=True) if label]
    negatives = [score for label, score in zip(labels, probabilities, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def pr_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    """Compute average precision, an interpolation-free PR-AUC summary."""

    _validate_equal(labels, probabilities)
    _finite(probabilities, "probability")
    if any(label not in (0, 1) for label in labels):
        raise SurgeonMetricError("classification labels must be 0/1")
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(
        zip(probabilities, labels, strict=True),
        key=lambda item: (-item[0], -item[1]),
    )
    true_positives = 0
    precisions = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label:
            true_positives += 1
            precisions += true_positives / rank
    return precisions / positives


def calibration_error(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    _validate_equal(labels, probabilities)
    if bins <= 0:
        raise SurgeonMetricError("calibration bins must be positive")
    if any(label not in (0, 1) for label in labels):
        raise SurgeonMetricError("classification labels must be 0/1")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise SurgeonMetricError("classification probabilities must be finite and within [0, 1]")
    total = len(labels)
    bucket_labels: list[list[int]] = [[] for _ in range(bins)]
    bucket_probabilities: list[list[float]] = [[] for _ in range(bins)]
    for label, probability in zip(labels, probabilities, strict=True):
        index = min(bins - 1, int(probability * bins))
        bucket_labels[index].append(label)
        bucket_probabilities[index].append(probability)
    error = 0.0
    for observed, predicted in zip(bucket_labels, bucket_probabilities, strict=True):
        if not observed:
            continue
        accuracy = math.fsum(observed) / len(observed)
        confidence = math.fsum(predicted) / len(predicted)
        error += len(observed) / total * abs(accuracy - confidence)
    return error


def precision_recall_at_n(
    labels: Sequence[int],
    scores: Sequence[float],
    n: int,
) -> tuple[float | None, float | None]:
    _validate_equal(labels, scores)
    if n <= 0:
        raise SurgeonMetricError("top-N must be positive")
    if any(label not in (0, 1) for label in labels):
        raise SurgeonMetricError("ranking labels must be 0/1")
    selected = sorted(
        enumerate(scores),
        key=lambda item: (-item[1], item[0]),
    )[: min(n, len(labels))]
    true_positives = sum(labels[index] for index, _ in selected)
    precision = true_positives / len(selected) if selected else None
    total_positives = sum(labels)
    recall = true_positives / total_positives if total_positives else None
    return precision, recall


def constraint_violation_rate(violations: Sequence[bool]) -> float:
    if not violations:
        raise SurgeonMetricError("constraint violation metrics require observations")
    return math.fsum(1.0 for value in violations if value) / len(violations)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise SurgeonMetricError("cannot take percentile of empty values")
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def grouped_bootstrap_interval(
    group_ids: Sequence[str],
    metric: Callable[[Sequence[int]], float | None],
    *,
    repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, int] | None:
    """Resample whole leakage groups with replacement and return percentile CI."""

    if not group_ids:
        raise SurgeonMetricError("grouped bootstrap requires observations")
    if any(not group for group in group_ids):
        raise SurgeonMetricError("group IDs cannot be blank")
    if repetitions <= 0:
        raise SurgeonMetricError("bootstrap repetitions must be positive")
    if not 0.0 < confidence < 1.0:
        raise SurgeonMetricError("bootstrap confidence must be within (0, 1)")
    grouped: defaultdict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        grouped[group].append(index)
    groups = tuple(sorted(grouped))
    randomizer = random.Random(seed)
    estimates: list[float] = []
    for _ in range(repetitions):
        sample: list[int] = []
        for _ in groups:
            chosen = randomizer.choice(groups)
            sample.extend(grouped[chosen])
        value = metric(sample)
        if value is not None and math.isfinite(value):
            estimates.append(value)
    if not estimates:
        return None
    tail = (1.0 - confidence) / 2.0
    return (
        _percentile(estimates, tail),
        _percentile(estimates, 1.0 - tail),
        len(estimates),
    )


def _estimate(
    name: str,
    value: float | None,
    reason: str | None,
    group_ids: Sequence[str] | None,
    bootstrap_metric: Callable[[Sequence[int]], float | None] | None,
    *,
    repetitions: int,
    confidence: float,
    seed: int,
) -> MetricEstimate:
    if value is None:
        return MetricEstimate(name, None, reason)
    if group_ids is None or bootstrap_metric is None:
        return MetricEstimate(name, value)
    interval = grouped_bootstrap_interval(
        group_ids,
        bootstrap_metric,
        repetitions=repetitions,
        confidence=confidence,
        seed=seed,
    )
    if interval is None:
        return MetricEstimate(name, value)
    return MetricEstimate(
        name,
        value,
        None,
        interval[0],
        interval[1],
        interval[2],
    )


@dataclass(frozen=True, slots=True)
class MetricReport:
    metrics: tuple[MetricEstimate, ...]
    schema_version: int = METRICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != METRICS_SCHEMA_VERSION:
            raise SurgeonMetricError("unsupported metric report schema version")
        names = tuple(metric.name for metric in self.metrics)
        if names != tuple(sorted(set(names))):
            raise SurgeonMetricError("metric report entries must be unique and canonical")

    def metric(self, name: str) -> MetricEstimate:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "metrics": [metric.to_record() for metric in self.metrics],
        }


def evaluate_regression(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    group_ids: Sequence[str] | None = None,
    bootstrap_repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> MetricReport:
    """Evaluate continuous surgeon predictions with optional grouped-bootstrap CIs."""

    _validate_equal(actual, predicted)
    if group_ids is not None:
        _validate_equal(actual, group_ids)
    actual_tuple = tuple(actual)
    predicted_tuple = tuple(predicted)

    def subset(
        function: Callable[[Sequence[float], Sequence[float]], float | None],
    ) -> Callable[[Sequence[int]], float | None]:
        def apply(indexes: Sequence[int]) -> float | None:
            return function(
                tuple(actual_tuple[index] for index in indexes),
                tuple(predicted_tuple[index] for index in indexes),
            )
        return apply

    r2 = r_squared(actual_tuple, predicted_tuple)
    estimates = (
        _estimate(
            "mae",
            mae(actual_tuple, predicted_tuple),
            None,
            group_ids,
            subset(mae),
            repetitions=bootstrap_repetitions,
            confidence=confidence,
            seed=seed,
        ),
        _estimate(
            "r2",
            r2,
            "R² is undefined when the actual target has zero variance" if r2 is None else None,
            group_ids,
            subset(r_squared),
            repetitions=bootstrap_repetitions,
            confidence=confidence,
            seed=seed + 1,
        ),
        _estimate(
            "rmse",
            rmse(actual_tuple, predicted_tuple),
            None,
            group_ids,
            subset(rmse),
            repetitions=bootstrap_repetitions,
            confidence=confidence,
            seed=seed + 2,
        ),
    )
    return MetricReport(tuple(sorted(estimates, key=lambda item: item.name)))


def evaluate_classification(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    top_n: int,
    group_ids: Sequence[str] | None = None,
    calibration_bins: int = 10,
    bootstrap_repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> MetricReport:
    """Evaluate safe-mutation probabilities and ranking utility."""

    _validate_equal(labels, probabilities)
    if group_ids is not None:
        _validate_equal(labels, group_ids)
    labels_tuple = tuple(labels)
    probabilities_tuple = tuple(probabilities)
    auc = roc_auc(labels_tuple, probabilities_tuple)
    average_precision = pr_auc(labels_tuple, probabilities_tuple)
    precision, recall = precision_recall_at_n(labels_tuple, probabilities_tuple, top_n)

    def by_indexes(
        function: Callable[[Sequence[int], Sequence[float]], float | None]
    ) -> Callable[[Sequence[int]], float | None]:
        def apply(indexes: Sequence[int]) -> float | None:
            return function(
                tuple(labels_tuple[index] for index in indexes),
                tuple(probabilities_tuple[index] for index in indexes),
            )
        return apply

    def precision_bootstrap(indexes: Sequence[int]) -> float | None:
        subset_labels = tuple(labels_tuple[index] for index in indexes)
        subset_scores = tuple(probabilities_tuple[index] for index in indexes)
        value, _ = precision_recall_at_n(subset_labels, subset_scores, top_n)
        return value

    def recall_bootstrap(indexes: Sequence[int]) -> float | None:
        subset_labels = tuple(labels_tuple[index] for index in indexes)
        subset_scores = tuple(probabilities_tuple[index] for index in indexes)
        _, value = precision_recall_at_n(subset_labels, subset_scores, top_n)
        return value

    def calibration_bootstrap(indexes: Sequence[int]) -> float:
        return calibration_error(
            tuple(labels_tuple[index] for index in indexes),
            tuple(probabilities_tuple[index] for index in indexes),
            bins=calibration_bins,
        )

    estimates = (
        _estimate(
            "auc",
            auc,
            "AUC is undefined when only one class is present" if auc is None else None,
            group_ids,
            by_indexes(roc_auc),
            repetitions=bootstrap_repetitions,
            confidence=confidence,
            seed=seed,
        ),
        _estimate(
            "calibration_error",
            calibration_error(labels_tuple, probabilities_tuple, bins=calibration_bins),
            None,
            group_ids,
            calibration_bootstrap,
            repetitions=bootstrap_repetitions,
            confidence=confidence,
            seed=seed + 1,
        ),
        _estimate(
            f"precision_at_{top_n}",
            precision,
            "precision@N is undefined for an empty selected set" if precision is None else None,
            group_ids,
            precision_bootstrap,
            repetitions=bootstrap_repetitions,
            confidence=confidence,
            seed=seed + 2,
        ),
        _estimate(
            "pr_auc",
            average_precision,
            "PR-AUC is undefined when no positive class is present"
            if average_precision is None
            else None,
            group_ids,
            by_indexes(pr_auc),
            repetitions=bootstrap_repetitions,
            confidence=confidence,
            seed=seed + 3,
        ),
        _estimate(
            f"recall_at_{top_n}",
            recall,
            "recall@N is undefined when no positive class is present" if recall is None else None,
            group_ids,
            recall_bootstrap,
            repetitions=bootstrap_repetitions,
            confidence=confidence,
            seed=seed + 4,
        ),
    )
    return MetricReport(tuple(sorted(estimates, key=lambda item: item.name)))


def evaluate_constraint_violations(
    violations: Sequence[bool],
    *,
    group_ids: Sequence[str] | None = None,
    bootstrap_repetitions: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> MetricEstimate:
    """Evaluate constraint violation rate with grouped-bootstrap uncertainty."""

    value = constraint_violation_rate(violations)
    if group_ids is None:
        return MetricEstimate("constraint_violation_rate", value)
    _validate_equal(violations, group_ids)

    def bootstrap(indexes: Sequence[int]) -> float:
        return constraint_violation_rate(tuple(violations[index] for index in indexes))

    return _estimate(
        "constraint_violation_rate",
        value,
        None,
        group_ids,
        bootstrap,
        repetitions=bootstrap_repetitions,
        confidence=confidence,
        seed=seed,
    )

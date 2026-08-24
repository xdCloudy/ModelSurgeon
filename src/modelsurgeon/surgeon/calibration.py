"""Validation-only calibration for safe-mutation probabilities."""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, Self

CALIBRATION_SCHEMA_VERSION: Final[int] = 1
_LOGIT_EPSILON: Final[float] = 1e-12


class ProbabilityCalibrationError(ValueError):
    """Raised when calibration inputs or serialized state are invalid."""


class CalibrationMethod(StrEnum):
    PLATT = "platt"
    ISOTONIC = "isotonic"


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_probability: float | None
    observed_frequency: float | None

    def to_record(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_probability": self.mean_probability,
            "observed_frequency": self.observed_frequency,
        }


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    brier_score: float
    expected_calibration_error: float
    reliability_curve: tuple[ReliabilityBin, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "reliability_curve": [item.to_record() for item in self.reliability_curve],
        }


class ProbabilityCalibrator(Protocol):
    method: CalibrationMethod

    def calibrate(self, probabilities: Sequence[float]) -> tuple[float, ...]: ...

    def to_record(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    slope: float
    intercept: float
    method: CalibrationMethod = CalibrationMethod.PLATT

    def __post_init__(self) -> None:
        if not math.isfinite(self.slope) or not math.isfinite(self.intercept):
            raise ProbabilityCalibrationError("Platt parameters must be finite")

    def calibrate(self, probabilities: Sequence[float]) -> tuple[float, ...]:
        _validate_probabilities(probabilities)
        return tuple(
            _sigmoid(self.slope * _logit(value) + self.intercept) for value in probabilities
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "method": self.method.value,
            "slope": self.slope,
            "intercept": self.intercept,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> Self:
        _validate_record_header(record, CalibrationMethod.PLATT)
        return cls(
            slope=_record_float(record, "slope"), intercept=_record_float(record, "intercept")
        )


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    thresholds: tuple[float, ...]
    values: tuple[float, ...]
    method: CalibrationMethod = CalibrationMethod.ISOTONIC

    def __post_init__(self) -> None:
        if not self.thresholds or len(self.thresholds) != len(self.values):
            raise ProbabilityCalibrationError(
                "isotonic thresholds and values must be non-empty and aligned"
            )
        _validate_probabilities(self.thresholds)
        _validate_probabilities(self.values)
        if any(
            left >= right for left, right in zip(self.thresholds, self.thresholds[1:], strict=False)
        ):
            raise ProbabilityCalibrationError("isotonic thresholds must be strictly increasing")
        if any(left > right for left, right in zip(self.values, self.values[1:], strict=False)):
            raise ProbabilityCalibrationError("isotonic values must be non-decreasing")

    def calibrate(self, probabilities: Sequence[float]) -> tuple[float, ...]:
        _validate_probabilities(probabilities)
        calibrated: list[float] = []
        for probability in probabilities:
            index = bisect.bisect_left(self.thresholds, probability)
            if index == 0:
                calibrated.append(self.values[0])
            elif index == len(self.thresholds):
                calibrated.append(self.values[-1])
            else:
                lower_x = self.thresholds[index - 1]
                upper_x = self.thresholds[index]
                fraction = (probability - lower_x) / (upper_x - lower_x)
                lower_y = self.values[index - 1]
                calibrated.append(lower_y + fraction * (self.values[index] - lower_y))
        return tuple(calibrated)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "method": self.method.value,
            "thresholds": list(self.thresholds),
            "values": list(self.values),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> Self:
        _validate_record_header(record, CalibrationMethod.ISOTONIC)
        return cls(
            thresholds=_record_float_tuple(record, "thresholds"),
            values=_record_float_tuple(record, "values"),
        )


@dataclass(frozen=True, slots=True)
class CalibrationCandidate:
    method: CalibrationMethod
    metrics: CalibrationMetrics

    def to_record(self) -> dict[str, object]:
        return {"method": self.method.value, "metrics": self.metrics.to_record()}


@dataclass(frozen=True, slots=True)
class CalibrationSelection:
    calibrator: PlattCalibrator | IsotonicCalibrator
    candidates: tuple[CalibrationCandidate, ...]
    observation_count: int
    reliability_bins: int
    selection_rule: str = "minimum_brier_then_ece_then_method"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "calibrator": self.calibrator.to_record(),
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "observation_count": self.observation_count,
            "reliability_bins": self.reliability_bins,
            "selection_rule": self.selection_rule,
            "fit_partition": "validation",
        }


def calibration_metrics(
    labels: Sequence[int], probabilities: Sequence[float], *, bins: int = 10
) -> CalibrationMetrics:
    """Calculate Brier score, ECE, and fixed-bin reliability-curve evidence."""

    count = _validate_inputs(probabilities, labels, require_both_classes=False)
    if bins <= 0:
        raise ProbabilityCalibrationError("reliability bin count must be positive")
    bucket_labels: list[list[int]] = [[] for _ in range(bins)]
    bucket_probabilities: list[list[float]] = [[] for _ in range(bins)]
    for label, probability in zip(labels, probabilities, strict=True):
        index = min(bins - 1, int(probability * bins))
        bucket_labels[index].append(label)
        bucket_probabilities[index].append(probability)
    curve: list[ReliabilityBin] = []
    ece = 0.0
    for index, (observed, predicted) in enumerate(
        zip(bucket_labels, bucket_probabilities, strict=True)
    ):
        mean_probability = math.fsum(predicted) / len(predicted) if predicted else None
        observed_frequency = math.fsum(observed) / len(observed) if observed else None
        if mean_probability is not None and observed_frequency is not None:
            ece += len(observed) / count * abs(observed_frequency - mean_probability)
        curve.append(
            ReliabilityBin(
                lower=index / bins,
                upper=(index + 1) / bins,
                count=len(observed),
                mean_probability=mean_probability,
                observed_frequency=observed_frequency,
            )
        )
    brier = (
        math.fsum(
            (probability - label) ** 2
            for label, probability in zip(labels, probabilities, strict=True)
        )
        / count
    )
    return CalibrationMetrics(brier, ece, tuple(curve))


def fit_platt_calibrator(
    validation_probabilities: Sequence[float],
    validation_labels: Sequence[int],
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> PlattCalibrator:
    """Fit deterministic two-parameter logistic scaling with Newton updates."""

    _validate_inputs(validation_probabilities, validation_labels, require_both_classes=True)
    if max_iterations <= 0 or not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ProbabilityCalibrationError("Platt optimization bounds must be positive and finite")
    logits = tuple(_logit(value) for value in validation_probabilities)
    slope = 1.0
    positives = sum(validation_labels)
    intercept = math.log((positives + 1.0) / (len(validation_labels) - positives + 1.0))
    regularization = 1e-9
    for _ in range(max_iterations):
        predictions = tuple(_sigmoid(slope * value + intercept) for value in logits)
        gradient_slope = (
            math.fsum(
                (prediction - label) * value
                for prediction, label, value in zip(
                    predictions, validation_labels, logits, strict=True
                )
            )
            + regularization * slope
        )
        gradient_intercept = math.fsum(
            prediction - label
            for prediction, label in zip(predictions, validation_labels, strict=True)
        )
        weights = tuple(prediction * (1.0 - prediction) for prediction in predictions)
        hessian_ss = (
            math.fsum(weight * value * value for weight, value in zip(weights, logits, strict=True))
            + regularization
        )
        hessian_si = math.fsum(
            weight * value for weight, value in zip(weights, logits, strict=True)
        )
        hessian_ii = math.fsum(weights) + regularization
        determinant = hessian_ss * hessian_ii - hessian_si * hessian_si
        if determinant <= 1e-18:
            break
        delta_slope = (hessian_ii * gradient_slope - hessian_si * gradient_intercept) / determinant
        delta_intercept = (
            hessian_ss * gradient_intercept - hessian_si * gradient_slope
        ) / determinant
        slope -= delta_slope
        intercept -= delta_intercept
        if max(abs(delta_slope), abs(delta_intercept)) <= tolerance:
            break
    return PlattCalibrator(slope=slope, intercept=intercept)


def fit_isotonic_calibrator(
    validation_probabilities: Sequence[float], validation_labels: Sequence[int]
) -> IsotonicCalibrator:
    """Fit deterministic isotonic regression using pair-adjacent violators."""

    _validate_inputs(validation_probabilities, validation_labels, require_both_classes=True)
    grouped: list[list[float]] = []
    for probability, label in sorted(zip(validation_probabilities, validation_labels, strict=True)):
        if grouped and grouped[-1][0] == probability:
            grouped[-1][1] += label
            grouped[-1][2] += 1.0
        else:
            grouped.append([probability, float(label), 1.0])
    blocks: list[list[float]] = []
    block_ranges: list[list[int]] = []
    for index, (_, label_sum, weight) in enumerate(grouped):
        blocks.append([label_sum, weight])
        block_ranges.append([index, index])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop()
            left = blocks.pop()
            right_range = block_ranges.pop()
            left_range = block_ranges.pop()
            blocks.append([left[0] + right[0], left[1] + right[1]])
            block_ranges.append([left_range[0], right_range[1]])
    values = [0.0] * len(grouped)
    for (label_sum, weight), (start, end) in zip(blocks, block_ranges, strict=True):
        values[start : end + 1] = [label_sum / weight] * (end - start + 1)
    return IsotonicCalibrator(thresholds=tuple(item[0] for item in grouped), values=tuple(values))


def fit_probability_calibrator(
    validation_probabilities: Sequence[float],
    validation_labels: Sequence[int],
    *,
    reliability_bins: int = 10,
) -> CalibrationSelection:
    """Fit and select a calibrator using validation observations only.

    The deliberately partition-specific parameter names prevent a test partition from
    entering the fitting API accidentally. Final test evaluation is a separate concern.
    """

    count = _validate_inputs(validation_probabilities, validation_labels, require_both_classes=True)
    calibrators: tuple[PlattCalibrator | IsotonicCalibrator, ...] = (
        fit_platt_calibrator(validation_probabilities, validation_labels),
        fit_isotonic_calibrator(validation_probabilities, validation_labels),
    )
    evaluated = tuple(
        (
            calibrator,
            calibration_metrics(
                validation_labels,
                calibrator.calibrate(validation_probabilities),
                bins=reliability_bins,
            ),
        )
        for calibrator in calibrators
    )
    selected, _ = min(
        evaluated,
        key=lambda item: (
            item[1].brier_score,
            item[1].expected_calibration_error,
            item[0].method.value,
        ),
    )
    candidates = tuple(CalibrationCandidate(item.method, metrics) for item, metrics in evaluated)
    return CalibrationSelection(selected, candidates, count, reliability_bins)


def calibrator_from_record(record: Mapping[str, object]) -> PlattCalibrator | IsotonicCalibrator:
    method_value = record.get("method")
    if method_value == CalibrationMethod.PLATT.value:
        return PlattCalibrator.from_record(record)
    if method_value == CalibrationMethod.ISOTONIC.value:
        return IsotonicCalibrator.from_record(record)
    raise ProbabilityCalibrationError(f"unsupported calibration method: {method_value!r}")


def _validate_inputs(
    probabilities: Sequence[float], labels: Sequence[int], *, require_both_classes: bool
) -> int:
    if len(probabilities) != len(labels) or not probabilities:
        raise ProbabilityCalibrationError("calibration arrays must be non-empty and equal length")
    _validate_probabilities(probabilities)
    if any(label not in (0, 1) for label in labels):
        raise ProbabilityCalibrationError("calibration labels must be binary 0/1")
    if require_both_classes and len(set(labels)) != 2:
        raise ProbabilityCalibrationError("calibrator fitting requires both binary classes")
    return len(probabilities)


def _validate_probabilities(probabilities: Sequence[float]) -> None:
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities):
        raise ProbabilityCalibrationError("probabilities must be finite and within [0, 1]")


def _logit(probability: float) -> float:
    clipped = min(1.0 - _LOGIT_EPSILON, max(_LOGIT_EPSILON, probability))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _validate_record_header(record: Mapping[str, object], method: CalibrationMethod) -> None:
    if record.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ProbabilityCalibrationError("unsupported calibration schema version")
    if record.get("method") != method.value:
        raise ProbabilityCalibrationError(f"serialized calibrator is not {method.value}")


def _record_float(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProbabilityCalibrationError(f"serialized {key} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ProbabilityCalibrationError(f"serialized {key} must be finite")
    return converted


def _record_float_tuple(record: Mapping[str, object], key: str) -> tuple[float, ...]:
    value = record.get(key)
    if not isinstance(value, list):
        raise ProbabilityCalibrationError(f"serialized {key} must be an array")
    return tuple(_record_float({key: item}, key) for item in value)

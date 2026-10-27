"""Validation-only calibration for safe-mutation probabilities."""

from __future__ import annotations

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol, Self, cast

CALIBRATION_SCHEMA_VERSION: Final[int] = 1
CALIBRATION_SELECTION_SCHEMA_VERSION: Final[int] = 2
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

    @classmethod
    def from_record(cls, record: object) -> Self:
        value = _record_mapping(
            record,
            "reliability bin",
            {"lower", "upper", "count", "mean_probability", "observed_frequency"},
        )
        lower = _record_float(value, "lower")
        upper = _record_float(value, "upper")
        count = _record_int(value, "count", minimum=0)
        mean_probability = _record_optional_probability(value, "mean_probability")
        observed_frequency = _record_optional_probability(value, "observed_frequency")
        if not 0.0 <= lower < upper <= 1.0:
            raise ProbabilityCalibrationError("reliability bin bounds are invalid")
        if count == 0:
            if mean_probability is not None or observed_frequency is not None:
                raise ProbabilityCalibrationError("empty reliability bins cannot have means")
        elif mean_probability is None or observed_frequency is None:
            raise ProbabilityCalibrationError("populated reliability bins require both means")
        return cls(lower, upper, count, mean_probability, observed_frequency)


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

    @classmethod
    def from_record(cls, record: object) -> Self:
        value = _record_mapping(
            record,
            "calibration metrics",
            {"brier_score", "expected_calibration_error", "reliability_curve"},
        )
        brier = _record_float(value, "brier_score")
        ece = _record_float(value, "expected_calibration_error")
        raw_curve = value["reliability_curve"]
        if (
            not 0.0 <= brier <= 1.0
            or not 0.0 <= ece <= 1.0
            or not isinstance(raw_curve, list)
            or not raw_curve
        ):
            raise ProbabilityCalibrationError("calibration metrics are outside their valid range")
        curve = tuple(ReliabilityBin.from_record(item) for item in raw_curve)
        count = sum(item.count for item in curve)
        if count <= 0:
            raise ProbabilityCalibrationError("reliability curve must contain observations")
        for index, item in enumerate(curve):
            expected_lower = index / len(curve)
            expected_upper = (index + 1) / len(curve)
            if not math.isclose(item.lower, expected_lower, abs_tol=1e-12) or not math.isclose(
                item.upper, expected_upper, abs_tol=1e-12
            ):
                raise ProbabilityCalibrationError("reliability bins must form a fixed partition")
        computed_ece = math.fsum(
            item.count / count * abs(item.observed_frequency - item.mean_probability)
            for item in curve
            if item.count > 0
            and item.mean_probability is not None
            and item.observed_frequency is not None
        )
        if not math.isclose(ece, computed_ece, rel_tol=1e-9, abs_tol=1e-12):
            raise ProbabilityCalibrationError("ECE does not match the reliability curve")
        return cls(brier, ece, curve)


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

    @classmethod
    def from_record(cls, record: object) -> Self:
        value = _record_mapping(record, "calibration candidate", {"method", "metrics"})
        raw_method = value["method"]
        if not isinstance(raw_method, str):
            raise ProbabilityCalibrationError("calibration candidate method is invalid")
        try:
            method = CalibrationMethod(raw_method)
        except (TypeError, ValueError) as error:
            raise ProbabilityCalibrationError("calibration candidate method is invalid") from error
        return cls(method, CalibrationMetrics.from_record(value["metrics"]))


@dataclass(frozen=True, slots=True)
class CalibrationSelection:
    calibrator: PlattCalibrator | IsotonicCalibrator
    candidates: tuple[CalibrationCandidate, ...]
    observation_count: int
    reliability_bins: int
    selection_rule: str = "minimum_brier_then_ece_then_method"
    cross_validation_folds: int = 0
    group_aware_folds: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.cross_validation_folds, bool)
            or not isinstance(self.cross_validation_folds, int)
            or self.cross_validation_folds < 0
        ):
            raise ProbabilityCalibrationError("cross-validation fold count cannot be negative")
        if self.group_aware_folds and self.cross_validation_folds < 2:
            raise ProbabilityCalibrationError("group-aware folds require cross-validation")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": CALIBRATION_SELECTION_SCHEMA_VERSION,
            "calibrator": self.calibrator.to_record(),
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "observation_count": self.observation_count,
            "reliability_bins": self.reliability_bins,
            "cross_validation_folds": self.cross_validation_folds,
            "selection_rule": self.selection_rule,
            "fit_partition": "validation",
            "evaluation_partition": (
                "validation_out_of_fold" if self.cross_validation_folds >= 2 else "unspecified"
            ),
            "fold_assignment": (
                "stratified_group_greedy_v1"
                if self.group_aware_folds
                else "stratified_round_robin_v1"
                if self.cross_validation_folds >= 2
                else "unspecified"
            ),
        }

    @classmethod
    def from_record(cls, record: object) -> Self:
        required = {
            "schema_version",
            "calibrator",
            "candidates",
            "observation_count",
            "reliability_bins",
            "cross_validation_folds",
            "selection_rule",
            "fit_partition",
            "evaluation_partition",
            "fold_assignment",
        }
        value = _record_mapping(record, "calibration selection", required)
        if _record_int(value, "schema_version", minimum=1) != CALIBRATION_SELECTION_SCHEMA_VERSION:
            raise ProbabilityCalibrationError("unsupported calibration selection schema version")
        raw_calibrator = value["calibrator"]
        if not isinstance(raw_calibrator, Mapping):
            raise ProbabilityCalibrationError("selected calibrator must be an object")
        calibrator_record = cast(Mapping[str, object], raw_calibrator)
        method = calibrator_record.get("method")
        calibrator_fields = (
            {"schema_version", "method", "slope", "intercept"}
            if method == CalibrationMethod.PLATT.value
            else {"schema_version", "method", "thresholds", "values"}
            if method == CalibrationMethod.ISOTONIC.value
            else set()
        )
        if not calibrator_fields or set(calibrator_record) != calibrator_fields:
            raise ProbabilityCalibrationError("selected calibrator fields are malformed")
        selected = calibrator_from_record(calibrator_record)
        raw_candidates = value["candidates"]
        if not isinstance(raw_candidates, list):
            raise ProbabilityCalibrationError("calibration candidates must be an array")
        candidates = tuple(CalibrationCandidate.from_record(item) for item in raw_candidates)
        methods = tuple(item.method for item in candidates)
        if len(set(methods)) != len(methods):
            raise ProbabilityCalibrationError("calibration candidate methods must be unique")
        observation_count = _record_int(value, "observation_count", minimum=1)
        reliability_bins = _record_int(value, "reliability_bins", minimum=1)
        cross_validation_folds = _record_int(value, "cross_validation_folds", minimum=0)
        if cross_validation_folds == 1 or (
            cross_validation_folds >= 2 and observation_count < 2 * cross_validation_folds
        ):
            raise ProbabilityCalibrationError("calibration cross-validation evidence is invalid")
        for candidate in candidates:
            if (
                len(candidate.metrics.reliability_curve) != reliability_bins
                or sum(item.count for item in candidate.metrics.reliability_curve)
                != observation_count
            ):
                raise ProbabilityCalibrationError(
                    "candidate reliability evidence does not match the selection dimensions"
                )
        if candidates and selected.method not in methods:
            raise ProbabilityCalibrationError("selected calibrator has no candidate evidence")
        if cross_validation_folds >= 2 and set(methods) != set(CalibrationMethod):
            raise ProbabilityCalibrationError("cross-fitted selection requires both candidates")
        selection_rule = value["selection_rule"]
        if not isinstance(selection_rule, str) or not selection_rule.strip():
            raise ProbabilityCalibrationError("selection rule must be a non-empty string")
        if value["fit_partition"] != "validation":
            raise ProbabilityCalibrationError("calibration fit partition must be validation")
        expected_evaluation = (
            "validation_out_of_fold" if cross_validation_folds >= 2 else "unspecified"
        )
        raw_assignment = value["fold_assignment"]
        group_aware_folds = raw_assignment == "stratified_group_greedy_v1"
        expected_assignment = (
            "stratified_group_greedy_v1"
            if group_aware_folds
            else "stratified_round_robin_v1"
            if cross_validation_folds >= 2
            else "unspecified"
        )
        if (
            value["evaluation_partition"] != expected_evaluation
            or value["fold_assignment"] != expected_assignment
        ):
            raise ProbabilityCalibrationError("calibration evaluation provenance is inconsistent")
        return cls(
            selected,
            candidates,
            observation_count,
            reliability_bins,
            selection_rule,
            cross_validation_folds,
            group_aware_folds,
        )


def calibration_metrics(
    labels: Sequence[int], probabilities: Sequence[float], *, bins: int = 10
) -> CalibrationMetrics:
    """Calculate Brier score, ECE, and fixed-bin reliability-curve evidence."""

    count = _validate_inputs(probabilities, labels, require_both_classes=False)
    _validate_positive_integer(bins, "reliability bin count")
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
    _validate_positive_integer(max_iterations, "Platt maximum iteration count")
    try:
        tolerance_value = float(tolerance)
    except (OverflowError, TypeError, ValueError) as error:
        raise ProbabilityCalibrationError(
            "Platt optimization bounds must be positive and finite"
        ) from error
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance_value)
        or tolerance_value <= 0.0
    ):
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
        if max(abs(delta_slope), abs(delta_intercept)) <= tolerance_value:
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
    group_ids: Sequence[str] | None = None,
) -> CalibrationSelection:
    """Fit and select a calibrator using validation observations only.

    The deliberately partition-specific parameter names prevent a test partition from
    entering the fitting API accidentally. Final test evaluation is a separate concern.
    """

    count = _validate_inputs(validation_probabilities, validation_labels, require_both_classes=True)
    _validate_positive_integer(reliability_bins, "reliability bin count")
    class_counts = (validation_labels.count(0), validation_labels.count(1))
    if group_ids is not None:
        if len(group_ids) != count or any(
            not isinstance(group, str) or not group for group in group_ids
        ):
            raise ProbabilityCalibrationError(
                "group IDs must be non-empty strings aligned with calibration rows"
            )
        group_indices: dict[str, list[int]] = {}
        group_class_counts: dict[str, list[int]] = {}
        for index, (label, group) in enumerate(zip(validation_labels, group_ids, strict=True)):
            group_indices.setdefault(group, []).append(index)
            counts = group_class_counts.setdefault(group, [0, 0])
            counts[label] += 1
        class_group_counts = tuple(
            sum(counts[label] > 0 for counts in group_class_counts.values()) for label in (0, 1)
        )
        folds = min(5, *class_group_counts)
        fold_indices = _stratified_group_fold_indices(
            group_indices, group_class_counts, class_counts, folds
        )
    else:
        folds = min(5, *class_counts)
        fold_indices = _stratified_row_fold_indices(validation_labels, folds)
    if folds < 2:
        raise ProbabilityCalibrationError(
            "cross-fitted calibration selection requires at least two examples or groups per class"
        )

    cross_fitted: dict[CalibrationMethod, list[float | None]] = {
        CalibrationMethod.PLATT: [None] * count,
        CalibrationMethod.ISOTONIC: [None] * count,
    }
    for held_out_indices in fold_indices:
        held_out = set(held_out_indices)
        training_indices = tuple(index for index in range(count) if index not in held_out)
        training_probabilities = tuple(
            validation_probabilities[index] for index in training_indices
        )
        training_labels = tuple(validation_labels[index] for index in training_indices)
        calibrators: tuple[PlattCalibrator | IsotonicCalibrator, ...] = (
            fit_platt_calibrator(training_probabilities, training_labels),
            fit_isotonic_calibrator(training_probabilities, training_labels),
        )
        held_out_probabilities = tuple(
            validation_probabilities[index] for index in held_out_indices
        )
        for calibrator in calibrators:
            predictions = calibrator.calibrate(held_out_probabilities)
            output = cross_fitted[calibrator.method]
            for index, prediction in zip(held_out_indices, predictions, strict=True):
                output[index] = prediction

    evaluated: list[tuple[CalibrationMethod, CalibrationMetrics]] = []
    for method in (CalibrationMethod.PLATT, CalibrationMethod.ISOTONIC):
        output_values = cross_fitted[method]
        complete_predictions: list[float] = []
        for oof_prediction in output_values:
            if oof_prediction is None:
                raise ProbabilityCalibrationError(
                    "cross-fitting did not predict every validation row"
                )
            complete_predictions.append(oof_prediction)
        metrics = calibration_metrics(
            validation_labels,
            complete_predictions,
            bins=reliability_bins,
        )
        evaluated.append((method, metrics))
    selected, _ = min(
        evaluated,
        key=lambda item: (
            item[1].brier_score,
            item[1].expected_calibration_error,
            item[0].value,
        ),
    )
    calibrator = (
        fit_platt_calibrator(validation_probabilities, validation_labels)
        if selected is CalibrationMethod.PLATT
        else fit_isotonic_calibrator(validation_probabilities, validation_labels)
    )
    candidates = tuple(CalibrationCandidate(method, metrics) for method, metrics in evaluated)
    return CalibrationSelection(
        calibrator,
        candidates,
        count,
        reliability_bins,
        selection_rule="minimum_cross_fitted_brier_then_ece_then_method",
        cross_validation_folds=folds,
        group_aware_folds=group_ids is not None,
    )


def _stratified_row_fold_indices(labels: Sequence[int], folds: int) -> list[list[int]]:
    fold_indices: list[list[int]] = [[] for _ in range(folds)]
    class_positions = [0, 0]
    for index, label in enumerate(labels):
        fold = class_positions[label] % folds
        fold_indices[fold].append(index)
        class_positions[label] += 1
    return fold_indices


def _stratified_group_fold_indices(
    group_indices: Mapping[str, list[int]],
    group_class_counts: Mapping[str, list[int]],
    class_counts: tuple[int, int],
    folds: int,
) -> list[list[int]]:
    if folds < 2:
        return []
    group_order = sorted(
        group_indices,
        key=lambda group: (
            -int(group_class_counts[group][0] > 0 and group_class_counts[group][1] > 0),
            -abs(
                group_class_counts[group][0] / class_counts[0]
                - group_class_counts[group][1] / class_counts[1]
            ),
            -len(group_indices[group]),
            group,
        ),
    )
    fold_counts = [[0, 0] for _ in range(folds)]
    fold_sizes = [0] * folds
    assigned: list[list[str]] = [[] for _ in range(folds)]
    remaining_support = [[0, 0] for _ in range(len(group_order) + 1)]
    for position in range(len(group_order) - 1, -1, -1):
        group = group_order[position]
        remaining_support[position] = remaining_support[position + 1].copy()
        for label in (0, 1):
            remaining_support[position][label] += int(group_class_counts[group][label] > 0)
    for group_position, group in enumerate(group_order):
        counts = group_class_counts[group]
        size = len(group_indices[group])
        options: list[tuple[float, int, int]] = []
        for fold in range(folds):
            preserves_training_classes = True
            for label in (0, 1):
                assigned_support = sum(fold_counts[index][label] > 0 for index in range(folds))
                if counts[label] > 0 and fold_counts[fold][label] == 0:
                    assigned_support += 1
                if assigned_support + remaining_support[group_position + 1][label] < 2:
                    preserves_training_classes = False
                    break
            if not preserves_training_classes:
                continue
            imbalance = 0.0
            for label in (0, 1):
                distribution = [
                    (fold_counts[index][label] + (counts[label] if index == fold else 0))
                    / class_counts[label]
                    for index in range(folds)
                ]
                mean = math.fsum(distribution) / folds
                imbalance += math.fsum((value - mean) ** 2 for value in distribution) / folds
            options.append((imbalance, fold_sizes[fold] + size, fold))
        if not options:
            raise ProbabilityCalibrationError(
                "group-aware calibration folds cannot retain both classes in every training split"
            )
        selected_fold = min(options)[2]
        assigned[selected_fold].append(group)
        fold_sizes[selected_fold] += size
        for label in (0, 1):
            fold_counts[selected_fold][label] += counts[label]
    if any(sum(fold_counts[fold][label] > 0 for fold in range(folds)) < 2 for label in (0, 1)):
        raise ProbabilityCalibrationError(
            "group-aware calibration folds cannot retain both classes in every training split"
        )
    result = [[index for group in fold for index in group_indices[group]] for fold in assigned]
    if any(not fold for fold in result):
        raise ProbabilityCalibrationError("group-aware calibration produced an empty fold")
    return result


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


def _validate_positive_integer(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProbabilityCalibrationError(f"{label} must be a positive integer")


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
    schema_version = record.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != CALIBRATION_SCHEMA_VERSION
    ):
        raise ProbabilityCalibrationError("unsupported calibration schema version")
    if record.get("method") != method.value:
        raise ProbabilityCalibrationError(f"serialized calibrator is not {method.value}")


def _record_float(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProbabilityCalibrationError(f"serialized {key} must be numeric")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ProbabilityCalibrationError(f"serialized {key} must be finite") from error
    if not math.isfinite(converted):
        raise ProbabilityCalibrationError(f"serialized {key} must be finite")
    return converted


def _record_float_tuple(record: Mapping[str, object], key: str) -> tuple[float, ...]:
    value = record.get(key)
    if not isinstance(value, list):
        raise ProbabilityCalibrationError(f"serialized {key} must be an array")
    return tuple(_record_float({key: item}, key) for item in value)


def _record_mapping(value: object, label: str, required: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != required:
        raise ProbabilityCalibrationError(f"serialized {label} fields are malformed")
    return cast(Mapping[str, object], value)


def _record_int(record: Mapping[str, object], key: str, *, minimum: int) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProbabilityCalibrationError(f"serialized {key} must be an integer >= {minimum}")
    return value


def _record_optional_probability(record: Mapping[str, object], key: str) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    probability = _record_float({key: value}, key)
    if not 0.0 <= probability <= 1.0:
        raise ProbabilityCalibrationError(f"serialized {key} must be within [0, 1]")
    return probability

from __future__ import annotations

import inspect
from copy import deepcopy

import pytest

from modelsurgeon.surgeon.calibration import (
    CalibrationMethod,
    CalibrationSelection,
    IsotonicCalibrator,
    PlattCalibrator,
    ProbabilityCalibrationError,
    _stratified_group_fold_indices,
    calibration_metrics,
    calibrator_from_record,
    fit_isotonic_calibrator,
    fit_platt_calibrator,
    fit_probability_calibrator,
)

PROBABILITIES = (0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95)
LABELS = (0, 0, 1, 0, 1, 0, 1, 1)


def test_platt_calibration_is_deterministic_and_serializable() -> None:
    first = fit_platt_calibrator(PROBABILITIES, LABELS)
    second = fit_platt_calibrator(PROBABILITIES, LABELS)

    assert first == second
    assert calibrator_from_record(first.to_record()) == first
    assert all(0.0 <= value <= 1.0 for value in first.calibrate((0.0, 0.5, 1.0)))


def test_isotonic_calibration_pools_violations_and_round_trips() -> None:
    fitted = fit_isotonic_calibrator(PROBABILITIES, LABELS)

    assert all(left <= right for left, right in zip(fitted.values, fitted.values[1:], strict=False))
    assert fitted.values[2] == fitted.values[3]
    assert fitted.values[4] == fitted.values[5]
    restored = calibrator_from_record(fitted.to_record())
    assert restored == fitted
    assert restored.calibrate((0.0, 0.5, 1.0)) == fitted.calibrate((0.0, 0.5, 1.0))


def test_selection_records_validation_evidence_and_fixed_reliability_bins() -> None:
    selection = fit_probability_calibrator(PROBABILITIES, LABELS, reliability_bins=4)
    assert fit_probability_calibrator(PROBABILITIES, LABELS, reliability_bins=4) == selection
    record = selection.to_record()

    assert selection.calibrator.method in {CalibrationMethod.PLATT, CalibrationMethod.ISOTONIC}
    assert selection.calibrator.method is CalibrationMethod.PLATT
    assert {candidate.method for candidate in selection.candidates} == {
        CalibrationMethod.PLATT,
        CalibrationMethod.ISOTONIC,
    }
    assert all(len(candidate.metrics.reliability_curve) == 4 for candidate in selection.candidates)
    assert record["fit_partition"] == "validation"
    assert record["schema_version"] == 2
    assert record["evaluation_partition"] == "validation_out_of_fold"
    assert record["fold_assignment"] == "stratified_round_robin_v1"
    assert record["cross_validation_folds"] == 4
    assert record["selection_rule"] == "minimum_cross_fitted_brier_then_ece_then_method"
    assert "test_labels" not in inspect.signature(fit_probability_calibrator).parameters
    assert CalibrationSelection.from_record(record) == selection

    fitted_metrics = calibration_metrics(
        LABELS, selection.calibrator.calibrate(PROBABILITIES), bins=4
    )
    selected_metrics = next(
        candidate.metrics
        for candidate in selection.candidates
        if candidate.method is selection.calibrator.method
    )
    assert selected_metrics.brier_score != pytest.approx(fitted_metrics.brier_score)


def test_grouped_crossfit_keeps_group_rows_together_and_records_protocol() -> None:
    labels = LABELS
    groups = tuple(f"group-{index // 2}" for index in range(len(labels)))
    group_indices: dict[str, list[int]] = {}
    group_class_counts: dict[str, list[int]] = {}
    class_counts = (labels.count(0), labels.count(1))
    for index, (label, group) in enumerate(zip(labels, groups, strict=True)):
        group_indices.setdefault(group, []).append(index)
        counts = group_class_counts.setdefault(group, [0, 0])
        counts[label] += 1
    class_group_counts = tuple(
        sum(counts[label] > 0 for counts in group_class_counts.values()) for label in (0, 1)
    )
    folds = min(5, *class_group_counts)
    assigned_folds = _stratified_group_fold_indices(
        group_indices, group_class_counts, class_counts, folds
    )
    fold_by_row = {index: fold for fold, indices in enumerate(assigned_folds) for index in indices}
    for group, indices in group_indices.items():
        assert len({fold_by_row[index] for index in indices}) == 1, group

    selection = fit_probability_calibrator(PROBABILITIES, labels, group_ids=groups)
    assert selection.group_aware_folds
    assert selection.cross_validation_folds == folds
    assert selection.to_record()["fold_assignment"] == "stratified_group_greedy_v1"
    assert CalibrationSelection.from_record(selection.to_record()) == selection


def test_grouped_crossfit_fails_closed_without_two_groups_per_class() -> None:
    with pytest.raises(ProbabilityCalibrationError, match="two examples or groups per class"):
        fit_probability_calibrator(
            (0.1, 0.2, 0.8, 0.9),
            (0, 0, 1, 1),
            group_ids=("one-negative-group", "one-negative-group", "positive-a", "positive-b"),
        )


def test_manually_constructed_selection_does_not_claim_fold_provenance() -> None:
    selection = CalibrationSelection(
        fit_platt_calibrator(PROBABILITIES, LABELS),
        (),
        len(PROBABILITIES),
        reliability_bins=4,
    )

    record = selection.to_record()
    assert record["cross_validation_folds"] == 0
    assert record["evaluation_partition"] == "unspecified"
    assert record["fold_assignment"] == "unspecified"
    assert CalibrationSelection.from_record(record) == selection


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), 2.0),
        (("fit_partition",), "test"),
        (("fold_assignment",), "random"),
        (("cross_validation_folds",), 1),
        (("candidates", 0, "metrics", "expected_calibration_error"), 0.9),
        (("candidates", 0, "metrics", "reliability_curve", 0, "count"), -1),
        (("calibrator", "method"), "unknown"),
    ],
)
def test_selection_record_rejects_inconsistent_or_malformed_evidence(
    path: tuple[str | int, ...], value: object
) -> None:
    record = fit_probability_calibrator(PROBABILITIES, LABELS, reliability_bins=4).to_record()
    malformed = deepcopy(record)
    target: object = malformed
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ProbabilityCalibrationError):
        CalibrationSelection.from_record(malformed)


def test_calibration_metrics_keep_empty_bins_explicit() -> None:
    metrics = calibration_metrics((0, 1), (0.1, 0.2), bins=4)

    assert metrics.brier_score == pytest.approx(0.325)
    assert len(metrics.reliability_curve) == 4
    assert metrics.reliability_curve[-1].count == 0
    assert metrics.reliability_curve[-1].mean_probability is None


@pytest.mark.parametrize(
    ("probabilities", "labels"),
    [
        ((), ()),
        ((0.2,), (0, 1)),
        ((float("nan"), 0.8), (0, 1)),
        ((-0.1, 0.8), (0, 1)),
        ((0.2, 0.8), (0, 2)),
        ((0.2, 0.8), (1, 1)),
        ((0.2, 0.3, 0.8), (0, 0, 1)),
    ],
)
def test_fitting_fails_closed_for_malformed_or_single_class_inputs(
    probabilities: tuple[float, ...], labels: tuple[int, ...]
) -> None:
    with pytest.raises(ProbabilityCalibrationError):
        fit_probability_calibrator(probabilities, labels)


def test_serialized_calibrators_fail_closed() -> None:
    with pytest.raises(ProbabilityCalibrationError, match="schema version"):
        PlattCalibrator.from_record(
            {"schema_version": 2, "method": "platt", "slope": 1.0, "intercept": 0.0}
        )
    with pytest.raises(ProbabilityCalibrationError, match="strictly increasing"):
        IsotonicCalibrator((0.2, 0.2), (0.0, 1.0))

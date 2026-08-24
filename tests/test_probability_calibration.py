from __future__ import annotations

import inspect

import pytest

from modelsurgeon.surgeon.calibration import (
    CalibrationMethod,
    IsotonicCalibrator,
    PlattCalibrator,
    ProbabilityCalibrationError,
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
    record = selection.to_record()

    assert selection.calibrator.method in {CalibrationMethod.PLATT, CalibrationMethod.ISOTONIC}
    assert {candidate.method for candidate in selection.candidates} == {
        CalibrationMethod.PLATT,
        CalibrationMethod.ISOTONIC,
    }
    assert all(len(candidate.metrics.reliability_curve) == 4 for candidate in selection.candidates)
    assert record["fit_partition"] == "validation"
    assert record["selection_rule"] == "minimum_brier_then_ece_then_method"
    assert "test_labels" not in inspect.signature(fit_probability_calibrator).parameters


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

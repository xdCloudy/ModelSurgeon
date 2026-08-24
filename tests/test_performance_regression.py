"""Tests for hardware-bound performance regression decisions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from modelsurgeon.evaluation.performance_regression import (
    PERFORMANCE_BUDGET_SCHEMA_VERSION,
    PerformanceBudget,
    PerformanceBudgetManifest,
    PerformanceMeasurement,
    PerformanceRegressionError,
    RegressionDirection,
    RegressionLane,
    evaluate_performance_regression,
    load_performance_budget_manifest,
)
from modelsurgeon.experiments.runtime_telemetry import HardwareNormalizationContext


def _hardware(*, processor: str = "test-cpu") -> HardwareNormalizationContext:
    return HardwareNormalizationContext(
        "Test OS",
        "x86_64",
        processor,
        8,
        16 * 1024**3,
        False,
        (),
    )


def _budgets() -> tuple[PerformanceBudget, ...]:
    return (
        PerformanceBudget(
            "cpu_case",
            "cpu",
            "throughput",
            "items_per_second",
            RegressionDirection.MINIMUM,
            100.0,
            0.1,
            5.0,
        ),
        PerformanceBudget(
            "cpu_case",
            "cpu",
            "wall_seconds",
            "seconds",
            RegressionDirection.MAXIMUM,
            10.0,
            0.1,
            0.5,
        ),
    )


def _manifest(*, repetitions: int = 3) -> PerformanceBudgetManifest:
    return PerformanceBudgetManifest(
        "test-profile",
        RegressionLane.PR_CPU,
        "fixture-v1",
        repetitions,
        {"class": "test runner", "cpu": "test-cpu"},
        _budgets(),
    )


def _measurements(
    *,
    wall: tuple[float, ...] = (5.0, 12.0, 10.0),
    throughput: tuple[float, ...] = (85.0, 95.0, 90.0),
    hardware: HardwareNormalizationContext | None = None,
) -> tuple[PerformanceMeasurement, ...]:
    context = hardware or _hardware()
    values: list[PerformanceMeasurement] = []
    for repetition, value in enumerate(throughput):
        values.append(
            PerformanceMeasurement(
                "cpu_case",
                "cpu",
                "throughput",
                "items_per_second",
                value,
                repetition,
                "fixture-v1",
                context,
            )
        )
    for repetition, value in enumerate(wall):
        values.append(
            PerformanceMeasurement(
                "cpu_case",
                "cpu",
                "wall_seconds",
                "seconds",
                value,
                repetition,
                "fixture-v1",
                context,
            )
        )
    return tuple(values)


def test_regression_report_uses_medians_and_both_threshold_directions() -> None:
    manifest = _manifest()

    report = evaluate_performance_regression(manifest, _measurements())

    assert report.passed
    assert [result.observed_median for result in report.results] == [90.0, 10.0]
    assert [result.budget.threshold for result in report.results] == [90.0, 11.0]
    assert report.hardware.context_id == _hardware().context_id
    record = report.to_record()
    assert record["hardware"] == _hardware().to_record()
    assert record["reference_hardware"] == manifest.reference_hardware
    assert record["budget_manifest_id"] == manifest.manifest_id
    assert record["report_id"] == report.report_id
    assert report.report_id == evaluate_performance_regression(
        manifest, _measurements()
    ).report_id


@pytest.mark.parametrize(
    ("wall", "throughput", "failed_metric"),
    [
        ((11.1, 12.0, 1.0), (90.0, 90.0, 90.0), "wall_seconds"),
        ((11.0, 11.0, 11.0), (80.0, 89.0, 200.0), "throughput"),
    ],
)
def test_regression_report_emits_actionable_alerts(
    wall: tuple[float, ...],
    throughput: tuple[float, ...],
    failed_metric: str,
) -> None:
    report = evaluate_performance_regression(
        _manifest(),
        _measurements(wall=wall, throughput=throughput),
    )

    assert not report.passed
    failed = [result.to_record() for result in report.results if not result.passed]
    assert len(failed) == 1
    assert failed[0]["budget"]["metric"] == failed_metric  # type: ignore[index]
    assert "violates" in str(failed[0]["alert"])


def test_regression_rejects_empty_or_incomplete_measurements() -> None:
    manifest = _manifest()

    with pytest.raises(PerformanceRegressionError, match="requires measurements"):
        evaluate_performance_regression(manifest, ())
    with pytest.raises(PerformanceRegressionError, match="fewer than 3 repetitions"):
        evaluate_performance_regression(manifest, _measurements()[:-1])
    wall_only = tuple(
        measurement
        for measurement in _measurements()
        if measurement.metric == "wall_seconds"
    )
    with pytest.raises(PerformanceRegressionError, match="fewer than 3 repetitions"):
        evaluate_performance_regression(manifest, wall_only)


def test_regression_rejects_mixed_hardware_fixture_and_unknown_metrics() -> None:
    manifest = _manifest()
    measurements = list(_measurements())
    measurements[-1] = replace(measurements[-1], hardware=_hardware(processor="other"))
    with pytest.raises(PerformanceRegressionError, match="mix incomparable hardware"):
        evaluate_performance_regression(manifest, tuple(measurements))

    wrong_fixture = tuple(
        replace(measurement, fixture_id="fixture-v2")
        for measurement in _measurements()
    )
    with pytest.raises(PerformanceRegressionError, match="budget fixture"):
        evaluate_performance_regression(manifest, wrong_fixture)

    unknown = (*_measurements(), replace(_measurements()[0], metric="unknown"))
    with pytest.raises(PerformanceRegressionError, match="no declared budget"):
        evaluate_performance_regression(manifest, unknown)


def test_regression_rejects_duplicate_repetitions_and_unit_drift() -> None:
    manifest = _manifest()
    duplicate = list(_measurements())
    duplicate[1] = replace(duplicate[1], repetition=0)
    with pytest.raises(PerformanceRegressionError, match="repeats a repetition index"):
        evaluate_performance_regression(manifest, tuple(duplicate))

    wrong_unit = list(_measurements())
    wrong_unit[0] = replace(wrong_unit[0], unit="seconds")
    with pytest.raises(PerformanceRegressionError, match="inconsistent units"):
        evaluate_performance_regression(manifest, tuple(wrong_unit))


def test_budget_and_measurement_validation_is_fail_closed() -> None:
    budget = _budgets()[0]
    with pytest.raises(PerformanceRegressionError, match="finite and non-negative"):
        replace(budget, baseline=-1.0)
    with pytest.raises(PerformanceRegressionError, match="finite and non-negative"):
        replace(budget, absolute_tolerance=float("nan"))
    with pytest.raises(PerformanceRegressionError, match="require canonical names"):
        replace(budget, metric="")

    measurement = _measurements()[0]
    with pytest.raises(PerformanceRegressionError, match="finite and non-negative"):
        replace(measurement, value=float("inf"))
    with pytest.raises(PerformanceRegressionError, match="repetition and fixture"):
        replace(measurement, repetition=-1)


def test_manifest_requires_unique_sorted_budgets_and_reference_context() -> None:
    with pytest.raises(PerformanceRegressionError, match="unique canonical keys"):
        replace(_manifest(), budgets=tuple(reversed(_budgets())))
    with pytest.raises(PerformanceRegressionError, match="unique canonical keys"):
        replace(_manifest(), budgets=(_budgets()[0], _budgets()[0]))
    with pytest.raises(PerformanceRegressionError, match="reference hardware"):
        replace(_manifest(), reference_hardware={})
    with pytest.raises(PerformanceRegressionError, match="repetitions must be positive"):
        replace(_manifest(), min_repetitions=0)


def test_manifest_identity_uses_an_immutable_reference_hardware_snapshot() -> None:
    reference: dict[str, object] = {"cpu": "before", "devices": [{"name": "gpu"}]}
    manifest = replace(_manifest(), reference_hardware=reference)
    manifest_id = manifest.manifest_id

    reference["cpu"] = "after"
    cast(list[dict[str, str]], reference["devices"])[0]["name"] = "changed"

    assert manifest.manifest_id == manifest_id
    assert manifest.reference_hardware_record == {
        "cpu": "before",
        "devices": [{"name": "gpu"}],
    }


def _budget_file() -> dict[str, object]:
    path = Path("tests/fixtures/performance_budgets_v1.json")
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_checked_in_budget_profiles_are_strict_and_hardware_bound() -> None:
    payload = json.dumps(_budget_file())

    pr = load_performance_budget_manifest(payload, "pr-cpu")
    cpu = load_performance_budget_manifest(payload, "consumer-cpu")
    gpu = load_performance_budget_manifest(payload, "consumer-gpu")

    assert pr.schema_version == PERFORMANCE_BUDGET_SCHEMA_VERSION
    assert pr.lane is RegressionLane.PR_CPU
    assert cpu.lane is RegressionLane.LARGE_CPU
    assert gpu.lane is RegressionLane.GPU
    assert pr.fixture_id == "pr-cpu-v1"
    assert "cpu_processor" in cpu.reference_hardware
    assert "cuda_devices" in gpu.reference_hardware
    assert all(
        manifest.manifest_id.startswith("performance_budget_")
        for manifest in (pr, cpu, gpu)
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda root: root.update({"unknown": True}), "budget file fields are invalid"),
        (lambda root: root.update({"schema_version": 99}), "unsupported performance budget"),
        (
            lambda root: root["profiles"]["pr-cpu"].update({"unknown": True}),
            "profile fields are invalid",
        ),
        (
            lambda root: root["profiles"]["pr-cpu"]["budgets"][0].update(
                {"unknown": True}
            ),
            "budget 0 fields are invalid",
        ),
    ],
)
def test_budget_loader_rejects_schema_drift(
    mutation: object,
    message: str,
) -> None:
    root = _budget_file()
    assert callable(mutation)
    mutation(root)

    with pytest.raises(PerformanceRegressionError, match=message):
        load_performance_budget_manifest(json.dumps(root), "pr-cpu")


def test_budget_loader_rejects_invalid_json_unknown_profiles_and_values() -> None:
    with pytest.raises(PerformanceRegressionError, match="invalid JSON"):
        load_performance_budget_manifest("{", "pr-cpu")
    with pytest.raises(PerformanceRegressionError, match="unknown performance profile"):
        load_performance_budget_manifest(json.dumps(_budget_file()), "missing")

    root = _budget_file()
    profiles = cast(dict[str, dict[str, Any]], root["profiles"])
    profile = profiles["pr-cpu"]
    profile["min_repetitions"] = True
    with pytest.raises(PerformanceRegressionError, match="must be an integer"):
        load_performance_budget_manifest(json.dumps(root), "pr-cpu")

    root = _budget_file()
    profiles = cast(dict[str, dict[str, Any]], root["profiles"])
    profile = profiles["pr-cpu"]
    profile["budgets"][0]["direction"] = "sideways"
    with pytest.raises(PerformanceRegressionError, match="direction is invalid"):
        load_performance_budget_manifest(json.dumps(root), "pr-cpu")

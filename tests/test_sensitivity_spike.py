"""Tests for the bounded Fisher/Hessian sensitivity research spike."""

from __future__ import annotations

import pytest

from modelsurgeon.features.sensitivity_spike import (
    SensitivityBudget,
    SensitivityMethod,
    SensitivityProbe,
    SensitivitySpikeError,
    default_tiny_probes,
    evaluate_sensitivity_methods,
)


def test_two_tiny_models_record_signal_stability_cost_and_recommend_fisher() -> None:
    probes = default_tiny_probes()
    report = evaluate_sensitivity_methods(
        probes,
        SensitivityBudget(max_workspace_bytes=4096, max_elapsed_seconds=1.0),
    )

    assert report.probes == ("tiny_quadratic_a", "tiny_quadratic_b")
    assert len(report.results) == 6
    assert report.recommendation is SensitivityMethod.EMPIRICAL_FISHER
    assert "recommend empirical_fisher" in report.rationale

    aggregates = {item.method: item for item in report.aggregates}
    fisher = aggregates[SensitivityMethod.EMPIRICAL_FISHER]
    hessian = aggregates[SensitivityMethod.DIAGONAL_HESSIAN]
    first_order = aggregates[SensitivityMethod.FIRST_ORDER]

    assert fisher.predictive_spearman == pytest.approx(1.0)
    assert hessian.predictive_spearman == pytest.approx(1.0)
    assert fisher.ranking_stability == pytest.approx(1.0)
    assert hessian.ranking_stability == pytest.approx(1.0)
    assert first_order.predictive_spearman < fisher.predictive_spearman
    assert fisher.max_workspace_bytes < hessian.max_workspace_bytes
    assert fisher.total_operation_units < hessian.total_operation_units
    assert all(item.total_elapsed_seconds >= 0.0 for item in report.aggregates)
    assert all(item.feasible for item in report.aggregates)

    record = report.to_record()
    assert record["recommendation"] == "empirical_fisher"
    assert record["budget"] == {
        "max_workspace_bytes": 4096,
        "max_elapsed_seconds": 1.0,
    }


def test_fixed_memory_budget_can_reject_all_candidates() -> None:
    report = evaluate_sensitivity_methods(
        default_tiny_probes(),
        SensitivityBudget(max_workspace_bytes=1, max_elapsed_seconds=1.0),
    )

    assert report.recommendation is None
    assert "reject all candidates" in report.rationale
    assert not any(item.feasible for item in report.aggregates)


def test_probe_and_spike_validation_fail_closed() -> None:
    with pytest.raises(SensitivitySpikeError, match="at least two gradient"):
        SensitivityProbe(
            "bad",
            (1.0,),
            ((1.0,),),
            (1.0,),
            (1.0,),
        )

    with pytest.raises(SensitivitySpikeError, match="at least two tiny probes"):
        evaluate_sensitivity_methods((default_tiny_probes()[0],))

    with pytest.raises(SensitivitySpikeError, match="predictive tolerance"):
        evaluate_sensitivity_methods(default_tiny_probes(), predictive_tolerance=-1.0)

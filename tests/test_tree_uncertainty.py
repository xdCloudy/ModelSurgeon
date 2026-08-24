from __future__ import annotations

import pytest

from modelsurgeon.active_learning.tree_uncertainty import (
    TREE_UNCERTAINTY_SCHEMA_VERSION,
    TreeMethodEvidence,
    TreeUncertaintyBudget,
    TreeUncertaintyError,
    TreeUncertaintyMethod,
    compare_tree_uncertainty,
    estimate_from_members,
    estimate_from_quantiles,
)


def _evidence(
    method: TreeUncertaintyMethod,
    predictions: tuple[object, ...],
    *,
    cpu_seconds: float,
) -> TreeMethodEvidence:
    return TreeMethodEvidence(
        method,
        predictions,  # type: ignore[arg-type]
        fit_count=3,
        cpu_seconds=cpu_seconds,
        model_bytes=100,
        technique_version=f"{method.value}-v1",
    )


def test_member_intervals_are_deterministic_bounded_and_versioned() -> None:
    members = ((0.0, 1.0), (0.2, 1.4), (0.4, 1.8))

    first = estimate_from_members(TreeUncertaintyMethod.ENSEMBLE, members, confidence=0.8)
    second = estimate_from_members(TreeUncertaintyMethod.ENSEMBLE, members, confidence=0.8)

    assert first == second
    assert first[0].point == pytest.approx(0.2)
    assert first[0].uncertainty == pytest.approx(0.2)
    assert first[0].to_record()["schema_version"] == TREE_UNCERTAINTY_SCHEMA_VERSION


def test_comparison_reports_coverage_ranking_and_cost_then_selects() -> None:
    targets = (0.0, 1.0, 2.0, 3.0)
    ensemble = estimate_from_members(
        TreeUncertaintyMethod.ENSEMBLE,
        ((0.0, 1.2, 1.2, 3.5), (0.0, 0.8, 2.8, 2.5), (0.0, 1.0, 2.0, 3.0)),
        confidence=0.75,
    )
    bootstrap = estimate_from_members(
        TreeUncertaintyMethod.BOOTSTRAP,
        ((-0.1, 0.9, 1.9, 2.9), (0.1, 1.1, 2.1, 3.1), (0.0, 1.0, 2.0, 3.0)),
        confidence=0.75,
    )
    quantile = estimate_from_quantiles(
        (-0.5, 0.5, 1.5, 2.5),
        (0.0, 1.0, 2.0, 3.0),
        (0.5, 1.5, 2.5, 3.5),
    )

    study = compare_tree_uncertainty(
        targets,
        (
            _evidence(TreeUncertaintyMethod.ENSEMBLE, ensemble, cpu_seconds=4.0),
            _evidence(TreeUncertaintyMethod.BOOTSTRAP, bootstrap, cpu_seconds=3.0),
            _evidence(TreeUncertaintyMethod.QUANTILE, quantile, cpu_seconds=2.0),
        ),
        confidence=0.75,
    )
    record = study.to_record()

    assert {candidate.method for candidate in study.candidates} == set(TreeUncertaintyMethod)
    assert all(0.0 <= candidate.coverage <= 1.0 for candidate in study.candidates)
    assert all(candidate.prediction_value_count == 4 for candidate in study.candidates)
    assert record["selection_rule"] == "coverage_then_ranking_then_cpu_bytes_method"
    assert record["candidates"][0]["cost"]["cpu_seconds"] >= 0.0  # type: ignore[index]
    assert all(
        prediction.schema_version == TREE_UNCERTAINTY_SCHEMA_VERSION
        for prediction in study.selected.predictions
    )


def test_fixed_budget_fails_closed() -> None:
    predictions = estimate_from_quantiles((0.0,), (0.5,), (1.0,))
    evidence = (
        _evidence(TreeUncertaintyMethod.ENSEMBLE, predictions, cpu_seconds=1.0),
        _evidence(TreeUncertaintyMethod.BOOTSTRAP, predictions, cpu_seconds=1.0),
        _evidence(TreeUncertaintyMethod.QUANTILE, predictions, cpu_seconds=11.0),
    )

    with pytest.raises(TreeUncertaintyError, match="CPU budget"):
        compare_tree_uncertainty(
            (0.5,),
            evidence,
            budget=TreeUncertaintyBudget(max_cpu_seconds_per_method=10.0),
        )


def test_prediction_memory_and_interval_contracts_fail_closed() -> None:
    with pytest.raises(TreeUncertaintyError, match="consumer-memory"):
        estimate_from_members(
            TreeUncertaintyMethod.ENSEMBLE,
            ((0.0, 1.0), (0.1, 1.1)),
            max_prediction_values=3,
        )
    with pytest.raises(TreeUncertaintyError, match="contain"):
        estimate_from_quantiles((0.6,), (0.5,), (1.0,))
    with pytest.raises(TreeUncertaintyError, match="ensemble, bootstrap, and quantile"):
        compare_tree_uncertainty(
            (0.5,),
            (
                _evidence(
                    TreeUncertaintyMethod.QUANTILE,
                    estimate_from_quantiles((0.0,), (0.5,), (1.0,)),
                    cpu_seconds=1.0,
                ),
            ),
        )

from __future__ import annotations

import pytest

from modelsurgeon.active_learning import (
    ActiveLearningStudyError,
    LearningCurve,
    LearningCurvePoint,
    SelectionStrategy,
    render_learning_curve_svg,
    summarize_active_learning_study,
)


def _curve(strategy, seed, values=(0.5, 0.6, 0.7)):
    return LearningCurve(
        strategy,
        seed,
        tuple(
            LearningCurvePoint(budget, value)
            for budget, value in zip((0, 10, 20), values, strict=True)
        ),
    )


def test_equal_budget_study_computes_aulc_intervals_and_plot() -> None:
    curves = tuple(
        _curve(strategy, seed, (0.5, 0.6 + 0.01 * seed, 0.7))
        for strategy in SelectionStrategy
        for seed in (1, 2, 3)
    )

    study = summarize_active_learning_study(curves, bootstrap_repetitions=100, bootstrap_seed=7)
    svg = render_learning_curve_svg(study)

    assert all(summary.mean_aulc == pytest.approx(0.61) for summary in study.summaries)
    assert study.negative_result is not None
    assert "<svg" in svg and "polyline" in svg


def test_unequal_budgets_fail_closed() -> None:
    curves = [_curve(strategy, 1) for strategy in SelectionStrategy]
    curves[-1] = LearningCurve(
        SelectionStrategy.UTILITY_ONLY,
        1,
        (LearningCurvePoint(0, 0.5), LearningCurvePoint(30, 0.7)),
    )

    with pytest.raises(ActiveLearningStudyError, match="identical budgets"):
        summarize_active_learning_study(curves)

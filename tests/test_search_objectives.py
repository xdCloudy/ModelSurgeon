import pytest

from modelsurgeon.config import (
    ObjectiveConfig,
    ObjectiveDirection,
    ObjectiveNormalization,
    ObjectiveTermConfig,
    OptimizeMetric,
    Settings,
)
from modelsurgeon.search.objectives import (
    ObjectiveError,
    ObjectiveObservation,
    ObjectiveSet,
    ObjectiveTerm,
    objectives_from_config,
)


def test_weighted_normalized_reward_composes_all_dimensions() -> None:
    objectives = ObjectiveSet(
        (
            ObjectiveTerm(
                OptimizeMetric.QUALITY,
                ObjectiveDirection.MAXIMIZE,
                2.0,
                ObjectiveNormalization.IDENTITY,
            ),
            ObjectiveTerm(
                OptimizeMetric.PARAMETER_COUNT,
                ObjectiveDirection.MINIMIZE,
                1.0,
            ),
            ObjectiveTerm(
                OptimizeMetric.LATENCY,
                ObjectiveDirection.MINIMIZE,
                1.0,
            ),
            ObjectiveTerm(
                OptimizeMetric.MEMORY,
                ObjectiveDirection.MINIMIZE,
                1.0,
            ),
            ObjectiveTerm(
                OptimizeMetric.DISK_SIZE,
                ObjectiveDirection.MINIMIZE,
                1.0,
            ),
        )
    )
    score = objectives.score(
        (
            ObjectiveObservation(OptimizeMetric.QUALITY, 0.98),
            ObjectiveObservation(OptimizeMetric.PARAMETER_COUNT, 80, 100),
            ObjectiveObservation(OptimizeMetric.LATENCY, 70, 100),
            ObjectiveObservation(OptimizeMetric.MEMORY, 60, 100),
            ObjectiveObservation(OptimizeMetric.DISK_SIZE, 50, 100),
        )
    )
    assert score.reward == pytest.approx((2 * 0.98 - 0.8 - 0.7 - 0.6 - 0.5) / 6)
    assert [item.metric for item in score.contributions] == sorted(
        OptimizeMetric, key=lambda metric: metric.value
    )


def test_config_and_run_identity_include_canonical_objective_definition() -> None:
    first = ObjectiveConfig(
        terms=(
            ObjectiveTermConfig(
                metric=OptimizeMetric.LATENCY,
                direction=ObjectiveDirection.MINIMIZE,
                weight=1,
            ),
        )
    )
    second = ObjectiveConfig(
        terms=(
            ObjectiveTermConfig(
                metric=OptimizeMetric.LATENCY,
                direction=ObjectiveDirection.MINIMIZE,
                weight=2,
            ),
        )
    )
    assert (
        objectives_from_config(first).objective_set_id
        != objectives_from_config(second).objective_set_id
    )
    assert Settings(objective=first).canonical_json() != Settings(objective=second).canonical_json()


def test_min_max_and_missing_baseline_validation_are_explicit() -> None:
    term = ObjectiveTerm(
        OptimizeMetric.QUALITY,
        ObjectiveDirection.MAXIMIZE,
        normalization=ObjectiveNormalization.MIN_MAX,
        minimum=0.8,
        maximum=1.0,
    )
    assert ObjectiveSet((term,)).score(
        (ObjectiveObservation(OptimizeMetric.QUALITY, 0.9),)
    ).reward == pytest.approx(0.5)
    ratio = ObjectiveSet((ObjectiveTerm(OptimizeMetric.LATENCY, ObjectiveDirection.MINIMIZE),))
    with pytest.raises(ObjectiveError, match="requires a baseline"):
        ratio.score((ObjectiveObservation(OptimizeMetric.LATENCY, 10),))
    with pytest.raises(ObjectiveError, match="missing observation"):
        ratio.score(())


def test_legacy_optimize_dimensions_remain_supported_without_hard_coding() -> None:
    objectives = objectives_from_config(
        ObjectiveConfig(optimize=(OptimizeMetric.DISK_SIZE, OptimizeMetric.QUALITY))
    )
    assert {term.metric for term in objectives.terms} == {
        OptimizeMetric.DISK_SIZE,
        OptimizeMetric.QUALITY,
    }

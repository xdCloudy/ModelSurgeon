from typing import cast

import pytest

from modelsurgeon.config import ObjectiveDirection, ObjectiveNormalization, OptimizeMetric
from modelsurgeon.search.constraints import (
    BaselineReference,
    ConstraintMetric,
    ConstraintObservation,
    ConstraintSet,
    OptimizationConstraint,
)
from modelsurgeon.search.objectives import (
    ObjectiveObservation,
    ObjectiveSet,
    ObjectiveTerm,
)
from modelsurgeon.search.policies import (
    PredictedSearchCandidate,
    SearchPolicy,
    SearchPolicyConfig,
    SearchPolicyKind,
)

OBJECTIVES = ObjectiveSet(
    (
        ObjectiveTerm(
            OptimizeMetric.QUALITY,
            ObjectiveDirection.MAXIMIZE,
            normalization=ObjectiveNormalization.IDENTITY,
        ),
    )
)
CONSTRAINTS = ConstraintSet(
    (
        OptimizationConstraint(
            ConstraintMetric.QUALITY_RETENTION,
            0.95,
            BaselineReference.IMMUTABLE_SOURCE,
        ),
    )
)


def _candidate(
    candidate_id: str,
    reward: float,
    *,
    uncertainty: float = 0,
    retention: float = 1,
) -> PredictedSearchCandidate:
    return PredictedSearchCandidate(
        candidate_id,
        "state_parent",
        (ObjectiveObservation(OptimizeMetric.QUALITY, reward),),
        (
            ConstraintObservation(
                ConstraintMetric.QUALITY_RETENTION,
                retention,
                BaselineReference.IMMUTABLE_SOURCE,
            ),
        ),
        uncertainty,
    )


def test_greedy_is_seeded_resumable_and_stores_predicted_outcomes() -> None:
    policy = SearchPolicy(
        SearchPolicyConfig(SearchPolicyKind.GREEDY, evaluation_budget=2, seed=7),
        OBJECTIVES,
        CONSTRAINTS,
    )
    candidates = (
        _candidate("a", 0.8),
        _candidate("b", 0.9),
        _candidate("unsafe", 1, retention=0.8),
    )
    first = policy.select(candidates)
    assert [decision.candidate_id for decision in first.selected] == ["b"]
    assert first.selected[0].objective_score is not None
    assert first.selected[0].reason == "highest_predicted_reward"
    second = policy.select(candidates, first.next_state)
    assert [decision.candidate_id for decision in second.selected] == ["a"]
    assert second.budget_exhausted is True
    assert policy.select(candidates).to_record() == first.to_record()


def test_beam_and_uncertainty_aware_apply_distinct_acquisition_rules() -> None:
    candidates = (
        _candidate("certain", 0.9, uncertainty=0.01),
        _candidate("uncertain", 0.8, uncertainty=0.2),
        _candidate("third", 0.7),
    )
    beam = SearchPolicy(
        SearchPolicyConfig(SearchPolicyKind.BEAM, evaluation_budget=3, beam_width=2),
        OBJECTIVES,
        CONSTRAINTS,
    ).select(candidates)
    assert {decision.candidate_id for decision in beam.selected} == {"certain", "uncertain"}

    uncertainty = SearchPolicy(
        SearchPolicyConfig(
            SearchPolicyKind.UNCERTAINTY_AWARE,
            evaluation_budget=1,
            beam_width=1,
            exploration_weight=1,
        ),
        OBJECTIVES,
        CONSTRAINTS,
    ).select(candidates)
    assert [decision.candidate_id for decision in uncertainty.selected] == ["uncertain"]
    assert uncertainty.selected[0].acquisition_score == pytest.approx(1.0)


def test_policy_configuration_and_duplicate_pool_fail_closed() -> None:
    with pytest.raises(ValueError, match="beam width one"):
        SearchPolicyConfig(SearchPolicyKind.GREEDY, 1, beam_width=2)
    policy = SearchPolicy(
        SearchPolicyConfig(SearchPolicyKind.BEAM, 2, beam_width=2),
        OBJECTIVES,
        CONSTRAINTS,
    )
    duplicate = _candidate("same", 1)
    with pytest.raises(ValueError, match="unique"):
        policy.select((duplicate, duplicate))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_budget", 1.5),
        ("evaluation_budget", True),
        ("evaluation_budget", "2"),
        ("beam_width", 1.5),
        ("beam_width", True),
        ("beam_width", "2"),
        ("seed", 1.5),
        ("seed", True),
        ("seed", "2"),
    ],
)
def test_policy_rejects_non_integer_budget_width_and_seed(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        SearchPolicyConfig(
            SearchPolicyKind.BEAM,
            evaluation_budget=cast(int, value) if field == "evaluation_budget" else 2,
            beam_width=cast(int, value) if field == "beam_width" else 1,
            seed=cast(int, value) if field == "seed" else 0,
        )


@pytest.mark.parametrize("value", [1.5, True, "1"])
def test_search_resume_cursor_requires_non_negative_integer(value: object) -> None:
    from modelsurgeon.search.policies import SearchPolicyState

    with pytest.raises(ValueError, match="decision index must be a non-negative integer"):
        SearchPolicyState("policy_test", decision_index=cast(int, value))

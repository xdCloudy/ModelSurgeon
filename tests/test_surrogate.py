from dataclasses import replace
from itertools import combinations

import pytest

from modelsurgeon.config import ObjectiveDirection, ObjectiveNormalization, OptimizeMetric
from modelsurgeon.search.architecture_policies import (
    ArchitectureEvidenceStatus,
    CompleteArchitectureState,
)
from modelsurgeon.search.candidate_space import ArchitectureCandidate
from modelsurgeon.search.constraints import (
    BaselineReference,
    ConstraintMetric,
    ConstraintObservation,
    ConstraintSet,
    OptimizationConstraint,
)
from modelsurgeon.search.deployable_state import ArchitectureAxis
from modelsurgeon.search.objectives import ObjectiveSet, ObjectiveTerm
from modelsurgeon.search.pareto import ParetoCandidate, ParetoObjectiveValue
from modelsurgeon.search.surrogate import (
    AcquisitionKind,
    SurrogateArchitecturePolicy,
    SurrogateBudget,
    SurrogateConfig,
    SurrogateError,
    SurrogateFitStatus,
    SurrogateModel,
    SurrogatePolicyState,
    SurrogatePrediction,
    SurrogateTrainingExample,
    _feasible_frontier,
    _hypervolume_1d_or_2d,
    fit_surrogate,
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


def _state(
    value: int,
    quality: float,
    status: ArchitectureEvidenceStatus,
    *,
    retention: float = 1.0,
) -> CompleteArchitectureState:
    return CompleteArchitectureState(
        ArchitectureCandidate(
            "state_root",
            ((ArchitectureAxis.QUERY_HEADS, value),),
            "cpu",
            value * 100,
            value * 10,
            float(value),
        ),
        (ParetoObjectiveValue(OptimizeMetric.QUALITY, quality),),
        (
            ConstraintObservation(
                ConstraintMetric.QUALITY_RETENTION,
                retention,
                BaselineReference.IMMUTABLE_SOURCE,
            ),
        ),
        status,
    )


def _measured() -> tuple[CompleteArchitectureState, ...]:
    return tuple(
        _state(value, 0.70 + (value * 0.04), ArchitectureEvidenceStatus.MEASURED)
        for value in range(1, 6)
    )


def test_surrogate_fit_is_calibrated_bounded_and_persistable() -> None:
    config = SurrogateConfig(AcquisitionKind.EXPECTED_IMPROVEMENT, 3, seed=11)
    fit = fit_surrogate(_measured(), OBJECTIVES, CONSTRAINTS, config)
    assert fit.status is SurrogateFitStatus.FITTED
    assert fit.model is not None
    assert fit.held_out_count == 1
    assert fit.calibration_rmse is not None
    assert fit.calibration_coverage is not None
    assert fit.calibration_feasibility_brier is not None
    record = fit.to_record()
    assert record["model"] is not None
    assert record["retained_outcome_ids"] == []
    assert fit.from_record(record).to_record() == record


def test_surrogate_acquisition_is_deterministic_constrained_and_resumable() -> None:
    measured = _measured()
    candidates = (
        _state(6, 0.95, ArchitectureEvidenceStatus.PREDICTED),
        _state(7, 0.95, ArchitectureEvidenceStatus.PREDICTED),
        _state(8, 0.95, ArchitectureEvidenceStatus.PREDICTED),
        _state(9, 0.95, ArchitectureEvidenceStatus.PREDICTED, retention=0.8),
        _state(10, 0.99, ArchitectureEvidenceStatus.UNSUPPORTED),
    )
    config = SurrogateConfig(
        AcquisitionKind.HYPERVOLUME_IMPROVEMENT,
        evaluation_budget=3,
        batch_size=2,
        seed=19,
    )
    fit = fit_surrogate(measured, OBJECTIVES, CONSTRAINTS, config)
    policy = SurrogateArchitecturePolicy(config, OBJECTIVES, CONSTRAINTS)
    first = policy.select(candidates, fit)
    second = policy.select(candidates, fit)
    assert first.to_record() == second.to_record()
    assert len(first.selected) == 2
    assert all(item.candidate_id != candidates[3].candidate_id for item in first.selected)
    assert any(item.reason == "predicted_constraint_violation" for item in first.decisions)
    assert any(item.reason == "retained_unsupported_evidence" for item in first.decisions)
    assert SurrogatePolicyState.from_record(first.next_state.to_record()) == first.next_state


def test_surrogate_falls_back_explicitly_and_respects_fit_budgets() -> None:
    candidate = _state(6, 0.95, ArchitectureEvidenceStatus.PREDICTED)
    insufficient = fit_surrogate(
        (_state(1, 0.8, ArchitectureEvidenceStatus.MEASURED),),
        OBJECTIVES,
        CONSTRAINTS,
        SurrogateConfig(AcquisitionKind.EXPECTED_IMPROVEMENT, 1),
    )
    assert insufficient.status is SurrogateFitStatus.INSUFFICIENT_DATA
    fallback = SurrogateArchitecturePolicy(
        SurrogateConfig(AcquisitionKind.EXPECTED_IMPROVEMENT, 1),
        OBJECTIVES,
        CONSTRAINTS,
    ).select((candidate,), insufficient)
    assert fallback.selected[0].reason == "fallback_insufficient_data"

    budget_limited = fit_surrogate(
        _measured(),
        OBJECTIVES,
        CONSTRAINTS,
        SurrogateConfig(
            AcquisitionKind.EXPECTED_IMPROVEMENT,
            1,
            budget=SurrogateBudget(max_model_bytes=1),
        ),
    )
    assert budget_limited.status is SurrogateFitStatus.BUDGET_EXCEEDED


def test_surrogate_rejects_foreign_resume_state() -> None:
    state = _state(6, 0.95, ArchitectureEvidenceStatus.PREDICTED)
    fit = fit_surrogate(
        _measured(),
        OBJECTIVES,
        CONSTRAINTS,
        SurrogateConfig(AcquisitionKind.EXPECTED_IMPROVEMENT, 1),
    )
    policy = SurrogateArchitecturePolicy(
        SurrogateConfig(AcquisitionKind.EXPECTED_IMPROVEMENT, 1),
        OBJECTIVES,
        CONSTRAINTS,
    )
    with pytest.raises(SurrogateError, match="another policy"):
        policy.select((state,), fit, SurrogatePolicyState("surrogate_policy_foreign"))


def test_exact_two_objective_hypervolume_and_feasible_frontier() -> None:
    assert _hypervolume_1d_or_2d(((0.8, 0.7), (0.7, 0.8)), (0.0, 0.0)) == pytest.approx(0.63)
    assert _hypervolume_1d_or_2d(((0.8, 0.7),), (0.0, 0.0)) == pytest.approx(0.56)
    assert _hypervolume_1d_or_2d(((0.8, -0.1),), (0.0, 0.0)) == 0.0

    examples = (
        SurrogateTrainingExample(
            "feasible", (0.5,), (ParetoObjectiveValue(OptimizeMetric.QUALITY, 0.8),), True
        ),
        SurrogateTrainingExample(
            "infeasible", (0.5,), (ParetoObjectiveValue(OptimizeMetric.QUALITY, 0.99),), False
        ),
    )
    assert tuple(item.candidate_id for item in _feasible_frontier(examples)) == ("feasible",)


def test_one_objective_hypervolume_rejects_overflow() -> None:
    with pytest.raises(SurrogateError, match="finite numeric range"):
        _hypervolume_1d_or_2d(((1.7976931348623157e308,),), (-1.7976931348623157e308,))


def test_two_objective_hypervolume_matches_independent_union_oracle() -> None:
    reference = (0.0, 0.0)
    grid = tuple((x, y) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0))

    def inclusion_exclusion(points: tuple[tuple[float, float], ...]) -> float:
        rectangles = tuple((max(0.0, x), max(0.0, y)) for x, y in points)
        area = 0.0
        for size in range(1, len(rectangles) + 1):
            sign = 1.0 if size % 2 else -1.0
            for subset in combinations(rectangles, size):
                area += sign * min(item[0] for item in subset) * min(item[1] for item in subset)
        return area

    cases = (
        *(points for size in range(1, 5) for points in combinations(grid, size)),
        ((1.0, 0.5), (1.0, 0.5)),
    )

    for points in cases:
        expected = inclusion_exclusion(points)
        assert _hypervolume_1d_or_2d(points, reference) == pytest.approx(expected)
        assert _hypervolume_1d_or_2d(tuple(reversed(points)), reference) == pytest.approx(expected)


def test_hypervolume_acquisition_uses_objective_directions_and_reference_units() -> None:
    objectives = ObjectiveSet(
        (
            ObjectiveTerm(
                OptimizeMetric.QUALITY,
                ObjectiveDirection.MAXIMIZE,
                normalization=ObjectiveNormalization.IDENTITY,
            ),
            ObjectiveTerm(
                OptimizeMetric.DISK_SIZE,
                ObjectiveDirection.MINIMIZE,
                normalization=ObjectiveNormalization.IDENTITY,
            ),
        )
    )
    config = SurrogateConfig(
        AcquisitionKind.HYPERVOLUME_IMPROVEMENT,
        evaluation_budget=2,
        reference_point=(150.0, 0.6),
    )
    policy = SurrogateArchitecturePolicy(config, objectives, CONSTRAINTS)
    frontier = (
        ParetoCandidate(
            "frontier-a",
            (
                ParetoObjectiveValue(OptimizeMetric.DISK_SIZE, 100.0),
                ParetoObjectiveValue(OptimizeMetric.QUALITY, 0.8),
            ),
            {},
        ),
        ParetoCandidate(
            "frontier-b",
            (
                ParetoObjectiveValue(OptimizeMetric.DISK_SIZE, 50.0),
                ParetoObjectiveValue(OptimizeMetric.QUALITY, 0.7),
            ),
            {},
        ),
    )
    useful = SurrogatePrediction(
        "useful",
        (
            ParetoObjectiveValue(OptimizeMetric.DISK_SIZE, 45.0),
            ParetoObjectiveValue(OptimizeMetric.QUALITY, 0.75),
        ),
        1.0,
        0.0,
        0.0,
    )
    outside_reference = SurrogatePrediction(
        "outside",
        (
            ParetoObjectiveValue(OptimizeMetric.DISK_SIZE, 200.0),
            ParetoObjectiveValue(OptimizeMetric.QUALITY, 0.95),
        ),
        1.0,
        0.0,
        0.0,
    )
    reference = policy._effective_reference_point((frontier[0].objectives, frontier[1].objectives))
    assert reference == (-150.0, 0.6)
    assert policy._acquisition(useful, frontier, reference) == pytest.approx(3.25)
    assert policy._acquisition(outside_reference, frontier, reference) == 0.0


def test_hypervolume_reference_is_recorded_in_objective_space() -> None:
    config = SurrogateConfig(
        AcquisitionKind.HYPERVOLUME_IMPROVEMENT,
        evaluation_budget=1,
    )
    fit = fit_surrogate(_measured(), OBJECTIVES, CONSTRAINTS, config)
    selection = SurrogateArchitecturePolicy(config, OBJECTIVES, CONSTRAINTS).select(
        (_state(6, 0.95, ArchitectureEvidenceStatus.PREDICTED),), fit
    )
    assert fit.model is not None
    observed = tuple(example.objectives[0].estimate for example in fit.model.examples)
    expected_reference = min(observed) - 0.1 * (max(observed) - min(observed))
    assert selection.acquisition_reference_point == pytest.approx((expected_reference,))
    assert selection.to_record()["acquisition_reference_point"] == pytest.approx(
        [expected_reference]
    )

    assert fit.model is not None
    polluted_model = replace(
        fit.model,
        examples=(
            *fit.model.examples,
            SurrogateTrainingExample(
                "infeasible-outlier",
                fit.model.examples[0].features,
                (ParetoObjectiveValue(OptimizeMetric.QUALITY, 1000.0),),
                False,
            ),
        ),
    )
    polluted_fit = replace(fit, model=polluted_model)
    polluted = SurrogateArchitecturePolicy(config, OBJECTIVES, CONSTRAINTS).select(
        (_state(6, 0.95, ArchitectureEvidenceStatus.PREDICTED),), polluted_fit
    )
    assert polluted.acquisition_reference_point == pytest.approx((expected_reference,))


def test_hypervolume_batch_recomputes_marginal_gain_after_each_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SurrogateConfig(
        AcquisitionKind.HYPERVOLUME_IMPROVEMENT,
        evaluation_budget=2,
        batch_size=2,
        exploration_weight=0.0,
        reference_point=(0.6,),
    )
    fit = fit_surrogate(_measured(), OBJECTIVES, CONSTRAINTS, config)

    def predict_identical_point(
        self: SurrogateModel, candidate_id: str, features: tuple[float, ...]
    ) -> SurrogatePrediction:
        del self, features
        return SurrogatePrediction(
            candidate_id,
            (ParetoObjectiveValue(OptimizeMetric.QUALITY, 0.95),),
            1.0,
            0.0,
            0.0,
        )

    monkeypatch.setattr(SurrogateModel, "predict", predict_identical_point)
    candidates = (
        _state(6, 0.95, ArchitectureEvidenceStatus.PREDICTED),
        _state(7, 0.95, ArchitectureEvidenceStatus.PREDICTED),
    )
    selection = SurrogateArchitecturePolicy(config, OBJECTIVES, CONSTRAINTS).select(candidates, fit)

    chosen = [decision for decision in selection.selected]
    assert len(chosen) == 2
    assert fit.model is not None
    best_measured = max(
        example.objectives[0].estimate for example in fit.model.examples if example.feasible
    )
    assert sorted(decision.acquisition for decision in chosen) == pytest.approx(
        [0.0, 0.95 - best_measured]
    )


def test_hypervolume_rejects_more_than_two_objectives() -> None:
    objectives = ObjectiveSet(
        tuple(
            ObjectiveTerm(metric, ObjectiveDirection.MAXIMIZE)
            for metric in (
                OptimizeMetric.QUALITY,
                OptimizeMetric.PARAMETER_COUNT,
                OptimizeMetric.DISK_SIZE,
            )
        )
    )
    config = SurrogateConfig(AcquisitionKind.HYPERVOLUME_IMPROVEMENT, evaluation_budget=1)
    with pytest.raises(SurrogateError, match="at most two"):
        SurrogateArchitecturePolicy(config, objectives, CONSTRAINTS)


def test_hypervolume_reference_rejects_overflow_and_non_numeric_values() -> None:
    with pytest.raises(SurrogateError, match="reference point must contain finite values"):
        SurrogateConfig(
            AcquisitionKind.HYPERVOLUME_IMPROVEMENT,
            evaluation_budget=1,
            reference_point=(10**400,),
        )
    with pytest.raises(SurrogateError, match="reference point must contain finite values"):
        SurrogateConfig(
            AcquisitionKind.HYPERVOLUME_IMPROVEMENT,
            evaluation_budget=1,
            reference_point=("0.5",),
        )

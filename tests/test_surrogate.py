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
from modelsurgeon.search.pareto import ParetoObjectiveValue
from modelsurgeon.search.surrogate import (
    AcquisitionKind,
    SurrogateArchitecturePolicy,
    SurrogateBudget,
    SurrogateConfig,
    SurrogateError,
    SurrogateFitStatus,
    SurrogatePolicyState,
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

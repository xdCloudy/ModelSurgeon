import pytest

from modelsurgeon.config import ObjectiveDirection, ObjectiveNormalization, OptimizeMetric
from modelsurgeon.search.architecture_policies import (
    ArchitectureEvidenceStatus,
    ArchitecturePolicyConfig,
    ArchitecturePolicyError,
    ArchitecturePolicyKind,
    ArchitecturePolicyState,
    ArchitectureSearchPolicy,
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

OBJECTIVES = ObjectiveSet(
    (
        ObjectiveTerm(
            OptimizeMetric.QUALITY,
            ObjectiveDirection.MAXIMIZE,
            normalization=ObjectiveNormalization.IDENTITY,
        ),
        ObjectiveTerm(
            OptimizeMetric.MEMORY,
            ObjectiveDirection.MINIMIZE,
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
    name: str,
    query_heads: int,
    kv_heads: int,
    quality: float,
    ram: float,
    *,
    quality_bounds: tuple[float, float] | None = None,
    status: ArchitectureEvidenceStatus = ArchitectureEvidenceStatus.PREDICTED,
) -> CompleteArchitectureState:
    candidate = ArchitectureCandidate(
        "state_root",
        (
            (ArchitectureAxis.KV_HEADS, kv_heads),
            (ArchitectureAxis.QUERY_HEADS, query_heads),
        ),
        "cpu",
        query_heads * 100,
        query_heads * 10,
        float(query_heads),
    )
    # The name is deliberately retained in provenance so test records remain human-auditable;
    # the canonical architecture identity itself comes from the complete candidate state.
    return CompleteArchitectureState(
        candidate,
        (
            ParetoObjectiveValue(
                OptimizeMetric.QUALITY,
                quality,
                None if quality_bounds is None else quality_bounds[0],
                None if quality_bounds is None else quality_bounds[1],
            ),
            ParetoObjectiveValue(OptimizeMetric.MEMORY, ram),
        ),
        (
            ConstraintObservation(
                ConstraintMetric.QUALITY_RETENTION,
                1.0,
                BaselineReference.IMMUTABLE_SOURCE,
            ),
        ),
        status,
        provenance=(("fixture", name),),
    )


def test_pareto_beam_is_conservative_deterministic_and_retains_outcomes() -> None:
    states = (
        _state("quality", 8, 2, 0.95, 120),
        _state("memory", 4, 1, 0.90, 80),
        _state("dominated", 6, 2, 0.90, 140),
        _state("unsupported", 2, 1, 0.99, 50, status=ArchitectureEvidenceStatus.UNSUPPORTED),
    )
    policy = ArchitectureSearchPolicy(
        ArchitecturePolicyConfig(ArchitecturePolicyKind.PARETO_BEAM, 2, beam_width=2),
        OBJECTIVES,
        CONSTRAINTS,
    )
    first = policy.select(states)
    second = policy.select(states)
    assert first.to_record() == second.to_record()
    assert ArchitecturePolicyState.from_record(first.next_state.to_record()) == first.next_state
    assert first.decisions[0].to_record()["objectives"]
    assert {item.candidate_id for item in first.selected} == {
        states[0].candidate_id,
        states[1].candidate_id,
    }
    assert any(item.reason == "retained_unsupported_evidence" for item in first.decisions)
    assert first.next_state.frontier_candidate_ids == tuple(
        sorted((states[0].candidate_id, states[1].candidate_id))
    )


def test_uncertainty_bounds_prevent_unsafe_dominance() -> None:
    uncertain = _state("uncertain", 8, 2, 0.95, 100, quality_bounds=(0.80, 1.0))
    precise = _state("precise", 4, 1, 0.90, 110)
    policy = ArchitectureSearchPolicy(
        ArchitecturePolicyConfig(ArchitecturePolicyKind.PARETO_BEAM, 2, beam_width=2),
        OBJECTIVES,
        CONSTRAINTS,
    )
    selection = policy.select((uncertain, precise))
    assert {item.candidate_id for item in selection.selected} == {
        uncertain.candidate_id,
        precise.candidate_id,
    }


def test_evolutionary_policy_uses_legal_mutation_crossover_and_resume_state() -> None:
    states = (
        _state("parent-a", 8, 2, 0.96, 120),
        _state("parent-b", 4, 1, 0.90, 80),
        _state("child", 4, 2, 0.94, 90),
        _state("mutation", 8, 1, 0.93, 110),
    )
    policy = ArchitectureSearchPolicy(
        ArchitecturePolicyConfig(
            ArchitecturePolicyKind.EVOLUTIONARY,
            evaluation_budget=4,
            population_size=2,
            generations=2,
            elitism=1,
            seed=9,
        ),
        OBJECTIVES,
        CONSTRAINTS,
    )
    first = policy.select(states)
    assert len(first.selected) == 2
    assert first.next_state.population_candidate_ids
    resumed = policy.select(states, first.next_state)
    assert all(
        item.candidate_id not in first.next_state.selected_candidate_ids
        for item in resumed.selected
    )
    assert any(item.operation in {"mutation", "crossover"} for item in resumed.decisions)
    assert len(resumed.next_state.selected_candidate_ids) <= 4


def test_policy_rejects_foreign_or_over_budget_resume_state() -> None:
    state = _state("one", 8, 2, 0.95, 120)
    policy = ArchitectureSearchPolicy(
        ArchitecturePolicyConfig(ArchitecturePolicyKind.PARETO_BEAM, 1),
        OBJECTIVES,
        CONSTRAINTS,
    )
    with pytest.raises(ArchitecturePolicyError, match="another policy"):
        policy.select((state,), ArchitecturePolicyState("architecture_policy_foreign"))
    with pytest.raises(ArchitecturePolicyError, match="exceeds"):
        policy.select((state,), ArchitecturePolicyState(policy.policy_id, ("a", "b")))

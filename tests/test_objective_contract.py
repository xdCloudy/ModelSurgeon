"""Versioned objective-contract and fail-closed evaluation tests."""

from __future__ import annotations

import pytest

from modelsurgeon.config import Settings
from modelsurgeon.plugins import PluginCapabilityCard, PluginExecutionMode, PluginKind
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    ContractMetric,
    ContractOutcome,
    HardConstraint,
    MetricObservation,
    MetricUnit,
    ObjectiveApprovalPolicy,
    ObjectiveContract,
    ObjectiveContractError,
    ObjectiveDirection,
    ObjectiveMode,
    ObjectiveNormalization,
    ObjectivePluginBinding,
    SoftObjective,
    contract_from_settings,
    contracts_dominate,
    evaluate_contract,
)


def _contract(*, mode: ObjectiveMode = ObjectiveMode.WEIGHTED) -> ObjectiveContract:
    return ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                ConstraintDirection.MINIMUM,
                0.95,
                MetricUnit.RATIO,
            ),
        ),
        objectives=(
            SoftObjective(
                ContractMetric.QUALITY,
                ObjectiveDirection.MAXIMIZE,
                MetricUnit.RATIO,
                baseline=1.0,
            ),
            SoftObjective(
                ContractMetric.LATENCY,
                ObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                baseline=100.0,
            ),
        ),
        mode=mode,
    )


def _observations(*, quality: float = 0.98, latency: float = 80.0) -> tuple[MetricObservation, ...]:
    return (
        MetricObservation(ContractMetric.QUALITY, quality, MetricUnit.RATIO),
        MetricObservation(ContractMetric.LATENCY, latency, MetricUnit.MILLISECONDS),
    )


def test_contract_identity_is_stable_and_legacy_settings_translate() -> None:
    contract = contract_from_settings(Settings())
    assert contract.contract_id == contract.to_record()["contract_id"]
    assert {item.metric for item in contract.objectives} == {
        ContractMetric.LATENCY,
        ContractMetric.PARAMETER_COUNT,
    }


def test_missing_hard_evidence_fails_closed_before_scoring() -> None:
    evaluation = evaluate_contract(_contract(), ())
    assert evaluation.outcome is ContractOutcome.INFEASIBLE
    assert not evaluation.feasible
    assert evaluation.objective_values is None
    assert evaluation.constraint_results[0].reason == "missing_observation"


def test_missing_soft_evidence_is_unknown_and_incomparable() -> None:
    evaluation = evaluate_contract(
        _contract(), (MetricObservation(ContractMetric.QUALITY, 0.98, MetricUnit.RATIO),)
    )
    assert evaluation.outcome is ContractOutcome.UNKNOWN
    assert not evaluation.comparable


def test_uncertainty_uses_conservative_bounds_for_constraints_and_objectives() -> None:
    observations = (
        MetricObservation(
            ContractMetric.QUALITY,
            0.98,
            MetricUnit.RATIO,
            lower=0.94,
        ),
        MetricObservation(
            ContractMetric.LATENCY,
            80.0,
            MetricUnit.MILLISECONDS,
            upper=110.0,
        ),
    )
    evaluation = evaluate_contract(_contract(), observations)
    assert evaluation.outcome is ContractOutcome.INFEASIBLE
    assert evaluation.constraint_results[0].observed == 0.94


def test_pareto_dominance_rejects_infeasible_or_incomparable_points() -> None:
    better = evaluate_contract(_contract(mode=ObjectiveMode.PARETO), _observations())
    best = evaluate_contract(
        _contract(mode=ObjectiveMode.PARETO), _observations(quality=0.99, latency=70.0)
    )
    incomplete = evaluate_contract(
        _contract(mode=ObjectiveMode.PARETO),
        (MetricObservation(ContractMetric.QUALITY, 0.99, MetricUnit.RATIO),),
    )
    infeasible = evaluate_contract(
        _contract(mode=ObjectiveMode.PARETO), _observations(quality=0.90)
    )
    assert contracts_dominate(best, better)
    assert not contracts_dominate(best, incomplete)
    assert not contracts_dominate(best, infeasible)


def test_custom_objective_requires_approved_capability_card() -> None:
    card = PluginCapabilityCard(
        name="energy-meter",
        kind=PluginKind.OBJECTIVE,
        plugin_version="1.0.0",
        api_version="1.0",
        capabilities=("energy",),
        license_id="apache-2.0",
        trust_modes=(PluginExecutionMode.SUBPROCESS,),
    )
    objective = SoftObjective(
        "energy_per_token",
        ObjectiveDirection.MINIMIZE,
        MetricUnit.JOULES,
        normalization=ObjectiveNormalization.IDENTITY,
        plugin=ObjectivePluginBinding(
            name=card.name,
            plugin_id=card.plugin_id,
            capability="energy",
            config_digest="sha256:" + "1" * 64,
        ),
    )
    contract = ObjectiveContract(
        constraints=_contract().constraints,
        objectives=(objective,),
        approval_policy=ObjectiveApprovalPolicy(allowed_plugin_names=(card.name,)),
    )
    observation = MetricObservation("energy_per_token", 4.0, MetricUnit.JOULES)
    quality_observation = MetricObservation(ContractMetric.QUALITY, 0.98, MetricUnit.RATIO)
    unsupported = evaluate_contract(contract, (quality_observation, observation))
    supported = evaluate_contract(
        contract,
        (MetricObservation(ContractMetric.QUALITY, 0.98, MetricUnit.RATIO), observation),
        plugin_cards={card.name: card},
    )
    assert unsupported.outcome is ContractOutcome.UNSUPPORTED
    assert supported.outcome is ContractOutcome.FEASIBLE


def test_contract_rejects_wrong_units_and_duplicate_terms() -> None:
    with pytest.raises(ObjectiveContractError, match="requires unit"):
        HardConstraint(ContractMetric.QUALITY, ConstraintDirection.MINIMUM, 0.9, MetricUnit.BYTES)
    with pytest.raises(ObjectiveContractError, match="unique"):
        ObjectiveContract(
            constraints=_contract().constraints,
            objectives=(_contract().objectives[0], _contract().objectives[0]),
        )

"""Deterministic strategy selection and refusal tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.search.strategy_selector import (
    EvidenceLevel,
    StrategyOption,
    StrategyOutcome,
    StrategySelectionRequest,
    StrategySelectorError,
    select_strategy,
)


def _option(
    name: str,
    kind: str,
    *,
    cost: float = 1.0,
    uncertainty: float = 0.1,
    strong_baseline: bool = False,
    compatible: bool = True,
) -> StrategyOption:
    return StrategyOption(
        name,
        kind,
        compatible,
        EvidenceLevel.VERIFIED,
        cost,
        uncertainty,
        strong_baseline,
        reason=f"fixture evidence for {name}",
    )


def _request(**overrides: object) -> StrategySelectionRequest:
    values: dict[str, object] = {
        "objective_contract_id": "objective_contract_1",
        "candidate_space_id": "space_1",
        "tool_revision": "tool-1",
        "seed": 7,
        "candidate_count": 12,
        "evaluation_budget": 8,
        "cost_budget": 20.0,
        "required_baselines": ("dense-baseline",),
        "options": (
            _option("dense-baseline", "baseline", cost=2.0, strong_baseline=True),
            _option("cold-surgeon", "surgeon", cost=3.0),
            _option("pretrained-surgeon", "surgeon", cost=2.0, uncertainty=0.4),
            _option("uncertainty", "acquisition", cost=1.0),
            _option("pareto-beam", "search_policy", cost=2.0),
        ),
    }
    values.update(overrides)
    return StrategySelectionRequest(**values)  # type: ignore[arg-type]


def test_rule_based_selection_is_deterministic_and_keeps_required_baseline() -> None:
    first = select_strategy(_request())
    second = select_strategy(_request())
    assert first.outcome is StrategyOutcome.SUPPORTED
    assert first.decision_id == second.decision_id
    assert first.baseline_names == ("dense-baseline",)
    assert first.surgeon == "pretrained-surgeon"
    assert first.acquisition == "uncertainty"
    assert first.search_policy == "pareto-beam"
    assert first.evaluation_budget == 8
    assert first.to_record()["alternatives"]


def test_incompatible_required_baseline_refuses_instead_of_substituting() -> None:
    request = _request(
        options=tuple(
            _option("dense-baseline", "baseline", strong_baseline=True, compatible=False)
            for _ in (0,)
        )
        + _request().options[1:]
    )
    decision = select_strategy(request)
    assert decision.outcome is StrategyOutcome.UNSUPPORTED
    assert "incompatible" in decision.reasons[0]
    assert decision.selected == ()


def test_override_requires_approval_and_must_remain_compatible() -> None:
    unapproved = select_strategy(_request(override="pretrained-surgeon"))
    assert unapproved.outcome is StrategyOutcome.FAILED
    approved = select_strategy(_request(override="pretrained-surgeon", override_approved=True))
    assert approved.outcome is StrategyOutcome.SUPPORTED
    assert approved.surgeon == "pretrained-surgeon"


def test_missing_strategy_evidence_and_budget_are_explicit() -> None:
    missing = select_strategy(
        _request(
            options=tuple(option for option in _request().options if option.kind != "acquisition")
        )
    )
    assert missing.outcome is StrategyOutcome.UNKNOWN
    assert "acquisition" in missing.reasons[0]

    over_budget = select_strategy(_request(cost_budget=1.0))
    assert over_budget.outcome is StrategyOutcome.FAILED
    assert "cost budget" in over_budget.reasons[0]


def test_request_rejects_duplicate_options() -> None:
    with pytest.raises(StrategySelectorError, match="unique"):
        replace(_request(), options=(*_request().options, _request().options[0]))

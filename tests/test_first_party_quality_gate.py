"""Unit coverage for the first-party runtime's declared quality contract."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from modelsurgeon.first_party_optimize_runtime import (
    FirstPartyOptimizeRuntimeError,
    HuggingFaceOptimizeRuntime,
    _perplexity_quality_gate,
)


def _gate(
    candidate: float,
    *,
    minimum: float = 0.99,
    maximum_delta: float | None = None,
    profile_delta: float | None = None,
) -> dict[str, object]:
    return _perplexity_quality_gate(
        100.0,
        candidate,
        min_quality_retention_ratio=minimum,
        max_perplexity_delta=maximum_delta,
        profile_max_perplexity_delta=profile_delta,
    )


def test_quality_retention_ratio_is_a_hard_lower_is_better_gate() -> None:
    accepted = _gate(101.0)
    rejected = _gate(101.1)

    assert accepted["quality_retention_ratio"] == pytest.approx(100.0 / 101.0)
    assert accepted["retention_constraint_passed"] is True
    assert accepted["accepted"] is True
    assert rejected["quality_retention_ratio"] == pytest.approx(100.0 / 101.1)
    assert rejected["retention_constraint_passed"] is False
    assert rejected["accepted"] is False
    assert "minimum quality-retention ratio violated" in rejected["rejection_reasons"]


def test_explicit_delta_and_profile_guard_are_retained_separately() -> None:
    rejected = _gate(100.6, maximum_delta=0.5, profile_delta=1.0)
    profile_rejected = _gate(100.6, profile_delta=0.2)

    assert rejected["max_delta_constraint_passed"] is False
    assert rejected["quality_profile_guard_passed"] is True
    assert rejected["accepted"] is False
    assert profile_rejected["max_delta_constraint_passed"] is True
    assert profile_rejected["quality_profile_guard_passed"] is False
    assert profile_rejected["accepted"] is False


def test_runtime_reads_hard_constraints_from_the_resolved_plan() -> None:
    runtime = object.__new__(HuggingFaceOptimizeRuntime)
    runtime.plan = SimpleNamespace(
        resolved_config={
            "constraints": {
                "min_quality_retention_ratio": 0.99,
                "max_perplexity_delta": 0.5,
            }
        },
        quality_profile=SimpleNamespace(max_perplexity_delta=10.0),
    )

    gate = runtime._quality_gate(100.0, 100.6)

    assert gate["retention_constraint_passed"] is True
    assert gate["max_delta_constraint_passed"] is False
    assert gate["quality_profile_guard_passed"] is True
    assert gate["accepted"] is False


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    (
        (0.0, 10.0, "perplexity must be positive"),
        (10.0, math.inf, "candidate_perplexity must be finite"),
    ),
)
def test_quality_gate_fails_closed_for_invalid_measurements(
    baseline: float, candidate: float, message: str
) -> None:
    with pytest.raises(FirstPartyOptimizeRuntimeError, match=message):
        _perplexity_quality_gate(
            baseline,
            candidate,
            min_quality_retention_ratio=0.99,
            max_perplexity_delta=None,
            profile_max_perplexity_delta=None,
        )

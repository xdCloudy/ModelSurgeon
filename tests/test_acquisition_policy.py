from __future__ import annotations

from modelsurgeon.active_learning import (
    AcquisitionCandidate,
    AcquisitionPolicyConfig,
    acquire_candidates,
)


def _candidates(count: int):
    return tuple(
        AcquisitionCandidate(
            f"cand_{index}",
            utility=float(count - index),
            safe_probability=0.5 + index / (2 * max(1, count)),
            uncertainty=float(index),
            diversity=float(index % 3),
        )
        for index in range(count)
    )


def test_explicit_fractions_reasons_and_propensities_are_recorded() -> None:
    report = acquire_candidates(
        _candidates(20),
        10,
        config=AcquisitionPolicyConfig(
            high_value_fraction=0.5,
            uncertain_fraction=0.3,
            diverse_fraction=0.2,
            seed=42,
        ),
    )

    assert sorted(dict(report.quotas).values()) == [2, 3, 5]
    assert len(report.selections) == 10
    assert len({item.candidate_id for item in report.selections}) == 10
    assert all(item.reason and item.propensity == 1.0 for item in report.selections)


def test_zero_and_oversubscribed_budgets_are_deterministic() -> None:
    candidates = _candidates(4)
    zero = acquire_candidates(candidates, 0)
    first = acquire_candidates(candidates, 100)
    second = acquire_candidates(candidates, 100)

    assert zero.selections == () and zero.effective_budget == 0
    assert first == second
    assert first.effective_budget == 4
    assert {item.candidate_id for item in first.selections} == {
        item.candidate_id for item in candidates
    }
    assert first.to_record()["oversubscribed"] is True

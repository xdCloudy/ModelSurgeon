from __future__ import annotations

from typing import cast

import pytest

from modelsurgeon.active_learning import (
    AcquisitionCandidate,
    AcquisitionPolicyConfig,
    AcquisitionPolicyError,
    DiversityCandidate,
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


@pytest.mark.parametrize("invalid_seed", [1.5, True, "42"])
def test_policy_rejects_non_integer_seeds(invalid_seed: object) -> None:
    with pytest.raises(AcquisitionPolicyError, match="seed must be unsigned 64-bit integer"):
        AcquisitionPolicyConfig(seed=cast(int, invalid_seed))


@pytest.mark.parametrize("invalid_budget", [1.5, True, "1"])
def test_acquisition_rejects_non_integer_budgets(invalid_budget: object) -> None:
    with pytest.raises(AcquisitionPolicyError, match="budget must be an integer"):
        acquire_candidates(_candidates(3), cast(int, invalid_budget))


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
    assert report.diversity_selection is None
    assert report.to_record()["diversity_selection"] is None


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


def test_feature_diversity_selects_from_remaining_pool_and_records_provenance() -> None:
    candidates = _candidates(6)
    candidates = tuple(
        AcquisitionCandidate(
            candidate.candidate_id,
            utility=100.0 - index if index < 2 else 1.0,
            safe_probability=1.0,
            uncertainty=100.0 if index == 2 else 0.0,
            diversity=float(6 - index),
        )
        for index, candidate in enumerate(candidates)
    )
    features = tuple(
        DiversityCandidate(
            candidate.candidate_id,
            (float(index - 3),),
            ("shared",),
            frozenset({"layer:0"}),
        )
        for index, candidate in enumerate(candidates)
    )
    config = AcquisitionPolicyConfig(
        high_value_fraction=0.34,
        uncertain_fraction=0.16,
        diverse_fraction=0.5,
        seed=43,
    )

    report = acquire_candidates(
        candidates,
        6,
        config=config,
        diversity_candidates=features,
    )
    repeated = acquire_candidates(
        candidates,
        6,
        config=config,
        diversity_candidates=features,
    )

    assert report == repeated
    assert [item.candidate_id for item in report.selections[:2]] == ["cand_0", "cand_1"]
    assert report.selections[2].candidate_id == "cand_2"
    assert all(item.reason.value == "diverse" for item in report.selections[3:])
    assert report.selections[3].candidate_id == "cand_5"
    assert len({item.candidate_id for item in report.selections}) == 6
    recorded_diversity = report.to_record()["diversity_selection"]
    assert report.to_record()["schema_version"] == 2
    assert isinstance(recorded_diversity, dict)
    assert recorded_diversity["candidate_count"] == 3
    assert recorded_diversity["initial_center_count"] == 3
    assert [item["candidate_id"] for item in recorded_diversity["selections"]] == [
        item.candidate_id for item in report.selections[3:]
    ]
    assert recorded_diversity["numeric_normalization"] == "candidate_pool_minmax_v1"


def test_feature_diversity_rejects_partial_candidate_alignment() -> None:
    candidates = _candidates(3)
    features = tuple(
        DiversityCandidate(candidate.candidate_id, (float(index),), (), frozenset())
        for index, candidate in enumerate(candidates[:2])
    )

    with pytest.raises(AcquisitionPolicyError, match="exactly match"):
        acquire_candidates(candidates, 2, diversity_candidates=features)


def test_feature_diversity_uses_prior_labeled_items_as_fixed_centers() -> None:
    candidates = (
        AcquisitionCandidate("cand_near", 0.0, 0.5, 0.0, 0.0),
        AcquisitionCandidate("cand_far", 0.0, 0.5, 0.0, 0.0),
    )
    features = (
        DiversityCandidate("cand_near", (1.0,), (), frozenset()),
        DiversityCandidate("cand_far", (10.0,), (), frozenset()),
    )
    initial_center = DiversityCandidate("cand_labeled", (0.0,), (), frozenset())

    report = acquire_candidates(
        candidates,
        1,
        config=AcquisitionPolicyConfig(
            high_value_fraction=0.0,
            uncertain_fraction=0.0,
            diverse_fraction=1.0,
            seed=2,
        ),
        diversity_candidates=features,
        diversity_initial_centers=(initial_center,),
    )

    assert report.selections[0].candidate_id == "cand_far"
    assert report.diversity_selection is not None
    assert report.diversity_selection.initial_center_count == 1


def test_feature_diversity_rejects_initial_center_candidate_id_collision() -> None:
    candidate = _candidates(1)[0]
    features = (DiversityCandidate(candidate.candidate_id, (0.0,), (), frozenset()),)

    with pytest.raises(AcquisitionPolicyError, match="outside the candidate pool"):
        acquire_candidates(
            (candidate,),
            1,
            diversity_candidates=features,
            diversity_initial_centers=features,
        )


def test_zero_uncertainty_quota_does_not_consume_the_diversity_pool() -> None:
    candidates = _candidates(6)
    features = tuple(
        DiversityCandidate(candidate.candidate_id, (float(index),), (), frozenset())
        for index, candidate in enumerate(candidates)
    )
    report = acquire_candidates(
        candidates,
        4,
        config=AcquisitionPolicyConfig(
            high_value_fraction=0.5,
            uncertain_fraction=0.0,
            diverse_fraction=0.5,
            seed=11,
        ),
        diversity_candidates=features,
    )

    assert len(report.selections) == 4
    assert [item.reason.value for item in report.selections] == [
        "high-value",
        "high-value",
        "diverse",
        "diverse",
    ]

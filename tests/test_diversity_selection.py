from __future__ import annotations

from typing import cast

import pytest

from modelsurgeon.active_learning import (
    DiversityCandidate,
    DiversitySelectionConfig,
    DiversitySelectionError,
    select_diverse_candidates,
)


def _candidate(identity: str, numeric: float, category: str, *topology: str):
    return DiversityCandidate(f"cand_{identity}", (numeric,), (category,), frozenset(topology))


def test_selection_is_deterministic_under_seed_and_memory_is_linear() -> None:
    candidates = tuple(
        _candidate(str(index), index / 10.0, str(index % 2), f"layer:{index % 3}")
        for index in range(20)
    )
    config = DiversitySelectionConfig(seed=42)

    first = select_diverse_candidates(candidates, 8, config=config)
    second = select_diverse_candidates(candidates, 8, config=config)

    assert first == second
    assert len({item.candidate_id for item in first.selections}) == 8
    assert first.working_distance_values == len(candidates)


def test_categorical_and_topology_space_change_farthest_selection() -> None:
    candidates = (
        _candidate("anchor", 0.0, "a", "layer:0", "scope:head"),
        _candidate("numeric", 0.1, "a", "layer:0", "scope:head"),
        _candidate("different", 0.0, "b", "layer:9", "scope:channel"),
    )
    config = DiversitySelectionConfig(
        seed=5, numeric_weight=0.0, categorical_weight=1.0, topology_weight=1.0
    )

    report = select_diverse_candidates(candidates, 3, config=config)

    assert {item.candidate_id for item in report.selections} == {
        "cand_anchor",
        "cand_numeric",
        "cand_different",
    }
    assert report.selections[1].minimum_distance is not None


def test_numeric_diversity_is_invariant_to_feature_units_and_scale_safe() -> None:
    vectors = (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.5, 0.25),
        (0.25, 0.75),
    )
    topology = frozenset({"layer:0"})
    candidates = tuple(
        DiversityCandidate(f"cand_{index}", vector, ("same",), topology)
        for index, vector in enumerate(vectors)
    )
    rescaled = tuple(
        DiversityCandidate(
            candidate.candidate_id,
            (
                candidate.numeric_features[0] * 1e290 - 3e289,
                candidate.numeric_features[1] * 1e300 + 2e300,
            ),
            candidate.categorical_features,
            candidate.topology,
        )
        for candidate in candidates
    )
    config = DiversitySelectionConfig(seed=13)

    original_report = select_diverse_candidates(candidates, 5, config=config)
    rescaled_report = select_diverse_candidates(rescaled, 5, config=config)

    assert tuple(item.candidate_id for item in original_report.selections) == tuple(
        item.candidate_id for item in rescaled_report.selections
    )
    for original, changed in zip(
        original_report.selections, rescaled_report.selections, strict=True
    ):
        assert original.minimum_distance == pytest.approx(changed.minimum_distance)
        if changed.minimum_distance is not None:
            assert 0.0 <= changed.minimum_distance <= 1.0
    assert rescaled_report.to_record()["numeric_normalization"] == "candidate_pool_minmax_v1"


@pytest.mark.parametrize("invalid_value", [True, "1.0", 10**400])
def test_numeric_diversity_rejects_invalid_values(invalid_value: object) -> None:
    with pytest.raises(DiversitySelectionError, match="finite numbers"):
        DiversityCandidate(
            "cand_invalid",
            (cast(float, invalid_value),),
            (),
            frozenset(),
        )


def test_large_finite_weights_do_not_overflow_distance() -> None:
    candidates = (
        _candidate("a", 0.0, "a", "layer:0"),
        _candidate("b", 1.0, "b", "layer:1"),
        _candidate("c", 0.5, "c", "layer:2"),
    )
    baseline = select_diverse_candidates(candidates, 3, config=DiversitySelectionConfig(seed=29))
    large_weights = select_diverse_candidates(
        candidates,
        3,
        config=DiversitySelectionConfig(
            seed=29,
            numeric_weight=1e308,
            categorical_weight=1e308,
            topology_weight=1e308,
        ),
    )

    assert tuple(item.candidate_id for item in baseline.selections) == tuple(
        item.candidate_id for item in large_weights.selections
    )
    assert large_weights.selections[1].minimum_distance == pytest.approx(
        baseline.selections[1].minimum_distance
    )


def test_existing_selected_candidates_are_fixed_farthest_first_centers() -> None:
    center = DiversityCandidate("cand_center", (0.0,), (), frozenset({"layer:0"}))
    candidates = (
        DiversityCandidate("cand_near", (0.1,), (), frozenset({"layer:0"})),
        DiversityCandidate("cand_mid", (5.0,), (), frozenset({"layer:1"})),
        DiversityCandidate("cand_far", (10.0,), (), frozenset({"layer:2"})),
    )

    report = select_diverse_candidates(
        candidates,
        2,
        config=DiversitySelectionConfig(seed=19),
        initial_centers=(center,),
    )

    assert report.selections[0].candidate_id == "cand_far"
    assert report.selections[0].minimum_distance is not None
    assert report.selections[1].candidate_id == "cand_mid"
    assert report.initial_center_count == 1
    assert report.to_record()["initial_center_count"] == 1


@pytest.mark.parametrize("invalid_seed", [1.5, True, "1"])
def test_config_rejects_non_integer_seeds(invalid_seed: object) -> None:
    with pytest.raises(DiversitySelectionError, match="seed must be unsigned 64-bit integer"):
        DiversitySelectionConfig(seed=cast(int, invalid_seed))


@pytest.mark.parametrize("invalid_bound", [1.5, True, "1"])
def test_config_rejects_non_integer_bounds(invalid_bound: object) -> None:
    with pytest.raises(DiversitySelectionError, match="candidate bound must be an integer"):
        DiversitySelectionConfig(seed=0, max_candidates=cast(int, invalid_bound))
    with pytest.raises(DiversitySelectionError, match="selection bound must be an integer"):
        DiversitySelectionConfig(seed=0, max_selection=cast(int, invalid_bound))


@pytest.mark.parametrize("invalid_count", [1.5, True, "1"])
def test_selection_rejects_non_integer_counts(invalid_count: object) -> None:
    with pytest.raises(DiversitySelectionError, match="selection must be an integer"):
        select_diverse_candidates(
            (_candidate("only", 0.0, "a"),),
            cast(int, invalid_count),
            config=DiversitySelectionConfig(seed=0),
        )


def test_zero_selection_is_deterministic() -> None:
    report = select_diverse_candidates(
        (_candidate("a", 0.0, "a", "layer:0"),),
        0,
        config=DiversitySelectionConfig(seed=0),
    )
    assert report.selections == ()
    assert report.working_distance_values == 1

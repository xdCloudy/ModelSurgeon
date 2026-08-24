from __future__ import annotations

from modelsurgeon.active_learning import (
    DiversityCandidate,
    DiversitySelectionConfig,
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


def test_zero_selection_is_deterministic() -> None:
    report = select_diverse_candidates(
        (_candidate("a", 0.0, "a", "layer:0"),),
        0,
        config=DiversitySelectionConfig(seed=0),
    )
    assert report.selections == ()
    assert report.working_distance_values == 1

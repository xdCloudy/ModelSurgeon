"""Real LightGBM smoke for fixed-budget tree uncertainty producers."""

from __future__ import annotations

import pytest

from modelsurgeon.active_learning import (
    LightGBMTreeUncertaintyConfig,
    TreeUncertaintyBudget,
    TreeUncertaintyMethod,
    run_lightgbm_tree_uncertainty_study,
)
from modelsurgeon.datasets.grouped_splits import SplitPartition
from modelsurgeon.surgeon.matrix import SurgeonMatrix

pytest.importorskip("lightgbm")


def _matrix(partition: SplitPartition, count: int) -> SurgeonMatrix:
    rows = tuple((index / count, (index % 3) / 3.0) for index in range(count))
    targets = tuple(2.0 * left - right for left, right in rows)
    return SurgeonMatrix(
        partition,
        tuple(f"{partition.value}-{index}" for index in range(count)),
        ("x", "y"),
        rows,
        "perplexity",
        targets,
        (True,) * count,
        (1.0,) * count,
        tuple(f"group-{partition.value}-{index}" for index in range(count)),
    )


def test_real_lightgbm_uncertainty_study_is_reproducible_and_complete() -> None:
    config = LightGBMTreeUncertaintyConfig(
        confidence=0.8,
        members=3,
        max_rounds=20,
        early_stopping_rounds=5,
        seed=7,
        budget=TreeUncertaintyBudget(
            max_fits_per_method=3,
            max_cpu_seconds_per_method=60.0,
            num_threads=2,
        ),
    )

    first = run_lightgbm_tree_uncertainty_study(
        _matrix(SplitPartition.TRAIN, 30),
        _matrix(SplitPartition.VALIDATION, 10),
        config=config,
    )
    second = run_lightgbm_tree_uncertainty_study(
        _matrix(SplitPartition.TRAIN, 30),
        _matrix(SplitPartition.VALIDATION, 10),
        config=config,
    )

    assert {item.method for item in first.candidates} == set(TreeUncertaintyMethod)
    assert [item.predictions for item in first.candidates] == [
        item.predictions for item in second.candidates
    ]
    assert all(item.cpu_seconds >= 0.0 and item.model_bytes > 0 for item in first.candidates)

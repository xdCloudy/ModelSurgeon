"""Tests for the v0.8 Q1 static-only study contract."""

from __future__ import annotations

import pytest

from modelsurgeon.datasets.grouped_splits import (
    GroupedSplitManifest,
    GroupedSplitMode,
    SplitGroup,
    SplitPartition,
    SplitRatios,
)
from modelsurgeon.evaluation.activation_feature_study import (
    run_activation_feature_ablation,
)
from modelsurgeon.evaluation.gradient_feature_study import (
    run_gradient_feature_ablation,
)
from modelsurgeon.evaluation.pruning_baseline_study import (
    PruningBaselineStudyConfig,
    run_pruning_baseline_study,
)
from modelsurgeon.evaluation.static_feature_study import (
    StaticFeatureStudyConfig,
    StaticFeatureStudyError,
    run_static_feature_study,
    select_static_records,
)


def _metric(value: float) -> dict[str, object]:
    return {
        "name": "perplexity",
        "state": "measured",
        "value": value,
        "unit": "perplexity",
        "reason": None,
    }


def _example(index: int, delta: float) -> dict[str, object]:
    return {
        "example_id": f"example-{index}",
        "model": {
            "identifier": "synthetic/tiny",
            "revision": "revision",
            "family": "llama",
            "format": "safetensors",
            "quantization": None,
        },
        "mutation": {
            "plan": {
                "request": {
                    "kind": "mask",
                    "parameters": {"candidate_scope": "mlp_channel"},
                }
            }
        },
        "pre_mutation_features": [
            {
                "name": "activation_rms",
                "kind": "scalar",
                "value": delta * 2.0,
                "sample_context": {"sample_ids": ["calibration"]},
            },
            {
                "name": "weight_l2_norm",
                "kind": "scalar",
                "value": delta,
                "sample_context": None,
            },
        ],
        "baseline_metrics": [_metric(10.0)],
        "post_metrics": [_metric(10.0 + delta)],
        "versions": {"feature_schema_version": 1},
    }


def _gradient_example(index: int, delta: float) -> dict[str, object]:
    record = _example(index, delta)
    features = record["pre_mutation_features"]
    assert isinstance(features, list)
    features.append(
        {
            "name": "first_order_removal_magnitude",
            "kind": "scalar",
            "value": delta * 3.0,
            "extractor": "gradient_features",
            "sample_context": {"sample_ids": ["calibration"]},
        }
    )
    features.extend(
        (
            {
                "name": "weight_count",
                "kind": "scalar",
                "value": 8.0,
                "sample_context": None,
            },
            {
                "name": "weight_l1_norm",
                "kind": "scalar",
                "value": delta * 8.0,
                "sample_context": None,
            },
        )
    )
    return record


def _split() -> GroupedSplitManifest:
    partitions = (
        *(SplitPartition.TRAIN for _ in range(8)),
        *(SplitPartition.VALIDATION for _ in range(4)),
        *(SplitPartition.TEST for _ in range(4)),
    )
    return GroupedSplitManifest(
        GroupedSplitMode.COMPONENT,
        43,
        SplitRatios(0.5, 0.25, 0.25),
        tuple(
            sorted(
                (
                    SplitGroup(
                        f"group-{index}",
                        partition,
                        (f"component:{index}",),
                        (f"example-{index}",),
                    )
                    for index, partition in enumerate(partitions)
                ),
                key=lambda group: group.group_id,
            )
        ),
    )


def test_static_selector_excludes_calibration_dependent_features() -> None:
    selected = select_static_records((_example(0, 0.1),))

    features = selected[0]["pre_mutation_features"]
    assert isinstance(features, list)
    assert [feature["name"] for feature in features] == ["weight_l2_norm"]


def test_static_study_trains_both_tasks_and_reports_intervals() -> None:
    pytest.importorskip("lightgbm")
    deltas = (0.05, 0.75, 0.10, 0.90) * 4
    result = run_static_feature_study(
        tuple(_example(index, delta) for index, delta in enumerate(deltas)),
        _split(),
        StaticFeatureStudyConfig(
            top_n=2,
            threads=1,
            bootstrap_repetitions=20,
        ),
    )

    assert result.feature_names == ("weight_l2_norm",)
    assert result.classifier.metric("auc").value is not None
    assert result.classifier.metric("precision_at_2").confidence_low is not None
    assert result.regressor.metric("mae").value is not None
    assert result.regressor.metric("rmse").confidence_high is not None


def test_activation_ablation_is_paired_on_identical_heldout_rows() -> None:
    pytest.importorskip("lightgbm")
    deltas = (0.05, 0.75, 0.10, 0.90) * 4
    result = run_activation_feature_ablation(
        tuple(_example(index, delta) for index, delta in enumerate(deltas)),
        _split(),
        StaticFeatureStudyConfig(
            top_n=2,
            threads=1,
            bootstrap_repetitions=20,
        ),
    )

    assert result.static.test_group_ids == result.static_activation.test_group_ids
    assert result.static_activation.feature_names == (
        "activation_rms",
        "weight_l2_norm",
    )
    assert {gain.name for gain in result.gains} == {
        "auc_gain",
        "mae_reduction",
        "precision_at_2_gain",
        "rmse_reduction",
    }
    assert all(gain.bootstrap_repetitions > 0 for gain in result.gains)


def test_gradient_ablation_excludes_gradients_from_its_paired_baseline() -> None:
    pytest.importorskip("lightgbm")
    deltas = (0.05, 0.75, 0.10, 0.90) * 4
    result = run_gradient_feature_ablation(
        tuple(_gradient_example(index, delta) for index, delta in enumerate(deltas)),
        _split(),
        StaticFeatureStudyConfig(
            top_n=2,
            threads=1,
            bootstrap_repetitions=20,
        ),
    )

    assert "first_order_removal_magnitude" not in result.static_activation.feature_names
    assert "first_order_removal_magnitude" in (result.static_activation_gradient.feature_names)
    assert result.static_activation.test_labels == (result.static_activation_gradient.test_labels)


def test_q4_baselines_use_equal_budgets_and_paired_seeds() -> None:
    pytest.importorskip("lightgbm")
    deltas = (0.05, 0.75, 0.10, 0.90) * 4
    result = run_pruning_baseline_study(
        tuple(_gradient_example(index, delta) for index, delta in enumerate(deltas)),
        _split(),
        PruningBaselineStudyConfig(
            selection_budget=2,
            seeds=(3, 5, 7),
            bootstrap_repetitions=20,
            threads=1,
            safe_perplexity_delta=0.25,
        ),
    )

    assert result.pool_size == 4
    assert result.selection_budget == 2
    assert result.parameters_per_channel == 8
    assert {method.method for method in result.methods} == {
        "learned_gradient_lightgbm",
        "magnitude_mean_absolute",
        "seeded_random",
    }
    assert all(
        len(selection.selected_example_ids) == 2
        for method in result.methods
        for selection in method.selections
    )
    assert all(len(method.selections) == 3 for method in result.methods)
    assert {gain.name for gain in result.paired_gains} == {
        "constraint_violation_reduction_vs_magnitude",
        "constraint_violation_reduction_vs_random",
        "mean_perplexity_delta_reduction_vs_magnitude",
        "mean_perplexity_delta_reduction_vs_random",
    }


def test_static_study_rejects_invalid_protocol() -> None:
    with pytest.raises(StaticFeatureStudyError, match="bootstrap_repetitions"):
        StaticFeatureStudyConfig(bootstrap_repetitions=0)

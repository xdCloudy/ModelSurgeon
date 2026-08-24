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
from modelsurgeon.evaluation.cross_family_transfer import (
    TransferExperiment,
    TransferProtocol,
    run_cross_family_transfer_study,
)
from modelsurgeon.evaluation.cross_model_transfer import (
    TransferDataset,
    run_cross_model_transfer,
)
from modelsurgeon.evaluation.gradient_feature_study import (
    run_gradient_feature_ablation,
)
from modelsurgeon.evaluation.matched_pruning_study import run_matched_pruning_selection
from modelsurgeon.evaluation.multi_model_active_learning import (
    MultiModelActiveLearningConfig,
    run_model_active_learning_study,
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
        "delta_metrics": [_metric(delta)],
        "versions": {"feature_schema_version": 1},
    }


def _gradient_example(index: int, delta: float) -> dict[str, object]:
    record = _example(index, delta)
    record["timings"] = [{"stage": "evaluate", "wall_seconds": 0.1 + index / 1000.0}]
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


def _transfer_dataset(prefix: str, identifier: str, family: str) -> TransferDataset:
    deltas = (0.05, 0.75, 0.10, 0.90) * 4
    records: list[dict[str, object]] = []
    groups: list[SplitGroup] = []
    for index, delta in enumerate(deltas):
        record = _gradient_example(index, delta)
        example_id = f"{prefix}-example-{index}"
        record["example_id"] = example_id
        model = record["model"]
        assert isinstance(model, dict)
        model["identifier"] = identifier
        model["revision"] = f"{prefix}-revision"
        model["family"] = family
        mutation = record["mutation"]
        assert isinstance(mutation, dict)
        plan = mutation["plan"]
        assert isinstance(plan, dict)
        request = plan["request"]
        assert isinstance(request, dict)
        parameters = request["parameters"]
        assert isinstance(parameters, dict)
        parameters["layer_index"] = index // 4
        parameters["channel_index"] = index
        records.append(record)
        partition = (
            SplitPartition.TRAIN
            if index < 8
            else SplitPartition.VALIDATION
            if index < 12
            else SplitPartition.TEST
        )
        groups.append(
            SplitGroup(
                f"{prefix}-group-{index:02d}",
                partition,
                (f"component:{prefix}:{index}",),
                (example_id,),
            )
        )
    return TransferDataset(
        tuple(records),
        GroupedSplitManifest(
            GroupedSplitMode.COMPONENT,
            43,
            SplitRatios(0.5, 0.25, 0.25),
            tuple(groups),
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


def test_q5_transfer_fits_only_source_preprocessing() -> None:
    pytest.importorskip("lightgbm")
    result = run_cross_model_transfer(
        (
            _transfer_dataset("source-a", "synthetic/source-a", "llama"),
            _transfer_dataset("source-b", "synthetic/source-b", "qwen"),
        ),
        _transfer_dataset("target", "synthetic/unseen-target", "llama"),
        StaticFeatureStudyConfig(
            top_n=2,
            threads=1,
            bootstrap_repetitions=20,
        ),
    )

    assert result.source_train_count == 16
    assert result.source_validation_count == 8
    assert result.target_test_count == 4
    assert result.target_model[0] == "synthetic/unseen-target"
    assert result.target_model not in result.source_models
    assert "synthetic/unseen-target" not in str(result.source_preprocessor)
    assert len(result.source_preprocessor_sha256) == 64
    assert {item.name for item in result.degradations} == {
        "auc_degradation",
        "mae_increase",
        "precision_at_2_degradation",
        "rmse_increase",
    }


def test_q5_transfer_rejects_target_leakage_and_unrepresented_family() -> None:
    source = _transfer_dataset("source", "synthetic/source", "llama")
    with pytest.raises(StaticFeatureStudyError, match="completely unseen"):
        run_cross_model_transfer((source,), source)

    target = _transfer_dataset("target", "synthetic/target", "mamba")
    with pytest.raises(StaticFeatureStudyError, match="family must be represented"):
        run_cross_model_transfer((source,), target)


def test_q6_compares_protocols_and_records_schema_failures() -> None:
    pytest.importorskip("lightgbm")
    llama_a = _transfer_dataset("llama-a", "synthetic/llama-a", "llama")
    llama_b = _transfer_dataset("llama-b", "synthetic/llama-b", "llama")
    qwen = _transfer_dataset("qwen", "synthetic/qwen", "qwen")
    result = run_cross_family_transfer_study(
        (
            TransferExperiment(TransferProtocol.SINGLE_FAMILY, (llama_a,), llama_b),
            TransferExperiment(TransferProtocol.MULTI_FAMILY, (llama_a, qwen), llama_b),
            TransferExperiment(TransferProtocol.HELD_OUT_FAMILY, (llama_a, llama_b), qwen),
        ),
        StaticFeatureStudyConfig(top_n=2, threads=1, bootstrap_repetitions=20),
    )

    assert [outcome.status for outcome in result.outcomes] == [
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    assert result.multi_family_improvement is not None
    assert set(result.multi_family_improvement) == {
        "auc_gain",
        "calibration_error_reduction",
        "precision_at_top_n_gain",
        "mae_reduction",
        "rmse_reduction",
    }


def test_q6_retains_inference_schema_failure_as_an_outcome() -> None:
    pytest.importorskip("lightgbm")
    source = _transfer_dataset("source", "synthetic/source", "llama")
    target = _transfer_dataset("target", "synthetic/target", "qwen")
    features = target.records[-1]["pre_mutation_features"]
    assert isinstance(features, list)
    target.records[-1]["pre_mutation_features"] = [
        item for item in features if item["name"] != "weight_l2_norm"
    ]

    result = run_cross_family_transfer_study(
        (
            TransferExperiment(
                TransferProtocol.HELD_OUT_FAMILY,
                (source,),
                target,
            ),
        ),
        StaticFeatureStudyConfig(top_n=2, threads=1, bootstrap_repetitions=20),
    )

    outcome = result.outcomes[0]
    assert outcome.status == "failed"
    assert outcome.failure_kind == "TrainingMatrixError"
    assert outcome.failure_detail is not None
    assert "missing required trained features" in outcome.failure_detail


def test_q7_replays_paired_active_and_random_acquisition() -> None:
    pytest.importorskip("lightgbm")
    dataset = _transfer_dataset("active", "synthetic/active", "llama")
    result = run_model_active_learning_study(
        dataset.records,
        dataset.split,
        MultiModelActiveLearningConfig(
            budgets=(4, 8),
            seeds=(3, 5),
            target_auc=0.6,
            safe_perplexity_delta=0.25,
            threads=1,
        ),
    )

    assert len(result.curves) == 4
    assert all([point.experiments for point in curve.points] == [4, 8] for curve in result.curves)
    assert all(
        point.cumulative_gpu_hours > 0.0 for curve in result.curves for point in curve.points
    )
    assert set(result.comparison) == {
        "active_mean_aulc",
        "random_mean_aulc",
        "active_target_reaches",
        "random_target_reaches",
        "paired_target_reaches",
        "mean_experiment_reduction",
        "mean_gpu_hour_reduction",
    }


def test_q8_selection_matches_budget_and_retrains_after_revelation() -> None:
    pytest.importorskip("lightgbm")
    dataset = _transfer_dataset("pruning", "synthetic/pruning", "llama")
    result = run_matched_pruning_selection(
        dataset.records,
        dataset.split,
        budget=2,
        config=StaticFeatureStudyConfig(
            top_n=2,
            threads=1,
            bootstrap_repetitions=20,
            safe_perplexity_delta=0.25,
        ),
    )

    assert len(result.one_shot) == len(result.iterative) == result.budget == 2
    assert result.parameters_per_channel == 8
    assert len({item.example_id for item in result.iterative}) == 2
    assert result.one_shot_training_seconds > 0.0
    assert result.iterative_training_seconds > 0.0
    assert result.iterative_revealed_evaluation_seconds > 0.0


def test_static_study_rejects_invalid_protocol() -> None:
    with pytest.raises(StaticFeatureStudyError, match="bootstrap_repetitions"):
        StaticFeatureStudyConfig(bootstrap_repetitions=0)

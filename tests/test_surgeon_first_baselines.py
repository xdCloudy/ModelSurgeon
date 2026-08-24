"""Focused contracts for the first learned surgeon milestone."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.datasets.grouped_splits import SplitPartition
from modelsurgeon.experiments.candidates import CandidateScope, MutationCandidate
from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgeon.matrix import (
    TrainingMatrixError,
    build_training_matrices,
    transform_inference_record,
)
from modelsurgeon.surgeon.metrics import (
    evaluate_classification,
    evaluate_regression,
)
from modelsurgeon.surgeon.models import LinearConfig, LinearSurgeonModel, train_linear
from modelsurgeon.surgeon.ranking import (
    HeuristicConfig,
    MissingSignalPolicy,
    rank_heuristic,
    rank_magnitude,
    rank_random,
)
from modelsurgeon.surgeon.registry import (
    SurgeonModelRegistry,
    SurgeonRegistryError,
    TrainingModelIdentity,
)
from modelsurgeon.surgeon.targets import (
    DEFAULT_TARGET_SCHEMA,
    derive_supervised_targets,
    schema_with_thresholds,
)
from modelsurgeon.surgery.contracts import MutationKind, MutationRequest


def _metric(
    name: str, value: float | None, *, state: str = "measured"
) -> dict[str, object]:
    reason = None if state == "measured" else f"{name} unavailable"
    return {
        "name": name,
        "state": state,
        "value": value,
        "unit": None,
        "reason": reason,
    }


def _example(
    example_id: str,
    *,
    feature: float,
    family: str = "llama",
    baseline_ppl: float = 10.0,
    post_ppl: float = 10.5,
) -> dict[str, object]:
    component = "model.layers.0.mlp.up_proj"
    return {
        "example_id": example_id,
        "model": {
            "identifier": "tiny/model",
            "revision": "model-rev",
            "family": family,
            "format": "safetensors",
            "quantization": None,
        },
        "mutation": {
            "plan": {
                "request": {
                    "kind": "mask",
                    "targets": [component],
                    "parameters": {"candidate_scope": "component"},
                }
            }
        },
        "pre_mutation_features": [
            {
                "schema_version": 1,
                "component_id": component,
                "name": "weight_mean",
                "kind": "scalar",
                "value": feature,
            }
        ],
        "baseline_metrics": [_metric("perplexity", baseline_ppl)],
        "post_metrics": [_metric("perplexity", post_ppl)],
        "versions": {"feature_schema_version": 1},
    }


def test_targets_define_post_minus_baseline_units_masks_and_safe_labels() -> None:
    schema = schema_with_thresholds({"perplexity": 1.0})
    example = _example("example-1", feature=1.0, post_ppl=10.75)
    targets = derive_supervised_targets(example, schema)

    perplexity = targets.value("perplexity")
    assert perplexity.mask
    assert perplexity.unit == "perplexity"
    assert perplexity.value == pytest.approx(0.75)
    assert targets.safe_mutation_mask
    assert targets.safe_mutation is True

    missing = _example("example-2", feature=1.0)
    missing["post_metrics"] = [
        _metric("perplexity", None, state="skipped")
    ]
    masked = derive_supervised_targets(missing, schema)
    assert not masked.value("perplexity").mask
    assert not masked.safe_mutation_mask
    assert "perplexity" in (masked.safe_label_reason or "")


def _candidate(index: int) -> MutationCandidate:
    component = ComponentId.parse(f"model.layers.{index}.mlp.up_proj")
    request = MutationRequest(
        MutationKind.MASK,
        (component,),
        (("candidate_scope", "component"),),
    )
    return MutationCandidate(
        f"cand_{index}",
        CandidateScope.COMPONENT,
        component,
        "projection",
        index,
        request,
        (component,),
        (),
    )


def _feature(component: ComponentId, name: str, value: float) -> FeatureRecord:
    return FeatureRecord(
        component,
        name,
        FeatureKind.SCALAR,
        value,
        "float64",
        "test",
        "1",
        PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION, "float32", "float64"
        ),
    )


def test_random_magnitude_and_heuristic_rankings_are_deterministic_and_traced() -> None:
    candidates = (_candidate(0), _candidate(1), _candidate(2))
    first = rank_random(candidates, seed=42, select_count=2)
    repeated = rank_random(candidates, seed=42, select_count=2)
    assert first.to_record() == repeated.to_record()
    assert all(
        entry.selection_propensity == pytest.approx(2 / 3)
        for entry in first.entries
    )

    features: list[FeatureRecord] = []
    for candidate, l1, activation, sensitivity, similarity in zip(
        candidates,
        (1.0, 4.0, 9.0),
        (0.1, 0.3, 0.9),
        (0.1, 0.2, 0.8),
        (0.9, 0.5, 0.1),
        strict=True,
    ):
        component = candidate.component_id
        features.extend(
            (
                _feature(component, "weight_count", 4.0),
                _feature(component, "weight_l1_norm", l1),
                _feature(component, "activation_rms", activation),
                _feature(component, "sensitivity", sensitivity),
                _feature(component, "cosine_similarity", similarity),
            )
        )

    magnitude = rank_magnitude(candidates, features)
    assert [entry.candidate_id for entry in magnitude.entries] == [
        "cand_0",
        "cand_1",
        "cand_2",
    ]
    assert magnitude.entries[0].score == pytest.approx(0.25)

    heuristic = rank_heuristic(
        candidates,
        features,
        config=HeuristicConfig(missing_policy=MissingSignalPolicy.ERROR),
    )
    assert heuristic.ranking.entries[0].candidate_id == "cand_0"
    assert len(heuristic.traces) == 3
    assert all(len(trace.contributions) == 4 for trace in heuristic.traces)


def test_training_preprocessing_is_fit_only_on_train_and_inference_fails_closed() -> None:
    examples = (
        _example("train-a", feature=1.0, family="llama"),
        _example("train-b", feature=3.0, family="llama"),
        _example("validation", feature=1_000_000.0, family="qwen"),
        _example("test", feature=-1_000_000.0, family="gemma"),
    )
    split = {
        "train-a": SplitPartition.TRAIN,
        "train-b": SplitPartition.TRAIN,
        "validation": SplitPartition.VALIDATION,
        "test": SplitPartition.TEST,
    }
    matrices = build_training_matrices(
        examples,
        split,
        target_schema=DEFAULT_TARGET_SCHEMA,
        target_name="perplexity",
    )

    numeric = {item.name: item for item in matrices.preprocessor.numeric}
    assert numeric["weight_mean"].mean == pytest.approx(2.0)
    assert numeric["weight_mean"].scale == pytest.approx(1.0)
    family = next(
        item
        for item in matrices.preprocessor.categorical
        if item.name == "model_family"
    )
    assert "llama" in family.categories
    assert "qwen" not in family.categories
    assert "gemma" not in family.categories

    incomplete = dict(_example("inference", feature=2.0))
    incomplete["pre_mutation_features"] = []
    with pytest.raises(
        TrainingMatrixError, match="missing required trained features"
    ):
        transform_inference_record(incomplete, matrices.preprocessor)


def test_linear_model_trains_and_registry_round_trips_with_schema_guards(
    tmp_path: Path,
) -> None:
    examples = (
        _example("train-a", feature=1.0, post_ppl=10.2),
        _example("train-b", feature=2.0, post_ppl=10.4),
        _example("validation", feature=3.0, post_ppl=10.6),
        _example("test", feature=4.0, post_ppl=10.8),
    )
    split = {
        "train-a": SplitPartition.TRAIN,
        "train-b": SplitPartition.TRAIN,
        "validation": SplitPartition.VALIDATION,
        "test": SplitPartition.TEST,
    }
    matrices = build_training_matrices(
        examples,
        split,
        target_schema=DEFAULT_TARGET_SCHEMA,
        target_name="perplexity",
    )
    model = train_linear(
        matrices.train,
        config=LinearConfig(
            alpha=1e-6, learning_rate=0.01, max_epochs=100
        ),
        validation=matrices.validation,
    )
    assert isinstance(model, LinearSurgeonModel)
    assert model.feature_names == matrices.preprocessor.output_feature_names

    registry = SurgeonModelRegistry(str(tmp_path / "registry"))
    artifact = registry.publish(
        model,
        matrices.preprocessor,
        DEFAULT_TARGET_SCHEMA,
        training_models=(
            TrainingModelIdentity("tiny/model", "model-rev", "Q4_K_M"),
        ),
        metrics={"mae": 0.1},
        split_manifest=matrices.split_manifest,
        provenance={"seed": 7, "dataset_revision": "dataset-rev"},
    )
    loaded = registry.load(
        str(artifact.metadata.digest),
        expected_feature_schema_version=1,
        expected_target_schema=DEFAULT_TARGET_SCHEMA,
    )
    assert loaded.model.to_record() == model.to_record()
    assert loaded.card.training_models[0].quantization == "Q4_K_M"

    with pytest.raises(SurgeonRegistryError, match="feature schema version"):
        registry.load(
            str(artifact.metadata.digest),
            expected_feature_schema_version=2,
        )


def test_metric_reports_include_undefined_cases_top_n_and_grouped_bootstrap() -> None:
    regression = evaluate_regression(
        (1.0, 2.0, 3.0, 4.0),
        (1.1, 1.9, 3.2, 3.8),
        group_ids=("a", "a", "b", "b"),
        bootstrap_repetitions=50,
        seed=5,
    )
    values = {item.name: item for item in regression.metrics}
    assert values["mae"].defined
    assert values["mae"].confidence_low is not None
    assert values["rmse"].defined
    assert values["r2"].defined

    classification = evaluate_classification(
        (1, 0, 1, 0),
        (0.9, 0.1, 0.8, 0.2),
        top_n=2,
        group_ids=("a", "a", "b", "b"),
        bootstrap_repetitions=20,
        seed=3,
    )
    class_values = {item.name: item for item in classification.metrics}
    assert class_values["auc"].value == pytest.approx(1.0)
    assert class_values["precision_at_2"].value == pytest.approx(1.0)
    assert class_values["recall_at_2"].value == pytest.approx(1.0)

    undefined = evaluate_classification((1, 1), (0.9, 0.8), top_n=1)
    undefined_values = {item.name: item for item in undefined.metrics}
    assert undefined_values["auc"].value is None
    assert "one class" in (undefined_values["auc"].reason or "")

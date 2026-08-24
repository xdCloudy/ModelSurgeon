"""Matched one-shot versus sequential evaluate/retrain pruning selection."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from modelsurgeon.datasets.grouped_splits import GroupedSplitManifest, SplitPartition
from modelsurgeon.surgeon.matrix import build_training_matrices
from modelsurgeon.surgeon.models import LightGBMConfig, ModelTask, train_lightgbm
from modelsurgeon.surgeon.targets import DEFAULT_TARGET_SCHEMA, schema_with_thresholds

from .cross_model_transfer import _identity
from .static_feature_study import (
    FeatureProfile,
    StaticFeatureStudyConfig,
    StaticFeatureStudyError,
    select_feature_profile_records,
)


@dataclass(frozen=True, slots=True)
class PruningSelection:
    example_id: str
    layer_index: int
    channel_index: int
    predicted_safe_probability: float
    measured_perplexity_delta: float

    def to_record(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "layer_index": self.layer_index,
            "channel_index": self.channel_index,
            "predicted_safe_probability": self.predicted_safe_probability,
            "measured_single_channel_perplexity_delta": self.measured_perplexity_delta,
        }


@dataclass(frozen=True, slots=True)
class MatchedPruningSelections:
    model: tuple[str, str, str]
    budget: int
    parameters_per_channel: int
    one_shot: tuple[PruningSelection, ...]
    iterative: tuple[PruningSelection, ...]
    one_shot_training_seconds: float
    iterative_training_seconds: float
    iterative_revealed_evaluation_seconds: float

    def to_record(self) -> dict[str, object]:
        return {
            "model": {
                "identifier": self.model[0],
                "revision": self.model[1],
                "family": self.model[2],
            },
            "budget": self.budget,
            "parameters_per_channel": self.parameters_per_channel,
            "matched_parameter_proxy": self.parameters_per_channel * self.budget,
            "one_shot": [item.to_record() for item in self.one_shot],
            "iterative": [item.to_record() for item in self.iterative],
            "cost": {
                "one_shot_training_seconds": self.one_shot_training_seconds,
                "iterative_training_seconds": self.iterative_training_seconds,
                "iterative_revealed_evaluation_seconds": (
                    self.iterative_revealed_evaluation_seconds
                ),
                "iterative_retraining_steps": self.budget,
            },
        }


def _partition_map(split: GroupedSplitManifest) -> dict[str, SplitPartition]:
    return {
        example_id: group.partition for group in split.groups for example_id in group.example_ids
    }


def _coordinates(record: Mapping[str, object]) -> tuple[int, int]:
    mutation = record.get("mutation")
    if not isinstance(mutation, Mapping) or not isinstance(mutation.get("plan"), Mapping):
        raise StaticFeatureStudyError("pruning selection requires mutation provenance")
    plan = mutation["plan"]
    assert isinstance(plan, Mapping)
    request = plan.get("request")
    if not isinstance(request, Mapping) or not isinstance(request.get("parameters"), Mapping):
        raise StaticFeatureStudyError("pruning selection requires mutation coordinates")
    parameters = request["parameters"]
    assert isinstance(parameters, Mapping)
    layer = parameters.get("layer_index")
    channel = parameters.get("channel_index")
    if not isinstance(layer, int) or not isinstance(channel, int):
        raise StaticFeatureStudyError("pruning selection coordinates must be integers")
    return layer, channel


def _perplexity_delta(record: Mapping[str, object]) -> float:
    raw = record.get("delta_metrics")
    if not isinstance(raw, list):
        raise StaticFeatureStudyError("pruning selection lacks measured delta")
    for item in raw:
        if isinstance(item, Mapping) and item.get("name") == "perplexity":
            value = item.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    raise StaticFeatureStudyError("pruning selection lacks measured perplexity delta")


def _evaluation_seconds(record: Mapping[str, object]) -> float:
    raw = record.get("timings")
    if not isinstance(raw, list):
        raise StaticFeatureStudyError("pruning selection lacks evaluation timing")
    values = [
        float(item["wall_seconds"])
        for item in raw
        if isinstance(item, Mapping)
        and item.get("stage") == "evaluate"
        and isinstance(item.get("wall_seconds"), (int, float))
        and not isinstance(item.get("wall_seconds"), bool)
    ]
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise StaticFeatureStudyError("pruning selection evaluation timing is invalid")
    return math.fsum(values)


def _parameter_count(record: Mapping[str, object]) -> int:
    raw = record.get("pre_mutation_features")
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping) and item.get("name") == "weight_count":
                value = item.get("value")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    count = int(value)
                    if count > 0 and float(count) == float(value):
                        return count
    raise StaticFeatureStudyError("pruning selection lacks parameter-count proxy")


def run_matched_pruning_selection(
    records: Sequence[Mapping[str, object]],
    split: GroupedSplitManifest,
    *,
    budget: int = 10,
    config: StaticFeatureStudyConfig | None = None,
) -> MatchedPruningSelections:
    """Select matched masks once or with label revelation and retraining after each step."""

    if budget <= 0:
        raise StaticFeatureStudyError("matched pruning budget must be positive")
    resolved = config or StaticFeatureStudyConfig()
    selected_records = select_feature_profile_records(
        records, FeatureProfile.STATIC_ACTIVATION_GRADIENT
    )
    partitions = _partition_map(split)
    by_id = {str(record.get("example_id")): record for record in selected_records}
    test_ids = tuple(sorted(key for key in by_id if partitions[key] is SplitPartition.TEST))
    if budget > len(test_ids):
        raise StaticFeatureStudyError("matched pruning budget exceeds test pool")
    target_schema = schema_with_thresholds(
        {"perplexity": resolved.safe_perplexity_delta}, base=DEFAULT_TARGET_SCHEMA
    )

    def fit(acquired: set[str]) -> tuple[tuple[tuple[str, float], ...], float]:
        assignment = {
            example_id: (
                SplitPartition.TRAIN
                if partition is SplitPartition.TRAIN or example_id in acquired
                else partition
            )
            for example_id, partition in partitions.items()
        }
        started = time.perf_counter()
        matrices = build_training_matrices(
            selected_records,
            assignment,
            target_schema=target_schema,
            target_name="safe_mutation",
        )
        model = train_lightgbm(
            matrices.train,
            matrices.validation,
            config=LightGBMConfig(
                ModelTask.CLASSIFICATION,
                num_threads=resolved.threads,
                seed=resolved.seed,
            ),
        )
        predictions = model.predict(matrices.test.values)
        return (
            tuple(
                (example_id, float(prediction))
                for example_id, prediction in zip(
                    matrices.test.example_ids, predictions, strict=True
                )
            ),
            time.perf_counter() - started,
        )

    initial_predictions, one_shot_seconds = fit(set())
    one_shot_ranked = sorted(initial_predictions, key=lambda item: (-item[1], item[0]))[:budget]

    def selection(example_id: str, probability: float) -> PruningSelection:
        layer, channel = _coordinates(by_id[example_id])
        return PruningSelection(
            example_id,
            layer,
            channel,
            float(probability),
            _perplexity_delta(by_id[example_id]),
        )

    one_shot = tuple(
        selection(example_id, probability) for example_id, probability in one_shot_ranked
    )
    acquired: set[str] = set()
    iterative: list[PruningSelection] = []
    iterative_training_seconds = 0.0
    for _ in range(budget):
        predictions, elapsed = fit(acquired)
        iterative_training_seconds += elapsed
        remaining = tuple(item for item in predictions if item[0] not in acquired)
        example_id, probability = min(remaining, key=lambda item: (-item[1], item[0]))
        iterative.append(selection(example_id, probability))
        acquired.add(example_id)
    return MatchedPruningSelections(
        _identity(records),
        budget,
        _parameter_count(by_id[test_ids[0]]),
        one_shot,
        tuple(iterative),
        one_shot_seconds,
        iterative_training_seconds,
        math.fsum(_evaluation_seconds(by_id[item.example_id]) for item in iterative),
    )

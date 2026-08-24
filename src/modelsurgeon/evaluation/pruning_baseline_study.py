"""Equal-budget learned, magnitude, and random pruning study for v0.8 Q4."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from modelsurgeon.datasets.grouped_splits import GroupedSplitManifest

from .static_feature_study import (
    FeatureProfile,
    StaticFeatureStudyConfig,
    StaticFeatureStudyError,
    run_feature_profile_study,
)

DEFAULT_Q4_SEEDS = tuple(range(20))


@dataclass(frozen=True, slots=True)
class PruningBaselineStudyConfig:
    selection_budget: int = 10
    seeds: tuple[int, ...] = DEFAULT_Q4_SEEDS
    bootstrap_repetitions: int = 1000
    bootstrap_confidence: float = 0.95
    safe_perplexity_delta: float = 0.01
    threads: int = 4

    def __post_init__(self) -> None:
        if self.selection_budget <= 0 or not self.seeds:
            raise StaticFeatureStudyError("Q4 requires a selection budget and seeds")
        if len(self.seeds) != len(set(self.seeds)) or any(seed < 0 for seed in self.seeds):
            raise StaticFeatureStudyError("Q4 seeds must be unique and non-negative")
        if self.bootstrap_repetitions <= 0 or not 0 < self.bootstrap_confidence < 1:
            raise StaticFeatureStudyError("Q4 bootstrap configuration is invalid")


@dataclass(frozen=True, slots=True)
class SeedSelection:
    seed: int
    selected_example_ids: tuple[str, ...]
    mean_perplexity_delta: float
    constraint_violation_rate: float
    selected_perplexity_deltas: tuple[float, ...]
    selected_violations: tuple[float, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "selected_example_ids": list(self.selected_example_ids),
            "mean_perplexity_delta": self.mean_perplexity_delta,
            "constraint_violation_rate": self.constraint_violation_rate,
        }


@dataclass(frozen=True, slots=True)
class SeedInterval:
    name: str
    value: float
    confidence_low: float
    confidence_high: float
    bootstrap_repetitions: int

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "bootstrap_repetitions": self.bootstrap_repetitions,
        }


@dataclass(frozen=True, slots=True)
class PruningMethodResult:
    method: str
    selections: tuple[SeedSelection, ...]
    metrics: tuple[SeedInterval, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "method": self.method,
            "selections": [selection.to_record() for selection in self.selections],
            "metrics": [metric.to_record() for metric in self.metrics],
        }


@dataclass(frozen=True, slots=True)
class PruningBaselineStudyResult:
    model_identifier: str
    model_revision: str
    family: str
    pool_size: int
    selection_budget: int
    parameters_per_channel: int
    methods: tuple[PruningMethodResult, ...]
    paired_gains: tuple[SeedInterval, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "model": {
                "identifier": self.model_identifier,
                "revision": self.model_revision,
                "family": self.family,
            },
            "pool_size": self.pool_size,
            "selection_budget": self.selection_budget,
            "parameters_per_channel": self.parameters_per_channel,
            "selected_parameter_proxy": self.selection_budget * self.parameters_per_channel,
            "methods": [method.to_record() for method in self.methods],
            "paired_gains": [gain.to_record() for gain in self.paired_gains],
        }


def _feature_values(record: Mapping[str, object]) -> dict[str, float]:
    raw = record.get("pre_mutation_features")
    if not isinstance(raw, list):
        raise StaticFeatureStudyError("Q4 records require features")
    output: dict[str, float] = {}
    for feature in raw:
        if not isinstance(feature, Mapping):
            continue
        name = feature.get("name")
        value = feature.get("value")
        if (
            isinstance(name, str)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            output[name] = float(value)
    return output


def _selection(
    seed: int,
    indexes: Sequence[int],
    example_ids: Sequence[str],
    labels: Sequence[int],
    targets: Sequence[float],
) -> SeedSelection:
    selected = tuple(indexes)
    return SeedSelection(
        seed,
        tuple(example_ids[index] for index in selected),
        math.fsum(targets[index] for index in selected) / len(selected),
        math.fsum(1 - labels[index] for index in selected) / len(selected),
        tuple(targets[index] for index in selected),
        tuple(float(1 - labels[index]) for index in selected),
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _estimate_interval(
    name: str,
    point: float,
    estimates: Sequence[float],
    config: PruningBaselineStudyConfig,
) -> SeedInterval:
    tail = (1 - config.bootstrap_confidence) / 2
    return SeedInterval(
        name,
        point,
        _percentile(estimates, tail),
        _percentile(estimates, 1 - tail),
        len(estimates),
    )


def _hierarchical_interval(
    name: str,
    selections: Sequence[SeedSelection],
    field: str,
    config: PruningBaselineStudyConfig,
    seed: int,
) -> SeedInterval:
    raw = tuple(getattr(selection, field) for selection in selections)
    point = math.fsum(math.fsum(values) / len(values) for values in raw) / len(raw)
    randomizer = random.Random(seed)
    estimates: list[float] = []
    for _ in range(config.bootstrap_repetitions):
        seed_means: list[float] = []
        for _ in selections:
            values = randomizer.choice(raw)
            sample = tuple(randomizer.choice(values) for _ in values)
            seed_means.append(math.fsum(sample) / len(sample))
        estimates.append(math.fsum(seed_means) / len(seed_means))
    return _estimate_interval(name, point, estimates, config)


def _method(
    name: str,
    selections: Sequence[SeedSelection],
    config: PruningBaselineStudyConfig,
    seed_offset: int,
) -> PruningMethodResult:
    return PruningMethodResult(
        name,
        tuple(selections),
        (
            _hierarchical_interval(
                "constraint_violation_rate",
                selections,
                "selected_violations",
                config,
                900 + seed_offset,
            ),
            _hierarchical_interval(
                "mean_perplexity_delta",
                selections,
                "selected_perplexity_deltas",
                config,
                901 + seed_offset,
            ),
        ),
    )


def run_pruning_baseline_study(
    records: Sequence[Mapping[str, object]],
    split: GroupedSplitManifest,
    config: PruningBaselineStudyConfig | None = None,
) -> PruningBaselineStudyResult:
    """Compare equal-size held-out selections under paired seed schedules."""

    resolved = config or PruningBaselineStudyConfig()
    records_by_id = {str(record.get("example_id")): record for record in records}
    learned: list[SeedSelection] = []
    magnitude: list[SeedSelection] = []
    random_selections: list[SeedSelection] = []
    identity: tuple[str, str, str] | None = None
    parameters_per_channel: int | None = None
    pool_size: int | None = None

    for seed in resolved.seeds:
        study = run_feature_profile_study(
            records,
            split,
            FeatureProfile.STATIC_ACTIVATION_GRADIENT,
            StaticFeatureStudyConfig(
                safe_perplexity_delta=resolved.safe_perplexity_delta,
                top_n=resolved.selection_budget,
                threads=resolved.threads,
                seed=seed,
                bootstrap_repetitions=10,
            ),
        )
        if resolved.selection_budget > len(study.test_example_ids):
            raise StaticFeatureStudyError("Q4 selection budget exceeds held-out pool")
        identity = (study.model_identifier, study.model_revision, study.family)
        pool_size = len(study.test_example_ids)
        features = tuple(
            _feature_values(records_by_id[example_id]) for example_id in study.test_example_ids
        )
        magnitudes: list[float] = []
        for values in features:
            count = values.get("weight_count")
            l1 = values.get("weight_l1_norm")
            if count is None or l1 is None or count <= 0:
                raise StaticFeatureStudyError("Q4 magnitude fields are incomplete")
            magnitudes.append(l1 / count)
            current_count = int(count)
            if parameters_per_channel is None:
                parameters_per_channel = current_count
            elif parameters_per_channel != current_count:
                raise StaticFeatureStudyError("Q4 channel costs differ within a model")
        learned_indexes = tuple(
            sorted(
                range(len(study.test_example_ids)),
                key=lambda index: (
                    -study.test_classifier_predictions[index],
                    study.test_example_ids[index],
                ),
            )[: resolved.selection_budget]
        )
        magnitude_indexes = tuple(
            sorted(
                range(len(study.test_example_ids)),
                key=lambda index: (magnitudes[index], study.test_example_ids[index]),
            )[: resolved.selection_budget]
        )
        random_indexes = list(range(len(study.test_example_ids)))
        random.Random(seed).shuffle(random_indexes)
        random_indexes = random_indexes[: resolved.selection_budget]
        arguments = (study.test_example_ids, study.test_labels, study.test_targets)
        learned.append(_selection(seed, learned_indexes, *arguments))
        magnitude.append(_selection(seed, magnitude_indexes, *arguments))
        random_selections.append(_selection(seed, random_indexes, *arguments))

    if identity is None or parameters_per_channel is None or pool_size is None:
        raise StaticFeatureStudyError("Q4 produced no study runs")
    methods = (
        _method("learned_gradient_lightgbm", learned, resolved, 0),
        _method("magnitude_mean_absolute", magnitude, resolved, 10),
        _method("seeded_random", random_selections, resolved, 20),
    )
    paired: list[SeedInterval] = []
    for baseline_name, baseline, offset in (
        ("magnitude", magnitude, 30),
        ("random", random_selections, 40),
    ):
        for metric_name, point_field, field, seed in (
            (
                "mean_perplexity_delta_reduction",
                "mean_perplexity_delta",
                "selected_perplexity_deltas",
                1000 + offset,
            ),
            (
                "constraint_violation_reduction",
                "constraint_violation_rate",
                "selected_violations",
                1001 + offset,
            ),
        ):
            point = math.fsum(
                getattr(right, point_field) - getattr(left, point_field)
                for left, right in zip(learned, baseline, strict=True)
            ) / len(learned)
            randomizer = random.Random(seed)
            estimates: list[float] = []
            for _ in range(resolved.bootstrap_repetitions):
                differences: list[float] = []
                for _ in learned:
                    index = randomizer.randrange(len(learned))
                    left_values = getattr(learned[index], field)
                    right_values = getattr(baseline[index], field)
                    left_mean = math.fsum(
                        randomizer.choice(left_values) for _ in left_values
                    ) / len(left_values)
                    right_mean = math.fsum(
                        randomizer.choice(right_values) for _ in right_values
                    ) / len(right_values)
                    differences.append(right_mean - left_mean)
                estimates.append(math.fsum(differences) / len(differences))
            paired.append(
                _estimate_interval(f"{metric_name}_vs_{baseline_name}", point, estimates, resolved)
            )
    return PruningBaselineStudyResult(
        identity[0],
        identity[1],
        identity[2],
        pool_size,
        resolved.selection_budget,
        parameters_per_channel,
        methods,
        tuple(sorted(paired, key=lambda item: item.name)),
    )

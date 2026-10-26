"""Offline active-versus-random acquisition over measured model campaigns."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.active_learning import (
    AcquisitionCandidate,
    AcquisitionPolicyConfig,
    DiversityCandidate,
    DiversitySelectionConfig,
    acquire_candidates,
)
from modelsurgeon.datasets.grouped_splits import GroupedSplitManifest, SplitPartition
from modelsurgeon.surgeon.matrix import (
    SurgeonPreprocessor,
    build_training_matrices,
    transform_inference_record,
)
from modelsurgeon.surgeon.metrics import roc_auc
from modelsurgeon.surgeon.models import (
    LightGBMConfig,
    LightGBMSurgeonModel,
    ModelTask,
    train_lightgbm,
)
from modelsurgeon.surgeon.targets import (
    DEFAULT_TARGET_SCHEMA,
    TargetSchema,
    derive_supervised_targets,
    schema_with_thresholds,
)

from .cross_model_transfer import _identity
from .static_feature_study import (
    FeatureProfile,
    StaticFeatureStudyError,
    select_feature_profile_records,
)


class AcquisitionStrategy(StrEnum):
    ACTIVE_UNCERTAINTY = "active_uncertainty"
    UNCERTAINTY_DIVERSITY = "uncertainty_diversity"
    SEEDED_RANDOM = "seeded_random"


@dataclass(frozen=True, slots=True)
class AcquisitionPoint:
    experiments: int
    test_auc: float
    cumulative_gpu_hours: float


@dataclass(frozen=True, slots=True)
class AcquisitionCurve:
    strategy: AcquisitionStrategy
    seed: int
    points: tuple[AcquisitionPoint, ...]
    experiments_to_target: int | None
    gpu_hours_to_target: float | None

    @property
    def area_under_learning_curve(self) -> float:
        width = self.points[-1].experiments - self.points[0].experiments
        return (
            math.fsum(
                (right.experiments - left.experiments) * (left.test_auc + right.test_auc) / 2.0
                for left, right in zip(self.points, self.points[1:], strict=False)
            )
            / width
        )

    def to_record(self) -> dict[str, object]:
        return {
            "strategy": self.strategy.value,
            "seed": self.seed,
            "area_under_learning_curve": self.area_under_learning_curve,
            "experiments_to_target": self.experiments_to_target,
            "gpu_hours_to_target": self.gpu_hours_to_target,
            "points": [
                {
                    "experiments": point.experiments,
                    "test_auc": point.test_auc,
                    "cumulative_gpu_hours": point.cumulative_gpu_hours,
                }
                for point in self.points
            ],
        }


@dataclass(frozen=True, slots=True)
class MultiModelActiveLearningConfig:
    budgets: tuple[int, ...] = (64, 128, 256, 384)
    seeds: tuple[int, ...] = (11, 23, 37, 53, 71)
    target_auc: float = 0.8
    safe_perplexity_delta: float = 0.01
    threads: int = 1
    feature_diversity_fraction: float = 0.25

    def __post_init__(self) -> None:
        if (
            len(self.budgets) < 2
            or self.budgets != tuple(sorted(set(self.budgets)))
            or self.budgets[0] < 2
        ):
            raise StaticFeatureStudyError("active-learning budgets must be canonical")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise StaticFeatureStudyError("active-learning seeds must be unique")
        if not 0.5 < self.target_auc <= 1.0:
            raise StaticFeatureStudyError("active-learning target AUC must be in (0.5, 1]")
        if (
            not math.isfinite(self.feature_diversity_fraction)
            or not 0.0 < self.feature_diversity_fraction < 1.0
        ):
            raise StaticFeatureStudyError("feature diversity fraction must be within (0, 1)")


@dataclass(frozen=True, slots=True)
class ModelActiveLearningResult:
    model: tuple[str, str, str]
    target_auc: float
    train_pool_size: int
    validation_size: int
    test_size: int
    curves: tuple[AcquisitionCurve, ...]
    comparison: Mapping[str, float | int | None]
    feature_diversity_fraction: float = 0.25

    def to_record(self) -> dict[str, object]:
        return {
            "model": {
                "identifier": self.model[0],
                "revision": self.model[1],
                "family": self.model[2],
            },
            "target_auc": self.target_auc,
            "feature_diversity_fraction": self.feature_diversity_fraction,
            "counts": {
                "train_pool": self.train_pool_size,
                "validation": self.validation_size,
                "test": self.test_size,
            },
            "curves": [curve.to_record() for curve in self.curves],
            "comparison": dict(self.comparison),
        }


def _partition_map(split: GroupedSplitManifest) -> dict[str, SplitPartition]:
    return {
        example_id: group.partition for group in split.groups for example_id in group.example_ids
    }


def _evaluation_seconds(record: Mapping[str, object]) -> float:
    raw = record.get("timings")
    if not isinstance(raw, list):
        raise StaticFeatureStudyError("active-learning record lacks timing provenance")
    values = []
    for item in raw:
        if isinstance(item, Mapping) and item.get("stage") == "evaluate":
            value = item.get("wall_seconds")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise StaticFeatureStudyError("active-learning evaluation timing is invalid")
    return math.fsum(values)


def _fit(
    selected_ids: set[str],
    records_by_id: Mapping[str, Mapping[str, object]],
    validation_ids: Sequence[str],
    test_ids: Sequence[str],
    target_schema: TargetSchema,
    config: MultiModelActiveLearningConfig,
) -> tuple[LightGBMSurgeonModel, SurgeonPreprocessor, float]:
    included = tuple(
        records_by_id[example_id]
        for example_id in (*sorted(selected_ids), *validation_ids, *test_ids)
    )
    split = {
        **{example_id: SplitPartition.TRAIN for example_id in selected_ids},
        **{example_id: SplitPartition.VALIDATION for example_id in validation_ids},
        **{example_id: SplitPartition.TEST for example_id in test_ids},
    }
    matrices = build_training_matrices(
        included,
        split,
        target_schema=target_schema,
        target_name="safe_mutation",
    )
    model = train_lightgbm(
        matrices.train,
        matrices.validation,
        config=LightGBMConfig(
            ModelTask.CLASSIFICATION,
            num_threads=config.threads,
            seed=42,
        ),
    )
    predictions = model.predict(matrices.test.values)
    auc = roc_auc(tuple(int(value) for value in matrices.test.target_values), predictions)
    if auc is None:
        raise StaticFeatureStudyError("active-learning test AUC is undefined")
    return model, matrices.preprocessor, auc


def _initial_order(
    ids: Sequence[str], labels: Mapping[str, int], seed: int, size: int
) -> list[str]:
    order = list(ids)
    random.Random(seed).shuffle(order)
    selected = order[:size]
    if len({labels[item] for item in selected}) == 1:
        missing = 1 - labels[selected[0]]
        try:
            replacement_index = next(
                index for index in range(size, len(order)) if labels[order[index]] == missing
            )
        except StopIteration as error:
            raise StaticFeatureStudyError(
                "active-learning training pool requires both safe-mutation classes"
            ) from error
        order[size - 1], order[replacement_index] = (
            order[replacement_index],
            order[size - 1],
        )
    return order


def _candidate_id(example_id: str) -> str:
    """Create the canonical acquisition ID without exposing raw corpus IDs."""

    return "cand_" + hashlib.sha256(example_id.encode("utf-8")).hexdigest()


def _structure_features(record: Mapping[str, object]) -> tuple[tuple[str, ...], frozenset[str]]:
    """Extract outcome-free categorical and topology hints from mutation metadata."""

    model = record.get("model")
    model_record = model if isinstance(model, Mapping) else {}
    mutation = record.get("mutation")
    mutation_record = mutation if isinstance(mutation, Mapping) else {}
    plan = mutation_record.get("plan")
    plan_record = plan if isinstance(plan, Mapping) else {}
    request = plan_record.get("request")
    request_record = request if isinstance(request, Mapping) else {}
    parameters = request_record.get("parameters")
    parameter_record = parameters if isinstance(parameters, Mapping) else {}

    categorical = tuple(
        f"{key}:{value}"
        for key, value in (
            ("family", model_record.get("family")),
            ("format", model_record.get("format")),
            ("quantization", model_record.get("quantization") or "none"),
            ("mutation", request_record.get("kind")),
            ("scope", parameter_record.get("candidate_scope")),
        )
        if isinstance(value, str) and value
    )
    topology = frozenset(
        f"{key}:{value}"
        for key in (
            "layer_index",
            "block_index",
            "channel_index",
            "head_index",
            "expert_index",
            "module_name",
            "target_name",
        )
        if isinstance((value := parameter_record.get(key)), (str, int))
        and not isinstance(value, bool)
    )
    return categorical, topology


def run_model_active_learning_study(
    records: Sequence[Mapping[str, object]],
    split: GroupedSplitManifest,
    config: MultiModelActiveLearningConfig | None = None,
) -> ModelActiveLearningResult:
    """Compare uncertainty, uncertainty/diversity, and paired random acquisition."""

    resolved = config or MultiModelActiveLearningConfig()
    selected_records = select_feature_profile_records(
        records, FeatureProfile.STATIC_ACTIVATION_GRADIENT
    )
    partitions = _partition_map(split)
    by_id = {str(record.get("example_id")): record for record in selected_records}
    train_ids = tuple(sorted(key for key in by_id if partitions[key] is SplitPartition.TRAIN))
    validation_ids = tuple(
        sorted(key for key in by_id if partitions[key] is SplitPartition.VALIDATION)
    )
    test_ids = tuple(sorted(key for key in by_id if partitions[key] is SplitPartition.TEST))
    if resolved.budgets[-1] > len(train_ids):
        raise StaticFeatureStudyError("active-learning budget exceeds training pool")
    target_schema = schema_with_thresholds(
        {"perplexity": resolved.safe_perplexity_delta}, base=DEFAULT_TARGET_SCHEMA
    )
    labels: dict[str, int] = {}
    for example_id in train_ids:
        safe = derive_supervised_targets(by_id[example_id], target_schema).safe_mutation
        if safe is None:
            raise StaticFeatureStudyError("active-learning training label is unavailable")
        labels[example_id] = int(safe)
    timings = {example_id: _evaluation_seconds(by_id[example_id]) for example_id in train_ids}
    curves: list[AcquisitionCurve] = []
    for seed in resolved.seeds:
        base_order = _initial_order(train_ids, labels, seed, resolved.budgets[0])
        for strategy in AcquisitionStrategy:
            selected = set(base_order[: resolved.budgets[0]])
            random_order = tuple(base_order)
            points: list[AcquisitionPoint] = []
            experiments_to_target: int | None = None
            gpu_hours_to_target: float | None = None
            for index, budget in enumerate(resolved.budgets):
                model, preprocessor, auc = _fit(
                    selected, by_id, validation_ids, test_ids, target_schema, resolved
                )
                gpu_hours = math.fsum(timings[item] for item in selected) / 3600.0
                points.append(AcquisitionPoint(budget, auc, gpu_hours))
                if experiments_to_target is None and auc >= resolved.target_auc:
                    experiments_to_target = budget
                    gpu_hours_to_target = gpu_hours
                if index + 1 == len(resolved.budgets):
                    continue
                next_budget = resolved.budgets[index + 1]
                remaining = [item for item in train_ids if item not in selected]
                if strategy is AcquisitionStrategy.SEEDED_RANDOM:
                    additions = [item for item in random_order if item not in selected][
                        : next_budget - budget
                    ]
                else:
                    rows = tuple(
                        transform_inference_record(by_id[example_id], preprocessor)
                        for example_id in remaining
                    )
                    probabilities = model.predict(rows)
                    if strategy is AcquisitionStrategy.ACTIVE_UNCERTAINTY:
                        scored = [
                            (abs(probability - 0.5), example_id)
                            for example_id, probability in zip(
                                remaining, probabilities, strict=True
                            )
                        ]
                        additions = [
                            item[1]
                            for item in sorted(scored, key=lambda item: (item[0], item[1]))[
                                : next_budget - budget
                            ]
                        ]
                    else:
                        additions = _acquire_uncertainty_diversity(
                            remaining,
                            probabilities,
                            rows,
                            by_id,
                            selected,
                            preprocessor,
                            next_budget - budget,
                            seed,
                            resolved.feature_diversity_fraction,
                        )
                selected.update(additions)
            curves.append(
                AcquisitionCurve(
                    strategy,
                    seed,
                    tuple(points),
                    experiments_to_target,
                    gpu_hours_to_target,
                )
            )

    grouped = {
        strategy: tuple(curve for curve in curves if curve.strategy is strategy)
        for strategy in AcquisitionStrategy
    }
    paired = tuple(
        (active, random_curve)
        for active in grouped[AcquisitionStrategy.ACTIVE_UNCERTAINTY]
        for random_curve in grouped[AcquisitionStrategy.SEEDED_RANDOM]
        if active.seed == random_curve.seed
        and active.experiments_to_target is not None
        and random_curve.experiments_to_target is not None
        and active.gpu_hours_to_target is not None
        and random_curve.gpu_hours_to_target is not None
    )
    experiment_reductions: list[float] = []
    gpu_hour_reductions: list[float] = []
    for active, random_curve in paired:
        assert active.experiments_to_target is not None
        assert random_curve.experiments_to_target is not None
        assert active.gpu_hours_to_target is not None
        assert random_curve.gpu_hours_to_target is not None
        experiment_reductions.append(
            float(random_curve.experiments_to_target - active.experiments_to_target)
        )
        gpu_hour_reductions.append(random_curve.gpu_hours_to_target - active.gpu_hours_to_target)
    comparison: dict[str, float | int | None] = {
        "active_mean_aulc": math.fsum(
            curve.area_under_learning_curve
            for curve in grouped[AcquisitionStrategy.ACTIVE_UNCERTAINTY]
        )
        / len(resolved.seeds),
        "random_mean_aulc": math.fsum(
            curve.area_under_learning_curve for curve in grouped[AcquisitionStrategy.SEEDED_RANDOM]
        )
        / len(resolved.seeds),
        "active_target_reaches": sum(
            curve.experiments_to_target is not None
            for curve in grouped[AcquisitionStrategy.ACTIVE_UNCERTAINTY]
        ),
        "random_target_reaches": sum(
            curve.experiments_to_target is not None
            for curve in grouped[AcquisitionStrategy.SEEDED_RANDOM]
        ),
        "paired_target_reaches": len(paired),
        "mean_experiment_reduction": (
            None if not paired else math.fsum(experiment_reductions) / len(paired)
        ),
        "mean_gpu_hour_reduction": (
            None if not paired else math.fsum(gpu_hour_reductions) / len(paired)
        ),
    }
    diverse_paired = tuple(
        (diverse, random_curve)
        for diverse in grouped[AcquisitionStrategy.UNCERTAINTY_DIVERSITY]
        for random_curve in grouped[AcquisitionStrategy.SEEDED_RANDOM]
        if diverse.seed == random_curve.seed
        and diverse.experiments_to_target is not None
        and random_curve.experiments_to_target is not None
        and diverse.gpu_hours_to_target is not None
        and random_curve.gpu_hours_to_target is not None
    )
    comparison.update(
        {
            "diversity_mean_aulc": math.fsum(
                curve.area_under_learning_curve
                for curve in grouped[AcquisitionStrategy.UNCERTAINTY_DIVERSITY]
            )
            / len(resolved.seeds),
            "diversity_target_reaches": sum(
                curve.experiments_to_target is not None
                for curve in grouped[AcquisitionStrategy.UNCERTAINTY_DIVERSITY]
            ),
            "paired_diversity_target_reaches": len(diverse_paired),
            "mean_diversity_experiment_reduction": (
                None
                if not diverse_paired
                else math.fsum(
                    float(random_curve.experiments_to_target - diverse.experiments_to_target)
                    for diverse, random_curve in diverse_paired
                    if diverse.experiments_to_target is not None
                    and random_curve.experiments_to_target is not None
                )
                / len(diverse_paired)
            ),
            "mean_diversity_gpu_hour_reduction": (
                None
                if not diverse_paired
                else math.fsum(
                    random_curve.gpu_hours_to_target - diverse.gpu_hours_to_target
                    for diverse, random_curve in diverse_paired
                    if diverse.gpu_hours_to_target is not None
                    and random_curve.gpu_hours_to_target is not None
                )
                / len(diverse_paired)
            ),
        }
    )
    return ModelActiveLearningResult(
        _identity(records),
        resolved.target_auc,
        len(train_ids),
        len(validation_ids),
        len(test_ids),
        tuple(curves),
        comparison,
        resolved.feature_diversity_fraction,
    )


def _acquire_uncertainty_diversity(
    remaining: Sequence[str],
    probabilities: Sequence[float],
    rows: Sequence[tuple[float, ...]],
    records: Mapping[str, Mapping[str, object]],
    selected_ids: set[str],
    preprocessor: SurgeonPreprocessor,
    budget: int,
    seed: int,
    diversity_fraction: float,
) -> list[str]:
    """Acquire a fixed uncertainty/diversity mixture from an unlabeled residual pool."""

    if not (len(remaining) == len(probabilities) == len(rows)):
        raise StaticFeatureStudyError("active-learning prediction rows do not align with pool")
    id_by_candidate = {example_id: _candidate_id(example_id) for example_id in remaining}
    example_by_candidate = {
        candidate_id: example_id for example_id, candidate_id in id_by_candidate.items()
    }
    acquisition_candidates = tuple(
        AcquisitionCandidate(
            id_by_candidate[example_id],
            utility=0.0,
            safe_probability=float(probability),
            uncertainty=4.0 * float(probability) * (1.0 - float(probability)),
            diversity=0.0,
        )
        for example_id, probability in zip(remaining, probabilities, strict=True)
    )
    diversity_candidates: list[DiversityCandidate] = []
    for example_id, row in zip(remaining, rows, strict=True):
        categorical, topology = _structure_features(records[example_id])
        diversity_candidates.append(
            DiversityCandidate(
                id_by_candidate[example_id],
                tuple(float(value) for value in row),
                categorical,
                topology,
            )
        )
    initial_centers: list[DiversityCandidate] = []
    for example_id in sorted(selected_ids):
        categorical, topology = _structure_features(records[example_id])
        initial_centers.append(
            DiversityCandidate(
                _candidate_id(example_id),
                transform_inference_record(records[example_id], preprocessor),
                categorical,
                topology,
            )
        )
    report = acquire_candidates(
        acquisition_candidates,
        budget,
        config=AcquisitionPolicyConfig(
            high_value_fraction=0.0,
            uncertain_fraction=1.0 - diversity_fraction,
            diverse_fraction=diversity_fraction,
            seed=seed,
        ),
        diversity_candidates=tuple(diversity_candidates),
        diversity_config=DiversitySelectionConfig(seed=seed),
        diversity_initial_centers=tuple(initial_centers),
    )
    return [example_by_candidate[item.candidate_id] for item in report.selections]

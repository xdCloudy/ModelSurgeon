"""Leakage-safe calibrated predictors conditioned on current state embeddings."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.datasets.grouped_splits import SplitPartition
from modelsurgeon.surgeon.matrix import SurgeonMatrix
from modelsurgeon.surgeon.models import LinearConfig, LinearSurgeonModel, train_linear
from modelsurgeon.surgeon.state_embedding import StateEmbedding

STATE_PREDICTOR_SCHEMA_VERSION: Final[int] = 1
STATE_PREDICTOR_PROTOCOL_REVISION: Final[str] = "state-predictor-v1"


class StatePredictorError(ValueError):
    """Raised when state predictor evidence or inference is unsafe."""


class StateTargetName(StrEnum):
    SAFETY_PROBABILITY = "safety_probability"
    QUALITY_DELTA = "quality_delta"
    LATENCY_DELTA = "latency_delta"
    MEMORY_DELTA = "memory_delta"
    NON_ADDITIVITY_ERROR = "non_additivity_error"


class StateTargetStatus(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class StatePredictorOutcome(StrEnum):
    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class StatePredictionStatus(StrEnum):
    PREDICTED = "predicted"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


_TARGET_UNITS: Final[dict[StateTargetName, str]] = {
    StateTargetName.SAFETY_PROBABILITY: "probability",
    StateTargetName.QUALITY_DELTA: "quality_loss",
    StateTargetName.LATENCY_DELTA: "seconds",
    StateTargetName.MEMORY_DELTA: "bytes",
    StateTargetName.NON_ADDITIVITY_ERROR: "delta",
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StatePredictorError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise StatePredictorError(f"{label} must be finite")
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StatePredictorError(f"{label} is required")
    return value


@dataclass(frozen=True, slots=True)
class StateTargetObservation:
    target: StateTargetName
    unit: str
    status: StateTargetStatus
    value: float | None = None
    additive_baseline: float | None = None

    def __post_init__(self) -> None:
        if self.unit != _TARGET_UNITS[self.target]:
            raise StatePredictorError(f"target unit mismatch for {self.target.value}")
        if self.status is StateTargetStatus.MEASURED:
            if self.value is None:
                raise StatePredictorError("measured state targets require a value")
            value = _finite(self.value, f"{self.target.value} value")
            if self.target is StateTargetName.SAFETY_PROBABILITY and not 0.0 <= value <= 1.0:
                raise StatePredictorError("safety probability must be within [0, 1]")
        elif self.value is not None:
            raise StatePredictorError("non-measured state targets cannot carry values")
        if self.additive_baseline is not None:
            _finite(self.additive_baseline, "additive baseline")

    def to_record(self) -> dict[str, object]:
        return {
            "target": self.target.value,
            "unit": self.unit,
            "status": self.status.value,
            "value": self.value,
            "additive_baseline": self.additive_baseline,
        }


@dataclass(frozen=True, slots=True)
class StatePredictionSample:
    sample_id: str
    parent_state_id: str
    model_revision: str
    candidate_id: str
    lineage_group_id: str
    embedding: StateEmbedding
    targets: tuple[StateTargetObservation, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("sample ID", self.sample_id),
            ("parent state ID", self.parent_state_id),
            ("model revision", self.model_revision),
            ("candidate ID", self.candidate_id),
            ("lineage group ID", self.lineage_group_id),
        ):
            _text(value, label)
        if self.embedding.state_id != self.parent_state_id:
            raise StatePredictorError("sample embedding does not match parent state")
        targets = tuple(item.target for item in self.targets)
        if len(targets) != len(set(targets)):
            raise StatePredictorError("sample targets must be unique and canonical")
        object.__setattr__(
            self,
            "targets",
            tuple(sorted(self.targets, key=lambda item: item.target.value)),
        )

    def target(self, target: StateTargetName) -> StateTargetObservation | None:
        return next((item for item in self.targets if item.target is target), None)

    def to_record(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "parent_state_id": self.parent_state_id,
            "model_revision": self.model_revision,
            "candidate_id": self.candidate_id,
            "lineage_group_id": self.lineage_group_id,
            "embedding_id": self.embedding.embedding_id,
            "targets": [item.to_record() for item in self.targets],
        }


@dataclass(frozen=True, slots=True)
class StatePredictionSplit:
    train_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]
    test_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = tuple(
            set(values)
            for values in (
                self.train_group_ids,
                self.validation_group_ids,
                self.test_group_ids,
            )
        )
        if any(not group for group in groups):
            raise StatePredictorError("every predictor split must contain groups")
        if any(
            len(group) != len(values)
            for group, values in zip(
                groups,
                (self.train_group_ids, self.validation_group_ids, self.test_group_ids),
                strict=True,
            )
        ):
            raise StatePredictorError("predictor split groups must be unique")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise StatePredictorError("predictor split groups must be disjoint")

    def to_record(self) -> dict[str, object]:
        return {
            "train_group_ids": list(self.train_group_ids),
            "validation_group_ids": list(self.validation_group_ids),
            "test_group_ids": list(self.test_group_ids),
        }


@dataclass(frozen=True, slots=True)
class StatePredictionDataset:
    samples: tuple[StatePredictionSample, ...]
    split: StatePredictionSplit
    dataset_revision: str
    schema_version: int = STATE_PREDICTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATE_PREDICTOR_SCHEMA_VERSION:
            raise StatePredictorError("unsupported state predictor dataset schema")
        _text(self.dataset_revision, "dataset revision")
        if not self.samples or len({sample.sample_id for sample in self.samples}) != len(
            self.samples
        ):
            raise StatePredictorError("predictor samples must be non-empty and unique")
        embedding_shape = (
            self.samples[0].embedding.feature_names,
            len(self.samples[0].embedding.values),
        )
        for sample in self.samples:
            if (sample.embedding.feature_names, len(sample.embedding.values)) != embedding_shape:
                raise StatePredictorError("all predictor embeddings must have one fixed shape")
        group_partition = {group: SplitPartition.TRAIN for group in self.split.train_group_ids}
        group_partition.update(
            {group: SplitPartition.VALIDATION for group in self.split.validation_group_ids}
        )
        group_partition.update({group: SplitPartition.TEST for group in self.split.test_group_ids})
        partitions: dict[SplitPartition, set[str]] = {
            SplitPartition.TRAIN: set(),
            SplitPartition.VALIDATION: set(),
            SplitPartition.TEST: set(),
        }
        for sample in self.samples:
            partition = group_partition.get(sample.lineage_group_id)
            if partition is None:
                raise StatePredictorError("every sample lineage must appear in one split")
            partitions[partition].update(self._leakage_keys(sample))
        if (
            partitions[SplitPartition.TRAIN] & partitions[SplitPartition.VALIDATION]
            or partitions[SplitPartition.TRAIN] & partitions[SplitPartition.TEST]
            or partitions[SplitPartition.VALIDATION] & partitions[SplitPartition.TEST]
        ):
            raise StatePredictorError("model, state, or candidate leakage crosses predictor splits")

    @staticmethod
    def _leakage_keys(sample: StatePredictionSample) -> set[str]:
        return {
            f"model:{sample.model_revision}",
            f"state:{sample.parent_state_id}",
            f"candidate:{sample.candidate_id}",
        }

    def partition_for(self, sample: StatePredictionSample) -> SplitPartition:
        if sample.lineage_group_id in self.split.train_group_ids:
            return SplitPartition.TRAIN
        if sample.lineage_group_id in self.split.validation_group_ids:
            return SplitPartition.VALIDATION
        if sample.lineage_group_id in self.split.test_group_ids:
            return SplitPartition.TEST
        raise StatePredictorError("sample lineage is absent from predictor split")

    @property
    def dataset_id(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "dataset_revision": self.dataset_revision,
            "split": self.split.to_record(),
            "samples": [sample.to_record() for sample in self.samples],
        }
        return (
            f"state_prediction_dataset_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"
        )


@dataclass(frozen=True, slots=True)
class StatePredictorConfig:
    target_names: tuple[StateTargetName, ...] = tuple(StateTargetName)
    minimum_train_examples: int = 3
    confidence: float = 0.95
    ridge: float = 1e-3
    seed: int = 0
    include_masks: bool = True

    def __post_init__(self) -> None:
        if len(self.target_names) != len(set(self.target_names)):
            raise StatePredictorError("predictor targets must be unique and canonical")
        if not self.target_names or self.minimum_train_examples < 2:
            raise StatePredictorError("predictor targets and minimum training count are invalid")
        if not 0.5 < self.confidence < 1.0 or not math.isfinite(self.confidence):
            raise StatePredictorError("predictor confidence must be in (0.5, 1)")
        if self.ridge < 0.0 or not math.isfinite(self.ridge):
            raise StatePredictorError("predictor ridge must be finite and non-negative")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise StatePredictorError("predictor seed must be unsigned")
        object.__setattr__(
            self,
            "target_names",
            tuple(sorted(self.target_names, key=lambda item: item.value)),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "target_names": [item.value for item in self.target_names],
            "minimum_train_examples": self.minimum_train_examples,
            "confidence": self.confidence,
            "ridge": self.ridge,
            "seed": self.seed,
            "include_masks": self.include_masks,
            "model_family": "linear_ridge",
            "calibration": "validation_absolute_residual_quantile",
            "baselines": ["stateless_mean", "additive_baseline"],
        }


def _feature_names(sample: StatePredictionSample, include_masks: bool) -> tuple[str, ...]:
    names = tuple(f"value:{name}" for name in sample.embedding.feature_names)
    if include_masks:
        return names + tuple(f"mask:{name}" for name in sample.embedding.feature_names)
    return names


def _feature_row(sample: StatePredictionSample, include_masks: bool) -> tuple[float, ...]:
    values = sample.embedding.values
    if include_masks:
        return values + sample.embedding.masks
    return values


def _quantile(values: Sequence[float], confidence: float) -> float:
    if not values:
        raise StatePredictorError("calibration requires residuals")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(confidence * len(ordered)) - 1))
    return ordered[index]


def _mae(predictions: Sequence[float], actual: Sequence[float]) -> float:
    return math.fsum(
        abs(left - right) for left, right in zip(predictions, actual, strict=True)
    ) / len(actual)


def _rmse(predictions: Sequence[float], actual: Sequence[float]) -> float:
    return math.sqrt(
        math.fsum((left - right) ** 2 for left, right in zip(predictions, actual, strict=True))
        / len(actual)
    )


@dataclass(frozen=True, slots=True)
class StatePrediction:
    target: StateTargetName
    status: StatePredictionStatus
    value: float | None
    interval_low: float | None
    interval_high: float | None
    failure_probability: float
    predictor_id: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.failure_probability <= 1.0:
            raise StatePredictorError("failure probability must be within [0, 1]")
        if self.status is StatePredictionStatus.PREDICTED:
            if self.value is None or self.interval_low is None or self.interval_high is None:
                raise StatePredictorError("predictions require a calibrated interval")
            if self.interval_low > self.value or self.value > self.interval_high:
                raise StatePredictorError("prediction interval must contain the value")
        elif (
            self.value is not None
            or self.interval_low is not None
            or self.interval_high is not None
        ):
            raise StatePredictorError("non-predicted results cannot carry numeric deltas")
        if self.status is not StatePredictionStatus.PREDICTED and not self.reason:
            raise StatePredictorError("non-predicted results require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "target": self.target.value,
            "status": self.status.value,
            "value": self.value,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "failure_probability": self.failure_probability,
            "predictor_id": self.predictor_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class StatePredictorModel:
    target: StateTargetName
    unit: str
    regressor: LinearSurgeonModel
    interval_radius: float
    failure_probability: float
    training_state_ids: tuple[str, ...]
    feature_bounds: tuple[tuple[str, float, float], ...]
    dataset_id: str
    predictor_id: str

    def __post_init__(self) -> None:
        if self.unit != _TARGET_UNITS[self.target] or self.interval_radius < 0.0:
            raise StatePredictorError("predictor model target metadata is invalid")
        if not 0.0 <= self.failure_probability <= 1.0:
            raise StatePredictorError("predictor failure probability must be within [0, 1]")
        if not self.training_state_ids:
            raise StatePredictorError("predictor model requires training state IDs")
        if len(self.feature_bounds) != len(self.regressor.feature_names):
            raise StatePredictorError("predictor feature bounds must align with regressor")

    def to_record(self) -> dict[str, object]:
        return {
            "target": self.target.value,
            "unit": self.unit,
            "regressor": self.regressor.to_record(),
            "interval_radius": self.interval_radius,
            "failure_probability": self.failure_probability,
            "training_state_ids": list(self.training_state_ids),
            "feature_bounds": [
                {"name": name, "low": low, "high": high} for name, low, high in self.feature_bounds
            ],
            "dataset_id": self.dataset_id,
            "predictor_id": self.predictor_id,
        }


@dataclass(frozen=True, slots=True)
class StatePredictorScore:
    target: StateTargetName
    outcome: StatePredictorOutcome
    model_mae: float | None
    model_rmse: float | None
    stateless_mae: float | None
    additive_mae: float | None
    failure_brier: float | None
    test_count: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.test_count < 0:
            raise StatePredictorError("score test count cannot be negative")
        for value in (
            self.model_mae,
            self.model_rmse,
            self.stateless_mae,
            self.additive_mae,
            self.failure_brier,
        ):
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise StatePredictorError("predictor scores must be finite and non-negative")

    def to_record(self) -> dict[str, object]:
        return {
            "target": self.target.value,
            "outcome": self.outcome.value,
            "model_mae": self.model_mae,
            "model_rmse": self.model_rmse,
            "stateless_mae": self.stateless_mae,
            "additive_mae": self.additive_mae,
            "failure_brier": self.failure_brier,
            "test_count": self.test_count,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class StatePredictorStudy:
    dataset_id: str
    config: StatePredictorConfig
    models: tuple[StatePredictorModel, ...]
    scores: tuple[StatePredictorScore, ...]
    schema_version: int = STATE_PREDICTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != STATE_PREDICTOR_SCHEMA_VERSION:
            raise StatePredictorError("unsupported predictor study schema")
        targets = tuple(model.target for model in self.models)
        if targets != tuple(sorted(set(targets), key=lambda item: item.value)):
            raise StatePredictorError("predictor models must be unique and canonical")

    def model_for(self, target: StateTargetName) -> StatePredictorModel | None:
        return next((model for model in self.models if model.target is target), None)

    def predict(self, embedding: StateEmbedding, target: StateTargetName) -> StatePrediction:
        model = self.model_for(target)
        if model is None:
            return StatePrediction(
                target,
                StatePredictionStatus.UNSUPPORTED,
                None,
                None,
                None,
                1.0,
                None,
                "target has no measured training evidence",
            )
        row = _feature_row_for_model(embedding, model.regressor.feature_names)
        if embedding.state_id not in model.training_state_ids:
            return StatePrediction(
                target,
                StatePredictionStatus.OUT_OF_DISTRIBUTION,
                None,
                None,
                None,
                model.failure_probability,
                model.predictor_id,
                "unseen parent state",
            )
        for value, (name, low, high) in zip(row, model.feature_bounds, strict=True):
            if value < low or value > high:
                return StatePrediction(
                    target,
                    StatePredictionStatus.OUT_OF_DISTRIBUTION,
                    None,
                    None,
                    None,
                    model.failure_probability,
                    model.predictor_id,
                    f"feature outside train range: {name}",
                )
        value = model.regressor.predict((row,))[0]
        return StatePrediction(
            target,
            StatePredictionStatus.PREDICTED,
            value,
            value - model.interval_radius,
            value + model.interval_radius,
            model.failure_probability,
            model.predictor_id,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": STATE_PREDICTOR_PROTOCOL_REVISION,
            "dataset_id": self.dataset_id,
            "config": self.config.to_record(),
            "models": [model.to_record() for model in self.models],
            "scores": [score.to_record() for score in self.scores],
        }


def _feature_row_for_model(embedding: StateEmbedding, names: Sequence[str]) -> tuple[float, ...]:
    expected_values = tuple(f"value:{name}" for name in embedding.feature_names)
    expected_masks = tuple(f"mask:{name}" for name in embedding.feature_names)
    expected = expected_values + expected_masks
    if tuple(names) == expected:
        return embedding.values + embedding.masks
    if tuple(names) == expected_values:
        return embedding.values
    raise StatePredictorError("embedding feature names are incompatible with predictor")


def _matrix(
    samples: Sequence[StatePredictionSample],
    partition: SplitPartition,
    target: StateTargetName,
    include_masks: bool,
) -> SurgeonMatrix:
    measured = tuple(sample for sample in samples if sample.target(target) is not None)
    names = _feature_names(measured[0], include_masks)
    values: list[tuple[float, ...]] = []
    targets: list[float] = []
    masks: list[bool] = []
    ids: list[str] = []
    groups: list[str] = []
    for sample in measured:
        observation = sample.target(target)
        assert observation is not None
        values.append(_feature_row(sample, include_masks))
        targets.append(0.0 if observation.value is None else observation.value)
        masks.append(observation.status is StateTargetStatus.MEASURED)
        ids.append(sample.sample_id)
        groups.append(sample.lineage_group_id)
    return SurgeonMatrix(
        partition,
        tuple(ids),
        names,
        tuple(values),
        target.value,
        tuple(targets),
        tuple(masks),
        tuple(1.0 for _ in ids),
        tuple(groups),
    )


def _model_id(target: StateTargetName, dataset_id: str, config: StatePredictorConfig) -> str:
    payload = {
        "target": target.value,
        "dataset": dataset_id,
        "config": config.to_record(),
    }
    digest = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    return f"state_predictor_{digest}"


def fit_state_predictors(
    dataset: StatePredictionDataset,
    config: StatePredictorConfig | None = None,
) -> StatePredictorStudy:
    """Fit calibrated state models and retain stateless/additive comparisons."""

    resolved = config or StatePredictorConfig()
    models: list[StatePredictorModel] = []
    scores: list[StatePredictorScore] = []
    for target in resolved.target_names:
        train_samples = tuple(
            sample
            for sample in dataset.samples
            if dataset.partition_for(sample) is SplitPartition.TRAIN
        )
        validation_samples = tuple(
            sample
            for sample in dataset.samples
            if dataset.partition_for(sample) is SplitPartition.VALIDATION
        )
        test_samples = tuple(
            sample
            for sample in dataset.samples
            if dataset.partition_for(sample) is SplitPartition.TEST
        )
        train_matrix = _matrix(train_samples, SplitPartition.TRAIN, target, resolved.include_masks)
        validation_matrix = _matrix(
            validation_samples, SplitPartition.VALIDATION, target, resolved.include_masks
        )
        if sum(train_matrix.target_mask) < resolved.minimum_train_examples:
            scores.append(
                StatePredictorScore(
                    target,
                    StatePredictorOutcome.UNSUPPORTED,
                    None,
                    None,
                    None,
                    None,
                    None,
                    0,
                    "insufficient measured training evidence",
                )
            )
            continue
        model = train_linear(
            train_matrix,
            config=LinearConfig(alpha=resolved.ridge, seed=resolved.seed),
            validation=validation_matrix,
        )
        validation_rows = tuple(
            row
            for row, observed in zip(
                validation_matrix.values, validation_matrix.target_mask, strict=True
            )
            if observed
        )
        validation_targets = tuple(
            value
            for value, observed in zip(
                validation_matrix.target_values, validation_matrix.target_mask, strict=True
            )
            if observed
        )
        interval = _quantile(
            tuple(
                abs(prediction - actual)
                for prediction, actual in zip(
                    model.predict(validation_rows), validation_targets, strict=True
                )
            ),
            resolved.confidence,
        )
        train_observed: list[tuple[StatePredictionSample, StateTargetObservation]] = []
        for sample in train_samples:
            observation = sample.target(target)
            if (
                observation is not None
                and observation.status is StateTargetStatus.MEASURED
                and observation.value is not None
            ):
                train_observed.append((sample, observation))
        train_values_float = tuple(
            float(observation.value)
            for _, observation in train_observed
            if observation.value is not None
        )
        bounds = tuple(
            (
                name,
                min(row[index] for row in train_matrix.values),
                max(row[index] for row in train_matrix.values),
            )
            for index, name in enumerate(train_matrix.feature_names)
        )
        failure_count = 0
        outcome_count = 0
        for sample in train_samples:
            observation = sample.target(target)
            if observation is not None:
                outcome_count += 1
                failure_count += int(observation.status is StateTargetStatus.FAILED)
        failure_probability = (failure_count + 1.0) / (outcome_count + 2.0)
        predictor_id = _model_id(target, dataset.dataset_id, resolved)
        models.append(
            StatePredictorModel(
                target,
                _TARGET_UNITS[target],
                model,
                interval,
                failure_probability,
                tuple(sample.parent_state_id for sample, _ in train_observed),
                bounds,
                dataset.dataset_id,
                predictor_id,
            )
        )
        test_observed: list[tuple[StatePredictionSample, StateTargetObservation]] = []
        for sample in test_samples:
            observation = sample.target(target)
            if (
                observation is not None
                and observation.status is StateTargetStatus.MEASURED
                and observation.value is not None
            ):
                test_observed.append((sample, observation))
        predictions = model.predict(
            tuple(_feature_row(sample, resolved.include_masks) for sample, _ in test_observed)
        )
        actual = tuple(
            float(observation.value)
            for _, observation in test_observed
            if observation.value is not None
        )
        if len(predictions) != len(actual) or not actual:
            scores.append(
                StatePredictorScore(
                    target,
                    StatePredictorOutcome.UNKNOWN,
                    None,
                    None,
                    None,
                    None,
                    None,
                    len(actual),
                    "no measured held-out test evidence",
                )
            )
            continue
        stateless = math.fsum(train_values_float) / len(train_values_float)
        stateless_predictions = tuple(stateless for _ in actual)
        additive_pairs = tuple(
            (prediction, float(observation.additive_baseline), value)
            for prediction, (_, observation), value in zip(
                predictions, test_observed, actual, strict=True
            )
            if observation.additive_baseline is not None
        )
        additive_mae = (
            _mae(
                tuple(item[1] for item in additive_pairs), tuple(item[2] for item in additive_pairs)
            )
            if additive_pairs
            else None
        )
        model_mae = _mae(predictions, actual)
        stateless_mae = _mae(stateless_predictions, actual)
        better_stateless = model_mae < stateless_mae
        better_additive = additive_mae is None or model_mae < additive_mae
        failure_brier = math.fsum(
            (failure_probability - float(observation.status is StateTargetStatus.FAILED)) ** 2
            for sample in test_samples
            for observation in (sample.target(target),)
            if observation is not None
        ) / max(1, len(test_samples))
        scores.append(
            StatePredictorScore(
                target,
                StatePredictorOutcome.MEASURED
                if better_stateless and better_additive
                else StatePredictorOutcome.NEGATIVE_RESULT,
                model_mae,
                _rmse(predictions, actual),
                stateless_mae,
                additive_mae,
                failure_brier,
                len(actual),
                None
                if better_stateless and better_additive
                else "state model did not beat all available baselines",
            )
        )
    return StatePredictorStudy(dataset.dataset_id, resolved, tuple(models), tuple(scores))

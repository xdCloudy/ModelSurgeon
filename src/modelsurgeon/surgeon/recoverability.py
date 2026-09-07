"""Conditional, calibrated repair-recoverability predictors."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.surgery.repair_outcome import RepairMethod

RECOVERABILITY_SCHEMA_VERSION: Final[int] = 1
RECOVERABILITY_PROTOCOL_REVISION: Final[str] = "repair-recoverability-v1"


class RecoverabilityError(ValueError):
    """Raised when recoverability evidence or inference is unsafe."""


class RecoverabilityTargetStatus(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RecoverabilityOutcome(StrEnum):
    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class RecoverabilityPredictionStatus(StrEnum):
    PREDICTED = "predicted"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class RecoverabilityDecisionStatus(StrEnum):
    REPAIR = "repair"
    NO_REPAIR = "no_repair"
    ABSTAIN = "abstain"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoverabilityError(f"{label} is required")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoverabilityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RecoverabilityError(f"{label} must be finite")
    return result


def _probability(value: object, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 1.0:
        raise RecoverabilityError(f"{label} must be within [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class RecoverabilityAction:
    """A conditional repair method and bounded budget arm."""

    method: RepairMethod
    budget_name: str
    compatible: bool = True
    compatibility_reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.budget_name, "recoverability budget name")
        if not self.compatible and not self.compatibility_reason:
            raise RecoverabilityError("incompatible actions require a compatibility reason")
        if self.compatible and self.compatibility_reason is not None:
            raise RecoverabilityError("compatible actions cannot carry a rejection reason")

    @property
    def key(self) -> tuple[str, str]:
        return self.method.value, self.budget_name

    def to_record(self) -> dict[str, object]:
        return {
            "method": self.method.value,
            "budget_name": self.budget_name,
            "compatible": self.compatible,
            "compatibility_reason": self.compatibility_reason,
        }


@dataclass(frozen=True, slots=True)
class RecoverabilityObservation:
    status: RecoverabilityTargetStatus
    quality_gain: float | None
    successful: bool | None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is RecoverabilityTargetStatus.MEASURED:
            if self.quality_gain is None or self.successful is None:
                raise RecoverabilityError(
                    "measured recoverability targets require gain and success"
                )
            if not isinstance(self.successful, bool):
                raise RecoverabilityError("recoverability success target must be boolean")
            _finite(self.quality_gain, "quality gain")
            if self.failure_reason is not None:
                raise RecoverabilityError("measured targets cannot carry a failure reason")
        else:
            if self.quality_gain is not None or self.successful is not None:
                raise RecoverabilityError("terminal recoverability targets cannot carry values")
            if not self.failure_reason:
                raise RecoverabilityError("terminal recoverability targets require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "quality_gain": self.quality_gain,
            "successful": self.successful,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class RecoverabilitySample:
    sample_id: str
    model_revision: str
    state_id: str
    candidate_id: str
    model_family: str
    lineage_group_id: str
    features: tuple[float, ...]
    action: RecoverabilityAction
    observation: RecoverabilityObservation

    def __post_init__(self) -> None:
        for label, value in (
            ("sample ID", self.sample_id),
            ("model revision", self.model_revision),
            ("state ID", self.state_id),
            ("candidate ID", self.candidate_id),
            ("model family", self.model_family),
            ("lineage group ID", self.lineage_group_id),
        ):
            _text(value, label)
        if not self.features or any(not math.isfinite(value) for value in self.features):
            raise RecoverabilityError("recoverability samples require finite features")

    def to_record(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "model_revision": self.model_revision,
            "state_id": self.state_id,
            "candidate_id": self.candidate_id,
            "model_family": self.model_family,
            "lineage_group_id": self.lineage_group_id,
            "features": list(self.features),
            "action": self.action.to_record(),
            "observation": self.observation.to_record(),
        }


@dataclass(frozen=True, slots=True)
class RecoverabilitySplit:
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
            raise RecoverabilityError("every recoverability split must contain groups")
        if any(
            values != tuple(sorted(group))
            for values, group in zip(
                (self.train_group_ids, self.validation_group_ids, self.test_group_ids),
                groups,
                strict=True,
            )
        ):
            raise RecoverabilityError("recoverability split groups must be canonical")
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise RecoverabilityError("recoverability split groups must be disjoint")

    def partition_for(self, group_id: str) -> str:
        if group_id in self.train_group_ids:
            return "train"
        if group_id in self.validation_group_ids:
            return "validation"
        if group_id in self.test_group_ids:
            return "test"
        raise RecoverabilityError("sample lineage is absent from recoverability split")

    def to_record(self) -> dict[str, object]:
        return {
            "train_group_ids": list(self.train_group_ids),
            "validation_group_ids": list(self.validation_group_ids),
            "test_group_ids": list(self.test_group_ids),
        }


@dataclass(frozen=True, slots=True)
class RecoverabilityDataset:
    samples: tuple[RecoverabilitySample, ...]
    split: RecoverabilitySplit
    feature_names: tuple[str, ...]
    dataset_revision: str
    schema_version: int = RECOVERABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERABILITY_SCHEMA_VERSION:
            raise RecoverabilityError("unsupported recoverability dataset schema")
        _text(self.dataset_revision, "recoverability dataset revision")
        if not self.feature_names or self.feature_names != tuple(sorted(set(self.feature_names))):
            raise RecoverabilityError("recoverability feature names must be canonical")
        sample_ids = tuple(item.sample_id for item in self.samples)
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise RecoverabilityError("recoverability samples must be non-empty and unique")
        partitions: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
        action_keys = {item.action.key for item in self.samples}
        if (RepairMethod.NO_REPAIR.value, "zero") not in action_keys:
            raise RecoverabilityError("recoverability datasets require a zero-budget no-repair arm")
        for sample in self.samples:
            if len(sample.features) != len(self.feature_names):
                raise RecoverabilityError("recoverability feature vectors have inconsistent width")
            partition = self.split.partition_for(sample.lineage_group_id)
            partitions[partition].update(
                {
                    f"model:{sample.model_revision}",
                    f"state:{sample.state_id}",
                    f"candidate:{sample.candidate_id}",
                }
            )
        if (
            partitions["train"] & partitions["validation"]
            or partitions["train"] & partitions["test"]
            or partitions["validation"] & partitions["test"]
        ):
            raise RecoverabilityError(
                "model, state, or candidate leakage crosses recoverability splits"
            )

    @property
    def dataset_id(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "protocol_revision": RECOVERABILITY_PROTOCOL_REVISION,
            "feature_names": list(self.feature_names),
            "dataset_revision": self.dataset_revision,
            "split": self.split.to_record(),
            "samples": [item.to_record() for item in self.samples],
        }
        digest = hashlib.sha256(_canonical(payload).encode()).hexdigest()
        return f"recoverability_dataset_{digest}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": RECOVERABILITY_PROTOCOL_REVISION,
            "dataset_id": self.dataset_id,
            "feature_names": list(self.feature_names),
            "dataset_revision": self.dataset_revision,
            "split": self.split.to_record(),
            "samples": [item.to_record() for item in self.samples],
        }


@dataclass(frozen=True, slots=True)
class RecoverabilityConfig:
    minimum_train_examples: int = 3
    confidence: float = 0.95
    ridge: float = 1e-3
    minimum_gain: float = 0.0
    minimum_success_probability: float = 0.5
    max_false_skip_risk: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        if self.minimum_train_examples < 2:
            raise RecoverabilityError("recoverability training minimum must be at least two")
        if not 0.5 < self.confidence < 1.0 or not math.isfinite(self.confidence):
            raise RecoverabilityError("recoverability confidence must be in (0.5, 1)")
        if self.ridge < 0 or not math.isfinite(self.ridge):
            raise RecoverabilityError("recoverability ridge must be finite and non-negative")
        if self.minimum_gain < 0 or not math.isfinite(self.minimum_gain):
            raise RecoverabilityError("minimum recoverability gain must be finite and non-negative")
        _probability(self.minimum_success_probability, "minimum success probability")
        _probability(self.max_false_skip_risk, "maximum false-skip risk")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64:
            raise RecoverabilityError("recoverability seed must be unsigned 64-bit")

    def to_record(self) -> dict[str, object]:
        return {
            "minimum_train_examples": self.minimum_train_examples,
            "confidence": self.confidence,
            "ridge": self.ridge,
            "minimum_gain": self.minimum_gain,
            "minimum_success_probability": self.minimum_success_probability,
            "max_false_skip_risk": self.max_false_skip_risk,
            "seed": self.seed,
            "model_family": "conditional_linear_ridge",
            "calibration": "validation_absolute_residual_quantile_and_laplace_success",
            "baselines": ["global_action_mean", "damage_only_feature_subset"],
        }


def _quantile(values: Sequence[float], confidence: float) -> float:
    if not values:
        raise RecoverabilityError("recoverability calibration requires residuals")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(confidence * len(ordered)) - 1))
    return ordered[index]


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise RecoverabilityError("recoverability mean requires values")
    return math.fsum(values) / len(values)


def _mae(predicted: Sequence[float], actual: Sequence[float]) -> float:
    return _mean(tuple(abs(left - right) for left, right in zip(predicted, actual, strict=True)))


def _normalization(
    rows: Sequence[tuple[float, ...]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    width = len(rows[0])
    means = tuple(_mean(tuple(row[index] for row in rows)) for index in range(width))
    scales = tuple(
        max(
            1e-9,
            math.sqrt(_mean(tuple((row[index] - means[index]) ** 2 for row in rows))),
        )
        for index in range(width)
    )
    return means, scales


def _normalize(
    row: tuple[float, ...], means: tuple[float, ...], scales: tuple[float, ...]
) -> tuple[float, ...]:
    return tuple(
        (value - mean) / scale for value, mean, scale in zip(row, means, scales, strict=True)
    )


def _fit_linear(
    rows: Sequence[tuple[float, ...]], targets: Sequence[float], ridge: float
) -> tuple[tuple[float, ...], float, tuple[float, ...], tuple[float, ...]]:
    means, scales = _normalization(rows)
    normalized = tuple(_normalize(row, means, scales) for row in rows)
    intercept = _mean(targets)
    coefficients = [0.0] * len(means)
    for step in range(400):
        predictions = tuple(
            intercept
            + math.fsum(
                coefficient * value for coefficient, value in zip(coefficients, row, strict=True)
            )
            for row in normalized
        )
        errors = tuple(
            prediction - target for prediction, target in zip(predictions, targets, strict=True)
        )
        learning_rate = 0.08 / (1.0 + step * 0.01)
        intercept -= learning_rate * _mean(errors)
        for index in range(len(coefficients)):
            gradient = math.fsum(
                error * row[index] for error, row in zip(errors, normalized, strict=True)
            )
            gradient = gradient / len(rows) + ridge * coefficients[index]
            coefficients[index] -= learning_rate * gradient
    return tuple(coefficients), intercept, means, scales


@dataclass(frozen=True, slots=True)
class RecoverabilityPrediction:
    action: RecoverabilityAction
    status: RecoverabilityPredictionStatus
    quality_gain: float | None
    interval_low: float | None
    interval_high: float | None
    success_probability: float | None
    predictor_id: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is RecoverabilityPredictionStatus.PREDICTED:
            if any(
                value is None
                for value in (
                    self.quality_gain,
                    self.interval_low,
                    self.interval_high,
                    self.success_probability,
                )
            ):
                raise RecoverabilityError(
                    "predictions require gain, interval, and success evidence"
                )
            assert self.quality_gain is not None
            assert self.interval_low is not None
            assert self.interval_high is not None
            if self.interval_low > self.quality_gain or self.quality_gain > self.interval_high:
                raise RecoverabilityError("recoverability interval must contain its prediction")
            _probability(self.success_probability, "predicted success probability")
        else:
            if any(
                value is not None
                for value in (self.quality_gain, self.interval_low, self.interval_high)
            ):
                raise RecoverabilityError("non-predicted recoverability cannot carry gain values")
            if self.success_probability is not None:
                _probability(self.success_probability, "non-predicted success probability")
            if not self.reason:
                raise RecoverabilityError("non-predicted recoverability requires a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "action": self.action.to_record(),
            "status": self.status.value,
            "quality_gain": self.quality_gain,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "success_probability": self.success_probability,
            "predictor_id": self.predictor_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RecoverabilityPredictorModel:
    action: RecoverabilityAction
    coefficients: tuple[float, ...]
    intercept: float
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    interval_radius: float
    success_probability: float
    training_state_ids: tuple[str, ...]
    feature_bounds: tuple[tuple[float, float], ...]
    dataset_id: str
    predictor_id: str

    def __post_init__(self) -> None:
        if len(self.coefficients) != len(self.feature_means) or len(self.coefficients) != len(
            self.feature_scales
        ):
            raise RecoverabilityError("recoverability predictor vectors do not align")
        if len(self.coefficients) != len(self.feature_bounds):
            raise RecoverabilityError("recoverability feature bounds do not align")
        if not self.training_state_ids:
            raise RecoverabilityError("recoverability predictors require training states")
        values = (
            *self.coefficients,
            self.intercept,
            *self.feature_means,
            *self.feature_scales,
            self.interval_radius,
        )
        if any(not math.isfinite(value) for value in values) or any(
            scale <= 0 for scale in self.feature_scales
        ):
            raise RecoverabilityError("recoverability predictor parameters must be finite")
        if self.interval_radius < 0:
            raise RecoverabilityError("recoverability predictor interval must be non-negative")
        _probability(self.success_probability, "recoverability predictor success probability")

    def _identity_record(self) -> dict[str, object]:
        return {
            "action": self.action.to_record(),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "interval_radius": self.interval_radius,
            "success_probability": self.success_probability,
            "training_state_ids": list(self.training_state_ids),
            "feature_bounds": [list(item) for item in self.feature_bounds],
            "dataset_id": self.dataset_id,
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["predictor_id"] = self.predictor_id
        return record


@dataclass(frozen=True, slots=True)
class RecoverabilityDecision:
    status: RecoverabilityDecisionStatus
    selected_action: RecoverabilityAction | None
    prediction: RecoverabilityPrediction | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status in {
            RecoverabilityDecisionStatus.REPAIR,
            RecoverabilityDecisionStatus.NO_REPAIR,
        }:
            if self.selected_action is None or self.prediction is None:
                raise RecoverabilityError(
                    "selected recoverability decisions require action evidence"
                )
        elif self.selected_action is not None or self.prediction is not None:
            raise RecoverabilityError(
                "unselected recoverability decisions cannot carry action evidence"
            )
        if self.status is not RecoverabilityDecisionStatus.REPAIR and not self.reason:
            raise RecoverabilityError("non-repair decisions require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "selected_action": None
            if self.selected_action is None
            else self.selected_action.to_record(),
            "prediction": None if self.prediction is None else self.prediction.to_record(),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RecoverabilityScore:
    action: RecoverabilityAction
    outcome: RecoverabilityOutcome
    model_mae: float | None
    mean_baseline_mae: float | None
    success_brier: float | None
    validation_count: int
    test_count: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if min(self.validation_count, self.test_count) < 0:
            raise RecoverabilityError("recoverability score counts cannot be negative")
        for value in (self.model_mae, self.mean_baseline_mae, self.success_brier):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise RecoverabilityError("recoverability scores must be finite and non-negative")
        if self.outcome is not RecoverabilityOutcome.MEASURED and not self.reason:
            raise RecoverabilityError("non-measured recoverability scores require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "action": self.action.to_record(),
            "outcome": self.outcome.value,
            "model_mae": self.model_mae,
            "mean_baseline_mae": self.mean_baseline_mae,
            "success_brier": self.success_brier,
            "validation_count": self.validation_count,
            "test_count": self.test_count,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RecoverabilityStudy:
    dataset_id: str
    config: RecoverabilityConfig
    models: tuple[RecoverabilityPredictorModel, ...]
    scores: tuple[RecoverabilityScore, ...]
    schema_version: int = RECOVERABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECOVERABILITY_SCHEMA_VERSION:
            raise RecoverabilityError("unsupported recoverability study schema")
        keys = tuple(item.action.key for item in self.models)
        if keys != tuple(sorted(set(keys))):
            raise RecoverabilityError("recoverability models must be unique and canonical")

    def model_for(self, action: RecoverabilityAction) -> RecoverabilityPredictorModel | None:
        return next((item for item in self.models if item.action.key == action.key), None)

    def _prediction(
        self,
        model: RecoverabilityPredictorModel | None,
        action: RecoverabilityAction,
        features: tuple[float, ...],
        state_id: str,
    ) -> RecoverabilityPrediction:
        if not action.compatible:
            return RecoverabilityPrediction(
                action,
                RecoverabilityPredictionStatus.UNSUPPORTED,
                None,
                None,
                None,
                None,
                None,
                action.compatibility_reason,
            )
        if model is None:
            return RecoverabilityPrediction(
                action,
                RecoverabilityPredictionStatus.UNSUPPORTED,
                None,
                None,
                None,
                None,
                None,
                "action has no measured training evidence",
            )
        if state_id not in model.training_state_ids:
            return RecoverabilityPrediction(
                action,
                RecoverabilityPredictionStatus.OUT_OF_DISTRIBUTION,
                None,
                None,
                None,
                model.success_probability,
                model.predictor_id,
                "unseen parent state",
            )
        if any(
            value < low or value > high
            for value, (low, high) in zip(features, model.feature_bounds, strict=True)
        ):
            return RecoverabilityPrediction(
                action,
                RecoverabilityPredictionStatus.OUT_OF_DISTRIBUTION,
                None,
                None,
                None,
                model.success_probability,
                model.predictor_id,
                "feature outside train range",
            )
        normalized = _normalize(features, model.feature_means, model.feature_scales)
        value = model.intercept + math.fsum(
            coefficient * feature
            for coefficient, feature in zip(model.coefficients, normalized, strict=True)
        )
        return RecoverabilityPrediction(
            action,
            RecoverabilityPredictionStatus.PREDICTED,
            value,
            value - model.interval_radius,
            value + model.interval_radius,
            model.success_probability,
            model.predictor_id,
        )

    def predict_actions(
        self,
        features: tuple[float, ...],
        state_id: str,
        actions: tuple[RecoverabilityAction, ...],
    ) -> tuple[RecoverabilityPrediction, ...]:
        if not features or any(not math.isfinite(value) for value in features):
            raise RecoverabilityError("recoverability inference features must be finite")
        if not actions:
            raise RecoverabilityError("recoverability inference requires actions")
        return tuple(
            self._prediction(self.model_for(action), action, features, state_id)
            for action in sorted(actions, key=lambda item: item.key)
        )

    def decide(
        self,
        features: tuple[float, ...],
        state_id: str,
        actions: tuple[RecoverabilityAction, ...],
    ) -> RecoverabilityDecision:
        predictions = self.predict_actions(features, state_id, actions)
        no_repair = next(
            (item for item in predictions if item.action.method is RepairMethod.NO_REPAIR),
            None,
        )
        predicted = tuple(
            item
            for item in predictions
            if item.status is RecoverabilityPredictionStatus.PREDICTED
            and item.action.method is not RepairMethod.NO_REPAIR
            and item.quality_gain is not None
            and item.interval_low is not None
            and item.success_probability is not None
            and item.interval_low >= self.config.minimum_gain
            and item.success_probability >= self.config.minimum_success_probability
        )
        if predicted:
            selected = max(
                predicted,
                key=lambda item: (
                    item.quality_gain or -math.inf,
                    item.success_probability or -math.inf,
                    item.action.key,
                ),
            )
            return RecoverabilityDecision(
                RecoverabilityDecisionStatus.REPAIR,
                selected.action,
                selected,
            )
        if no_repair is not None and no_repair.status is RecoverabilityPredictionStatus.PREDICTED:
            return RecoverabilityDecision(
                RecoverabilityDecisionStatus.NO_REPAIR,
                no_repair.action,
                no_repair,
                "no compatible repair clears the conservative gain and success gates",
            )
        if any(
            item.status is RecoverabilityPredictionStatus.OUT_OF_DISTRIBUTION
            for item in predictions
        ):
            return RecoverabilityDecision(
                RecoverabilityDecisionStatus.OUT_OF_DISTRIBUTION,
                None,
                None,
                "recoverability features or state are outside training support",
            )
        if any(item.status is RecoverabilityPredictionStatus.UNSUPPORTED for item in predictions):
            return RecoverabilityDecision(
                RecoverabilityDecisionStatus.UNSUPPORTED,
                None,
                None,
                "no-repair or compatible repair evidence is unavailable",
            )
        return RecoverabilityDecision(
            RecoverabilityDecisionStatus.UNKNOWN,
            None,
            None,
            "recoverability evidence is inconclusive",
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": RECOVERABILITY_PROTOCOL_REVISION,
            "dataset_id": self.dataset_id,
            "config": self.config.to_record(),
            "models": [item.to_record() for item in self.models],
            "scores": [item.to_record() for item in self.scores],
        }


def _fit_action(
    dataset: RecoverabilityDataset,
    action: RecoverabilityAction,
    config: RecoverabilityConfig,
) -> tuple[RecoverabilityPredictorModel | None, RecoverabilityScore]:
    train = tuple(
        item
        for item in dataset.samples
        if dataset.split.partition_for(item.lineage_group_id) == "train"
        and item.action.key == action.key
        and item.observation.status is RecoverabilityTargetStatus.MEASURED
    )
    validation = tuple(
        item
        for item in dataset.samples
        if dataset.split.partition_for(item.lineage_group_id) == "validation"
        and item.action.key == action.key
        and item.observation.status is RecoverabilityTargetStatus.MEASURED
    )
    test = tuple(
        item
        for item in dataset.samples
        if dataset.split.partition_for(item.lineage_group_id) == "test"
        and item.action.key == action.key
        and item.observation.status is RecoverabilityTargetStatus.MEASURED
    )
    if len(train) < config.minimum_train_examples:
        return None, RecoverabilityScore(
            action,
            RecoverabilityOutcome.UNSUPPORTED,
            None,
            None,
            None,
            len(validation),
            len(test),
            "insufficient measured training evidence",
        )
    if not validation:
        return None, RecoverabilityScore(
            action,
            RecoverabilityOutcome.UNKNOWN,
            None,
            None,
            None,
            0,
            len(test),
            "validation evidence is required for calibrated intervals",
        )
    rows = tuple(item.features for item in train)
    targets = tuple(item.observation.quality_gain for item in train)
    assert all(value is not None for value in targets)
    numeric_targets = tuple(value for value in targets if value is not None)
    coefficients, intercept, means, scales = _fit_linear(rows, numeric_targets, config.ridge)
    train_states = tuple(sorted({item.state_id for item in train}))
    bounds = tuple(
        (min(row[index] for row in rows), max(row[index] for row in rows))
        for index in range(len(dataset.feature_names))
    )
    validation_predictions = tuple(
        intercept
        + math.fsum(
            coefficient * value
            for coefficient, value in zip(
                coefficients,
                _normalize(item.features, means, scales),
                strict=True,
            )
        )
        for item in validation
    )
    validation_actual = tuple(
        value
        for item in validation
        for value in (item.observation.quality_gain,)
        if value is not None
    )
    radius = _quantile(
        tuple(
            abs(prediction - actual)
            for prediction, actual in zip(validation_predictions, validation_actual, strict=True)
        ),
        config.confidence,
    )
    successes = sum(bool(item.observation.successful) for item in train)
    success_probability = (successes + 1.0) / (len(train) + 2.0)
    predictor_identity = {
        "action": action.to_record(),
        "dataset_id": dataset.dataset_id,
        "config": config.to_record(),
        "coefficients": list(coefficients),
        "intercept": intercept,
        "feature_means": list(means),
        "feature_scales": list(scales),
        "interval_radius": radius,
        "success_probability": success_probability,
        "training_state_ids": list(train_states),
        "feature_bounds": [list(item) for item in bounds],
    }
    predictor_id = (
        "recoverability_predictor_"
        + hashlib.sha256(_canonical(predictor_identity).encode()).hexdigest()
    )
    model = RecoverabilityPredictorModel(
        action,
        coefficients,
        intercept,
        means,
        scales,
        radius,
        success_probability,
        train_states,
        bounds,
        dataset.dataset_id,
        predictor_id,
    )
    test_actual = tuple(
        value for item in test for value in (item.observation.quality_gain,) if value is not None
    )
    test_predictions = tuple(
        intercept
        + math.fsum(
            coefficient * value
            for coefficient, value in zip(
                coefficients,
                _normalize(item.features, means, scales),
                strict=True,
            )
        )
        for item in test
    )
    mean_baseline = _mean(numeric_targets)
    model_mae = _mae(test_predictions, test_actual) if test_actual else None
    baseline_mae = (
        _mae(tuple(mean_baseline for _ in test_actual), test_actual) if test_actual else None
    )
    success_brier = (
        _mean(
            tuple(
                (success_probability - float(success)) ** 2
                for item in test
                for success in (item.observation.successful,)
                if success is not None
            )
        )
        if test
        else None
    )
    if model_mae is None:
        outcome = RecoverabilityOutcome.UNKNOWN
        reason = "no measured held-out test evidence"
    elif model_mae < (baseline_mae or math.inf):
        outcome = RecoverabilityOutcome.MEASURED
        reason = None
    else:
        outcome = RecoverabilityOutcome.NEGATIVE_RESULT
        reason = "conditional model did not beat the global action mean"
    return model, RecoverabilityScore(
        action,
        outcome,
        model_mae,
        baseline_mae,
        success_brier,
        len(validation),
        len(test),
        reason,
    )


def fit_recoverability_predictors(
    dataset: RecoverabilityDataset,
    config: RecoverabilityConfig | None = None,
) -> RecoverabilityStudy:
    """Fit action-conditional gain models and retain no-repair/negative evidence."""

    resolved = config or RecoverabilityConfig()
    actions = tuple(sorted({item.action for item in dataset.samples}, key=lambda item: item.key))
    models: list[RecoverabilityPredictorModel] = []
    scores: list[RecoverabilityScore] = []
    for action in actions:
        model, score = _fit_action(dataset, action, resolved)
        if model is not None:
            models.append(model)
        scores.append(score)
    return RecoverabilityStudy(dataset.dataset_id, resolved, tuple(models), tuple(scores))

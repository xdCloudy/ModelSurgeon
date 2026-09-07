"""Calibrated resource-cost predictors and conservative repair preflight."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.surgeon.recoverability import (
    RecoverabilityAction,
    RecoverabilitySplit,
    _fit_linear,
    _normalize,
    _quantile,
)
from modelsurgeon.surgery.repair_outcome import RepairMethod

REPAIR_COST_SCHEMA_VERSION: Final[int] = 1
REPAIR_COST_PROTOCOL_REVISION: Final[str] = "repair-cost-predictor-v1"


class RepairCostPredictorError(ValueError):
    """Raised when resource evidence or budget preflight is unsafe."""


class RepairCostTarget(StrEnum):
    TOKENS = "tokens"
    WALL_SECONDS = "wall_seconds"
    GPU_SECONDS = "gpu_seconds"
    CPU_SECONDS = "cpu_seconds"
    PEAK_RAM_BYTES = "peak_ram_bytes"
    PEAK_VRAM_BYTES = "peak_vram_bytes"
    ARTIFACT_BYTES = "artifact_bytes"
    CACHE_BYTES = "cache_bytes"
    ENERGY_JOULES = "energy_joules"


class RepairCostObservationStatus(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class RepairCostPredictionStatus(StrEnum):
    PREDICTED = "predicted"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class RepairCostPreflightStatus(StrEnum):
    ALLOWED = "allowed"
    OVER_BUDGET = "over_budget"
    UNSUPPORTED = "unsupported"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    UNKNOWN = "unknown"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RepairCostPredictorError(f"{label} is required")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RepairCostPredictorError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RepairCostPredictorError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class RepairCostLimits:
    """Hard resource limits applied to conservative upper bounds."""

    max_tokens: float
    max_wall_seconds: float
    max_gpu_seconds: float
    max_cpu_seconds: float
    max_peak_ram_bytes: float
    max_peak_vram_bytes: float
    max_artifact_bytes: float
    max_cache_bytes: float
    max_energy_joules: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.max_tokens,
            self.max_wall_seconds,
            self.max_gpu_seconds,
            self.max_cpu_seconds,
            self.max_peak_ram_bytes,
            self.max_peak_vram_bytes,
            self.max_artifact_bytes,
            self.max_cache_bytes,
        )
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise RepairCostPredictorError("repair cost limits must be positive and finite")
        if self.max_energy_joules is not None and (
            not math.isfinite(self.max_energy_joules) or self.max_energy_joules <= 0
        ):
            raise RepairCostPredictorError("energy limit must be positive and finite")

    def limit_for(self, target: RepairCostTarget) -> float | None:
        return {
            RepairCostTarget.TOKENS: self.max_tokens,
            RepairCostTarget.WALL_SECONDS: self.max_wall_seconds,
            RepairCostTarget.GPU_SECONDS: self.max_gpu_seconds,
            RepairCostTarget.CPU_SECONDS: self.max_cpu_seconds,
            RepairCostTarget.PEAK_RAM_BYTES: self.max_peak_ram_bytes,
            RepairCostTarget.PEAK_VRAM_BYTES: self.max_peak_vram_bytes,
            RepairCostTarget.ARTIFACT_BYTES: self.max_artifact_bytes,
            RepairCostTarget.CACHE_BYTES: self.max_cache_bytes,
            RepairCostTarget.ENERGY_JOULES: self.max_energy_joules,
        }[target]

    def to_record(self) -> dict[str, object]:
        return {target.value: self.limit_for(target) for target in RepairCostTarget}


@dataclass(frozen=True, slots=True)
class RepairCostObservation:
    status: RepairCostObservationStatus
    values: tuple[tuple[RepairCostTarget, float], ...] = ()
    reason: str | None = None

    def __post_init__(self) -> None:
        names = tuple(target for target, _ in self.values)
        if names != tuple(sorted(set(names), key=lambda item: item.value)):
            raise RepairCostPredictorError("repair cost values must be unique and canonical")
        for target, value in self.values:
            numeric = _finite(value, f"{target.value} cost")
            if numeric < 0:
                raise RepairCostPredictorError("repair cost values cannot be negative")
        if self.status is RepairCostObservationStatus.MEASURED:
            if not self.values:
                raise RepairCostPredictorError("measured repair costs require values")
            if self.reason is not None:
                raise RepairCostPredictorError("measured repair costs cannot carry a reason")
        elif self.values or not self.reason:
            raise RepairCostPredictorError(
                "terminal repair costs require a reason and cannot carry values"
            )

    def value_for(self, target: RepairCostTarget) -> float | None:
        return next((value for name, value in self.values if name is target), None)

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "values": {target.value: value for target, value in self.values},
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RepairCostSample:
    sample_id: str
    model_revision: str
    hardware_profile_id: str
    state_id: str
    candidate_id: str
    lineage_group_id: str
    features: tuple[float, ...]
    action: RecoverabilityAction
    observation: RepairCostObservation

    def __post_init__(self) -> None:
        for label, value in (
            ("repair cost sample ID", self.sample_id),
            ("model revision", self.model_revision),
            ("hardware profile ID", self.hardware_profile_id),
            ("state ID", self.state_id),
            ("candidate ID", self.candidate_id),
            ("lineage group ID", self.lineage_group_id),
        ):
            _text(value, label)
        if not self.features or any(not math.isfinite(value) for value in self.features):
            raise RepairCostPredictorError("repair cost samples require finite features")

    def to_record(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "model_revision": self.model_revision,
            "hardware_profile_id": self.hardware_profile_id,
            "state_id": self.state_id,
            "candidate_id": self.candidate_id,
            "lineage_group_id": self.lineage_group_id,
            "features": list(self.features),
            "action": self.action.to_record(),
            "observation": self.observation.to_record(),
        }


@dataclass(frozen=True, slots=True)
class RepairCostDataset:
    samples: tuple[RepairCostSample, ...]
    split: RecoverabilitySplit
    feature_names: tuple[str, ...]
    dataset_revision: str
    schema_version: int = REPAIR_COST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPAIR_COST_SCHEMA_VERSION:
            raise RepairCostPredictorError("unsupported repair cost dataset schema")
        _text(self.dataset_revision, "repair cost dataset revision")
        if not self.feature_names or self.feature_names != tuple(sorted(set(self.feature_names))):
            raise RepairCostPredictorError("repair cost feature names must be canonical")
        sample_ids = tuple(item.sample_id for item in self.samples)
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise RepairCostPredictorError("repair cost samples must be non-empty and unique")
        actions = {item.action.key for item in self.samples}
        if (RepairMethod.NO_REPAIR.value, "zero") not in actions:
            raise RepairCostPredictorError(
                "repair cost datasets require a zero-budget no-repair arm"
            )
        partitions: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
        for sample in self.samples:
            if len(sample.features) != len(self.feature_names):
                raise RepairCostPredictorError(
                    "repair cost feature vectors have inconsistent width"
                )
            partition = self.split.partition_for(sample.lineage_group_id)
            partitions[partition].update(
                {
                    f"model:{sample.model_revision}",
                    f"hardware:{sample.hardware_profile_id}",
                    f"state:{sample.state_id}",
                    f"candidate:{sample.candidate_id}",
                }
            )
        if (
            partitions["train"] & partitions["validation"]
            or partitions["train"] & partitions["test"]
            or partitions["validation"] & partitions["test"]
        ):
            raise RepairCostPredictorError(
                "model, hardware, state, or candidate leakage crosses splits"
            )

    @property
    def dataset_id(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "protocol_revision": REPAIR_COST_PROTOCOL_REVISION,
            "dataset_revision": self.dataset_revision,
            "feature_names": list(self.feature_names),
            "split": self.split.to_record(),
            "samples": [item.to_record() for item in self.samples],
        }
        return "repair_cost_dataset_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": REPAIR_COST_PROTOCOL_REVISION,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "feature_names": list(self.feature_names),
            "split": self.split.to_record(),
            "samples": [item.to_record() for item in self.samples],
        }


@dataclass(frozen=True, slots=True)
class RepairCostConfig:
    targets: tuple[RepairCostTarget, ...] = tuple(
        sorted(RepairCostTarget, key=lambda item: item.value)
    )
    minimum_train_examples: int = 3
    confidence: float = 0.95
    ridge: float = 1e-3

    def __post_init__(self) -> None:
        if not self.targets or self.targets != tuple(
            sorted(set(self.targets), key=lambda item: item.value)
        ):
            raise RepairCostPredictorError("repair cost targets must be unique and canonical")
        if self.minimum_train_examples < 2:
            raise RepairCostPredictorError("repair cost training minimum must be at least two")
        if not 0.5 < self.confidence < 1.0 or not math.isfinite(self.confidence):
            raise RepairCostPredictorError("repair cost confidence must be in (0.5, 1)")
        if self.ridge < 0 or not math.isfinite(self.ridge):
            raise RepairCostPredictorError("repair cost ridge must be finite and non-negative")

    def to_record(self) -> dict[str, object]:
        return {
            "targets": [item.value for item in self.targets],
            "minimum_train_examples": self.minimum_train_examples,
            "confidence": self.confidence,
            "ridge": self.ridge,
            "model_family": "conditional_linear_ridge",
            "calibration": "validation_absolute_residual_quantile",
            "baselines": ["analytic_lower_bound", "action_mean"],
        }


@dataclass(frozen=True, slots=True)
class RepairCostPrediction:
    action: RecoverabilityAction
    target: RepairCostTarget
    status: RepairCostPredictionStatus
    upper_bound: float | None
    point: float | None
    interval_low: float | None
    predictor_id: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is RepairCostPredictionStatus.PREDICTED:
            if any(value is None for value in (self.upper_bound, self.point, self.interval_low)):
                raise RepairCostPredictorError(
                    "cost predictions require point and interval evidence"
                )
            assert self.upper_bound is not None
            assert self.point is not None
            assert self.interval_low is not None
            if self.interval_low > self.point or self.point > self.upper_bound:
                raise RepairCostPredictorError("cost interval is not ordered")
            if self.interval_low < 0:
                raise RepairCostPredictorError("cost intervals cannot be negative")
        else:
            if any(
                value is not None for value in (self.upper_bound, self.point, self.interval_low)
            ):
                raise RepairCostPredictorError("non-predicted costs cannot carry numeric values")
            if not self.reason:
                raise RepairCostPredictorError("non-predicted costs require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "action": self.action.to_record(),
            "target": self.target.value,
            "status": self.status.value,
            "upper_bound": self.upper_bound,
            "point": self.point,
            "interval_low": self.interval_low,
            "predictor_id": self.predictor_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RepairCostModel:
    action: RecoverabilityAction
    target: RepairCostTarget
    coefficients: tuple[float, ...]
    intercept: float
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    interval_radius: float
    training_state_ids: tuple[str, ...]
    feature_bounds: tuple[tuple[float, float], ...]
    dataset_id: str
    predictor_id: str

    def __post_init__(self) -> None:
        lengths = {
            len(self.coefficients),
            len(self.feature_means),
            len(self.feature_scales),
            len(self.feature_bounds),
        }
        if len(lengths) != 1 or not self.training_state_ids:
            raise RepairCostPredictorError(
                "repair cost model vectors or training states are invalid"
            )
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
            raise RepairCostPredictorError("repair cost model parameters must be finite")
        if self.interval_radius < 0:
            raise RepairCostPredictorError("repair cost interval must be non-negative")

    def _identity_record(self) -> dict[str, object]:
        return {
            "action": self.action.to_record(),
            "target": self.target.value,
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "interval_radius": self.interval_radius,
            "training_state_ids": list(self.training_state_ids),
            "feature_bounds": [list(item) for item in self.feature_bounds],
            "dataset_id": self.dataset_id,
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["predictor_id"] = self.predictor_id
        return record


@dataclass(frozen=True, slots=True)
class RepairCostScore:
    action: RecoverabilityAction
    target: RepairCostTarget
    outcome: RepairCostObservationStatus
    model_mae: float | None
    mean_baseline_mae: float | None
    validation_count: int
    test_count: int
    reason: str | None = None

    def __post_init__(self) -> None:
        if min(self.validation_count, self.test_count) < 0:
            raise RepairCostPredictorError("repair cost score counts cannot be negative")
        for value in (self.model_mae, self.mean_baseline_mae):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise RepairCostPredictorError("repair cost scores must be finite and non-negative")
        if self.outcome is not RepairCostObservationStatus.MEASURED and not self.reason:
            raise RepairCostPredictorError("non-measured repair cost scores require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "action": self.action.to_record(),
            "target": self.target.value,
            "outcome": self.outcome.value,
            "model_mae": self.model_mae,
            "mean_baseline_mae": self.mean_baseline_mae,
            "validation_count": self.validation_count,
            "test_count": self.test_count,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RepairCostPreflight:
    status: RepairCostPreflightStatus
    predictions: tuple[RepairCostPrediction, ...]
    exceeded_targets: tuple[RepairCostTarget, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is RepairCostPreflightStatus.OVER_BUDGET and not self.exceeded_targets:
            raise RepairCostPredictorError("over-budget preflight requires exceeded targets")
        if self.status is not RepairCostPreflightStatus.OVER_BUDGET and self.exceeded_targets:
            raise RepairCostPredictorError("only over-budget preflight can carry exceeded targets")
        if self.status is not RepairCostPreflightStatus.ALLOWED and not self.reason:
            raise RepairCostPredictorError("non-allowed preflight requires a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "predictions": [item.to_record() for item in self.predictions],
            "exceeded_targets": [item.value for item in self.exceeded_targets],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RepairCostStudy:
    dataset_id: str
    config: RepairCostConfig
    models: tuple[RepairCostModel, ...]
    scores: tuple[RepairCostScore, ...]
    schema_version: int = REPAIR_COST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPAIR_COST_SCHEMA_VERSION:
            raise RepairCostPredictorError("unsupported repair cost study schema")
        keys = tuple((item.action.key, item.target.value) for item in self.models)
        if keys != tuple(sorted(set(keys))):
            raise RepairCostPredictorError("repair cost models must be unique and canonical")

    def model_for(
        self, action: RecoverabilityAction, target: RepairCostTarget
    ) -> RepairCostModel | None:
        return next(
            (
                item
                for item in self.models
                if item.action.key == action.key and item.target is target
            ),
            None,
        )

    def predict(
        self,
        features: tuple[float, ...],
        state_id: str,
        actions: tuple[RecoverabilityAction, ...],
    ) -> tuple[RepairCostPrediction, ...]:
        if not features or any(not math.isfinite(value) for value in features):
            raise RepairCostPredictorError("repair cost features must be finite")
        predictions: list[RepairCostPrediction] = []
        for action in sorted(actions, key=lambda item: item.key):
            for target in self.config.targets:
                model = self.model_for(action, target)
                if not action.compatible:
                    predictions.append(
                        RepairCostPrediction(
                            action,
                            target,
                            RepairCostPredictionStatus.UNSUPPORTED,
                            None,
                            None,
                            None,
                            None,
                            action.compatibility_reason,
                        )
                    )
                    continue
                if model is None:
                    predictions.append(
                        RepairCostPrediction(
                            action,
                            target,
                            RepairCostPredictionStatus.UNSUPPORTED,
                            None,
                            None,
                            None,
                            None,
                            "target has no measured training evidence",
                        )
                    )
                    continue
                if state_id not in model.training_state_ids:
                    predictions.append(
                        RepairCostPrediction(
                            action,
                            target,
                            RepairCostPredictionStatus.OUT_OF_DISTRIBUTION,
                            None,
                            None,
                            None,
                            model.predictor_id,
                            "unseen parent state",
                        )
                    )
                    continue
                if any(
                    value < low or value > high
                    for value, (low, high) in zip(features, model.feature_bounds, strict=True)
                ):
                    predictions.append(
                        RepairCostPrediction(
                            action,
                            target,
                            RepairCostPredictionStatus.OUT_OF_DISTRIBUTION,
                            None,
                            None,
                            None,
                            model.predictor_id,
                            "feature outside train range",
                        )
                    )
                    continue
                point = max(
                    0.0,
                    model.intercept
                    + math.fsum(
                        coefficient * value
                        for coefficient, value in zip(
                            model.coefficients,
                            _normalize(features, model.feature_means, model.feature_scales),
                            strict=True,
                        )
                    ),
                )
                predictions.append(
                    RepairCostPrediction(
                        action,
                        target,
                        RepairCostPredictionStatus.PREDICTED,
                        point + model.interval_radius,
                        point,
                        max(0.0, point - model.interval_radius),
                        model.predictor_id,
                    )
                )
        return tuple(predictions)

    def preflight(
        self,
        features: tuple[float, ...],
        state_id: str,
        action: RecoverabilityAction,
        limits: RepairCostLimits,
    ) -> RepairCostPreflight:
        predictions = tuple(
            item
            for item in self.predict(features, state_id, (action,))
            if item.action.key == action.key
        )
        if any(
            item.status is RepairCostPredictionStatus.OUT_OF_DISTRIBUTION for item in predictions
        ):
            return RepairCostPreflight(
                RepairCostPreflightStatus.OUT_OF_DISTRIBUTION,
                predictions,
                (),
                "cost action is outside training support",
            )
        if any(
            item.status is RepairCostPredictionStatus.UNSUPPORTED
            and item.target is not RepairCostTarget.ENERGY_JOULES
            for item in predictions
        ):
            return RepairCostPreflight(
                RepairCostPreflightStatus.UNSUPPORTED,
                predictions,
                (),
                "one or more required cost targets are unavailable",
            )
        exceeded_values: list[RepairCostTarget] = []
        for item in predictions:
            limit = limits.limit_for(item.target)
            if item.upper_bound is not None and limit is not None and item.upper_bound > limit:
                exceeded_values.append(item.target)
        exceeded = tuple(exceeded_values)
        if exceeded:
            return RepairCostPreflight(
                RepairCostPreflightStatus.OVER_BUDGET,
                predictions,
                exceeded,
                "conservative cost bound exceeds a hard resource limit",
            )
        return RepairCostPreflight(RepairCostPreflightStatus.ALLOWED, predictions, ())

    @property
    def study_id(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "protocol_revision": REPAIR_COST_PROTOCOL_REVISION,
            "dataset_id": self.dataset_id,
            "config": self.config.to_record(),
            "models": [item.to_record() for item in self.models],
            "scores": [item.to_record() for item in self.scores],
        }
        return "repair_cost_study_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": REPAIR_COST_PROTOCOL_REVISION,
            "study_id": self.study_id,
            "dataset_id": self.dataset_id,
            "config": self.config.to_record(),
            "models": [item.to_record() for item in self.models],
            "scores": [item.to_record() for item in self.scores],
        }


def _fit_target(
    dataset: RepairCostDataset,
    action: RecoverabilityAction,
    target: RepairCostTarget,
    config: RepairCostConfig,
) -> tuple[RepairCostModel | None, RepairCostScore]:
    def partition(name: str) -> tuple[RepairCostSample, ...]:
        return tuple(
            item
            for item in dataset.samples
            if dataset.split.partition_for(item.lineage_group_id) == name
            and item.action.key == action.key
            and item.observation.status is RepairCostObservationStatus.MEASURED
            and item.observation.value_for(target) is not None
        )

    train, validation, test = partition("train"), partition("validation"), partition("test")
    if len(train) < config.minimum_train_examples:
        return None, RepairCostScore(
            action,
            target,
            RepairCostObservationStatus.UNSUPPORTED,
            None,
            None,
            len(validation),
            len(test),
            "insufficient measured training evidence",
        )
    if not validation:
        return None, RepairCostScore(
            action,
            target,
            RepairCostObservationStatus.UNKNOWN,
            None,
            None,
            0,
            len(test),
            "validation evidence is required for calibrated intervals",
        )
    rows = tuple(item.features for item in train)
    targets = tuple(item.observation.value_for(target) for item in train)
    assert all(value is not None for value in targets)
    numeric_targets = tuple(value for value in targets if value is not None)
    coefficients, intercept, means, scales = _fit_linear(rows, numeric_targets, config.ridge)
    states = tuple(sorted({item.state_id for item in train}))
    bounds = tuple(
        (min(row[index] for row in rows), max(row[index] for row in rows))
        for index in range(len(dataset.feature_names))
    )

    def predict(item: RepairCostSample) -> float:
        return intercept + math.fsum(
            coefficient * value
            for coefficient, value in zip(
                coefficients,
                _normalize(item.features, means, scales),
                strict=True,
            )
        )

    validation_predictions = tuple(predict(item) for item in validation)
    validation_actual = tuple(
        value
        for item in validation
        for value in (item.observation.value_for(target),)
        if value is not None
    )
    radius = _quantile(
        tuple(
            abs(predicted - actual)
            for predicted, actual in zip(validation_predictions, validation_actual, strict=True)
        ),
        config.confidence,
    )
    identity = {
        "action": action.to_record(),
        "target": target.value,
        "dataset_id": dataset.dataset_id,
        "config": config.to_record(),
        "coefficients": list(coefficients),
        "intercept": intercept,
        "feature_means": list(means),
        "feature_scales": list(scales),
        "interval_radius": radius,
        "training_state_ids": list(states),
        "feature_bounds": [list(item) for item in bounds],
    }
    predictor_id = (
        "repair_cost_predictor_" + hashlib.sha256(_canonical(identity).encode()).hexdigest()
    )
    model = RepairCostModel(
        action,
        target,
        coefficients,
        intercept,
        means,
        scales,
        radius,
        states,
        bounds,
        dataset.dataset_id,
        predictor_id,
    )
    test_actual = tuple(
        value
        for item in test
        for value in (item.observation.value_for(target),)
        if value is not None
    )
    test_predictions = tuple(predict(item) for item in test)
    baseline = sum(numeric_targets) / len(numeric_targets)
    model_mae = (
        sum(
            abs(predicted - actual)
            for predicted, actual in zip(test_predictions, test_actual, strict=True)
        )
        / len(test_actual)
        if test_actual
        else None
    )
    baseline_mae = (
        sum(abs(baseline - actual) for actual in test_actual) / len(test_actual)
        if test_actual
        else None
    )
    outcome = (
        RepairCostObservationStatus.UNKNOWN
        if model_mae is None
        else RepairCostObservationStatus.MEASURED
        if model_mae < (baseline_mae or math.inf)
        else RepairCostObservationStatus.FAILED
    )
    reason = (
        "no measured held-out test evidence"
        if model_mae is None
        else None
        if outcome is RepairCostObservationStatus.MEASURED
        else "cost model did not beat the action mean baseline"
    )
    return model, RepairCostScore(
        action,
        target,
        outcome,
        model_mae,
        baseline_mae,
        len(validation),
        len(test),
        reason,
    )


def fit_repair_cost_predictors(
    dataset: RepairCostDataset,
    config: RepairCostConfig | None = None,
) -> RepairCostStudy:
    """Fit action/target cost models while retaining optional-energy gaps."""

    resolved = config or RepairCostConfig()
    actions = tuple(sorted({item.action for item in dataset.samples}, key=lambda item: item.key))
    models: list[RepairCostModel] = []
    scores: list[RepairCostScore] = []
    for action in actions:
        for target in resolved.targets:
            model, score = _fit_target(dataset, action, target, resolved)
            if model is not None:
                models.append(model)
            scores.append(score)
    return RepairCostStudy(dataset.dataset_id, resolved, tuple(models), tuple(scores))

"""Classical and bounded learned surgeon baselines."""

from __future__ import annotations

import base64
import io
import json
import math
import random
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from typing import Any, Final, cast

import numpy as np

from .matrix import SurgeonMatrix

MODEL_SCHEMA_VERSION: Final[int] = 1


class SurgeonModelError(ValueError):
    """Raised when a surgeon model cannot be trained or used safely."""


class SurgeonDependencyError(SurgeonModelError):
    """Raised when an optional model backend is not installed."""


class ModelTask(StrEnum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


def _measured_rows(
    matrix: SurgeonMatrix,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    x: list[tuple[float, ...]] = []
    y: list[float] = []
    weights: list[float] = []
    for row, target, mask, weight in zip(
        matrix.values,
        matrix.target_values,
        matrix.target_mask,
        matrix.sample_weights,
        strict=True,
    ):
        if mask:
            x.append(row)
            y.append(target)
            weights.append(weight)
    if not x:
        raise SurgeonModelError(
            f"{matrix.partition.value} split has no measured labels for {matrix.target_name!r}"
        )
    return tuple(x), tuple(y), tuple(weights)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


@dataclass(frozen=True, slots=True)
class LinearConfig:
    alpha: float = 1e-3
    l1_ratio: float = 0.0
    learning_rate: float = 0.03
    max_epochs: int = 2000
    tolerance: float = 1e-8
    seed: int = 0

    def __post_init__(self) -> None:
        numeric = (self.alpha, self.l1_ratio, self.learning_rate, self.tolerance)
        if any(not math.isfinite(value) for value in numeric):
            raise SurgeonModelError("linear hyperparameters must be finite")
        if self.alpha < 0 or not 0.0 <= self.l1_ratio <= 1.0:
            raise SurgeonModelError("linear alpha/l1_ratio are outside supported range")
        if self.learning_rate <= 0 or self.max_epochs <= 0 or self.tolerance < 0:
            raise SurgeonModelError("linear optimizer limits must be positive")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64:
            raise SurgeonModelError("linear seed must be unsigned 64-bit")

    def to_record(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "l1_ratio": self.l1_ratio,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
            "seed": self.seed,
        }


DEFAULT_LINEAR_CONFIG: Final[LinearConfig] = LinearConfig()


@dataclass(frozen=True, slots=True)
class LinearSurgeonModel:
    feature_names: tuple[str, ...]
    target_name: str
    coefficients: tuple[float, ...]
    intercept: float
    config: LinearConfig
    epochs: int
    validation_loss: float | None = None
    schema_version: int = MODEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_SCHEMA_VERSION:
            raise SurgeonModelError("unsupported linear model schema version")
        if len(self.feature_names) != len(self.coefficients):
            raise SurgeonModelError("linear coefficients must align with feature names")
        if not self.target_name or self.epochs <= 0:
            raise SurgeonModelError("linear model target and training epochs are required")
        if any(not math.isfinite(value) for value in (*self.coefficients, self.intercept)):
            raise SurgeonModelError("linear model parameters must be finite")

    def predict(self, rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
        for row in rows:
            if len(row) != len(self.coefficients):
                raise SurgeonModelError("linear inference feature width is incompatible")
        return tuple(self.intercept + _dot(self.coefficients, row) for row in rows)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "linear",
            "feature_names": list(self.feature_names),
            "target_name": self.target_name,
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "config": self.config.to_record(),
            "epochs": self.epochs,
            "validation_loss": self.validation_loss,
        }

    @classmethod
    def from_record(cls, value: dict[str, object]) -> LinearSurgeonModel:
        if value.get("kind") != "linear" or value.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise SurgeonModelError("persisted linear model kind/schema is incompatible")
        feature_names = value.get("feature_names")
        coefficients = value.get("coefficients")
        target_name = value.get("target_name")
        config_raw = value.get("config")
        epochs = value.get("epochs")
        intercept = value.get("intercept")
        validation_loss = value.get("validation_loss")
        if (
            not isinstance(feature_names, list)
            or not all(isinstance(item, str) for item in feature_names)
            or not isinstance(coefficients, list)
            or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in coefficients
            )
            or not isinstance(target_name, str)
            or not isinstance(config_raw, dict)
            or not isinstance(epochs, int)
            or isinstance(epochs, bool)
            or not isinstance(intercept, (int, float))
            or isinstance(intercept, bool)
        ):
            raise SurgeonModelError("persisted linear model fields are malformed")
        config = LinearConfig(
            alpha=float(config_raw["alpha"]),
            l1_ratio=float(config_raw["l1_ratio"]),
            learning_rate=float(config_raw["learning_rate"]),
            max_epochs=int(config_raw["max_epochs"]),
            tolerance=float(config_raw["tolerance"]),
            seed=int(config_raw["seed"]),
        )
        resolved_validation = (
            None if validation_loss is None else float(cast(int | float, validation_loss))
        )
        return cls(
            tuple(cast(list[str], feature_names)),
            target_name,
            tuple(float(item) for item in cast(list[int | float], coefficients)),
            float(intercept),
            config,
            epochs,
            resolved_validation,
        )


def _mse(
    predictions: Sequence[float],
    targets: Sequence[float],
    weights: Sequence[float],
) -> float:
    total_weight = math.fsum(weights)
    return (
        math.fsum(
            weight * (prediction - target) ** 2
            for prediction, target, weight in zip(predictions, targets, weights, strict=True)
        )
        / total_weight
    )


def train_linear(
    train: SurgeonMatrix,
    *,
    config: LinearConfig = DEFAULT_LINEAR_CONFIG,
    validation: SurgeonMatrix | None = None,
) -> LinearSurgeonModel:
    """Train ridge/elastic-net regression by deterministic proximal gradient descent."""

    x, y, weights = _measured_rows(train)
    width = len(train.feature_names)
    if width == 0:
        raise SurgeonModelError("linear regression requires at least one feature")
    if any(len(row) != width for row in x):
        raise SurgeonModelError("training matrix rows have inconsistent widths")
    total_weight = math.fsum(weights)
    mean_y = (
        math.fsum(target * weight for target, weight in zip(y, weights, strict=True)) / total_weight
    )
    coefficients = [0.0] * width
    intercept = mean_y
    previous = math.inf
    epochs = 0

    l2 = config.alpha * (1.0 - config.l1_ratio)
    l1 = config.alpha * config.l1_ratio
    for epoch in range(1, config.max_epochs + 1):
        predictions = [intercept + _dot(coefficients, row) for row in x]
        errors = [prediction - target for prediction, target in zip(predictions, y, strict=True)]
        grad_intercept = (
            2.0
            * math.fsum(weight * error for weight, error in zip(weights, errors, strict=True))
            / total_weight
        )
        gradients = [
            (
                2.0
                * math.fsum(
                    weight * error * row[index]
                    for row, error, weight in zip(x, errors, weights, strict=True)
                )
                / total_weight
                + 2.0 * l2 * coefficients[index]
            )
            for index in range(width)
        ]
        intercept -= config.learning_rate * grad_intercept
        threshold = config.learning_rate * l1
        for index, gradient in enumerate(gradients):
            updated = coefficients[index] - config.learning_rate * gradient
            if updated > threshold:
                updated -= threshold
            elif updated < -threshold:
                updated += threshold
            else:
                updated = 0.0
            coefficients[index] = updated

        predictions = [intercept + _dot(coefficients, row) for row in x]
        loss = (
            _mse(predictions, y, weights)
            + l2 * math.fsum(value * value for value in coefficients)
            + l1 * math.fsum(abs(value) for value in coefficients)
        )
        epochs = epoch
        if abs(previous - loss) <= config.tolerance:
            break
        previous = loss

    validation_loss: float | None = None
    if validation is not None:
        val_x, val_y, val_weights = _measured_rows(validation)
        validation_loss = _mse(
            tuple(intercept + _dot(coefficients, row) for row in val_x),
            val_y,
            val_weights,
        )
    return LinearSurgeonModel(
        train.feature_names,
        train.target_name,
        tuple(coefficients),
        intercept,
        config,
        epochs,
        validation_loss,
    )


def select_linear_model(
    train: SurgeonMatrix,
    validation: SurgeonMatrix,
    configs: Iterable[LinearConfig],
) -> LinearSurgeonModel:
    """Select hyperparameters using train+validation only; test labels are not accepted."""

    candidates = tuple(configs)
    if not candidates:
        raise SurgeonModelError("linear model selection requires at least one configuration")
    models = tuple(
        train_linear(train, config=config, validation=validation) for config in candidates
    )
    return min(
        models,
        key=lambda model: (
            math.inf if model.validation_loss is None else model.validation_loss,
            model.config.alpha,
            model.config.l1_ratio,
        ),
    )


@dataclass(frozen=True, slots=True)
class LogisticConfig:
    alpha: float = 1e-3
    learning_rate: float = 0.03
    max_epochs: int = 2000
    tolerance: float = 1e-8
    class_weight: str = "balanced"
    threshold: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        if self.alpha < 0 or not math.isfinite(self.alpha):
            raise SurgeonModelError("logistic alpha must be finite and non-negative")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise SurgeonModelError("logistic learning rate must be finite and positive")
        if self.max_epochs <= 0 or self.tolerance < 0 or not math.isfinite(self.tolerance):
            raise SurgeonModelError("logistic optimizer limits are invalid")
        if self.class_weight not in {"balanced", "none"}:
            raise SurgeonModelError("class_weight must be 'balanced' or 'none'")
        if not 0.0 < self.threshold < 1.0:
            raise SurgeonModelError("classification threshold must be within (0, 1)")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64:
            raise SurgeonModelError("logistic seed must be unsigned 64-bit")

    def to_record(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tolerance": self.tolerance,
            "class_weight": self.class_weight,
            "threshold": self.threshold,
            "seed": self.seed,
        }


DEFAULT_LOGISTIC_CONFIG: Final[LogisticConfig] = LogisticConfig()


@dataclass(frozen=True, slots=True)
class LogisticSurgeonModel:
    feature_names: tuple[str, ...]
    target_name: str
    coefficients: tuple[float, ...]
    intercept: float
    config: LogisticConfig
    epochs: int
    positive_weight: float
    validation_log_loss: float | None = None
    schema_version: int = MODEL_SCHEMA_VERSION

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
        for row in rows:
            if len(row) != len(self.coefficients):
                raise SurgeonModelError("logistic inference feature width is incompatible")
        return tuple(_sigmoid(self.intercept + _dot(self.coefficients, row)) for row in rows)

    def predict(self, rows: Sequence[Sequence[float]]) -> tuple[bool, ...]:
        return tuple(value >= self.config.threshold for value in self.predict_proba(rows))

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "logistic",
            "feature_names": list(self.feature_names),
            "target_name": self.target_name,
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "config": self.config.to_record(),
            "epochs": self.epochs,
            "positive_weight": self.positive_weight,
            "validation_log_loss": self.validation_log_loss,
        }

    @classmethod
    def from_record(cls, value: dict[str, object]) -> LogisticSurgeonModel:
        if value.get("kind") != "logistic" or value.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise SurgeonModelError("persisted logistic model kind/schema is incompatible")
        try:
            feature_names = tuple(cast(list[str], value["feature_names"]))
            coefficients = tuple(float(item) for item in cast(list[float], value["coefficients"]))
            config_raw = cast(dict[str, object], value["config"])
            config = LogisticConfig(
                alpha=float(cast(float, config_raw["alpha"])),
                learning_rate=float(cast(float, config_raw["learning_rate"])),
                max_epochs=int(cast(int, config_raw["max_epochs"])),
                tolerance=float(cast(float, config_raw["tolerance"])),
                class_weight=str(config_raw["class_weight"]),
                threshold=float(cast(float, config_raw["threshold"])),
                seed=int(cast(int, config_raw["seed"])),
            )
            validation_raw = value.get("validation_log_loss")
            validation = None if validation_raw is None else float(cast(float, validation_raw))
            return cls(
                feature_names,
                str(value["target_name"]),
                coefficients,
                float(cast(float, value["intercept"])),
                config,
                int(cast(int, value["epochs"])),
                float(cast(float, value["positive_weight"])),
                validation,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SurgeonModelError("persisted logistic model fields are malformed") from error


def _binary_labels(values: Sequence[float]) -> tuple[int, ...]:
    labels: list[int] = []
    for value in values:
        if value not in (0.0, 1.0):
            raise SurgeonModelError("classification targets must be encoded as 0/1")
        labels.append(int(value))
    if len(set(labels)) < 2:
        raise SurgeonModelError(
            "training split contains one safe-mutation class; increase data or use a split "
            "that contains both safe and unsafe mutations"
        )
    return tuple(labels)


def _log_loss(
    probabilities: Sequence[float],
    labels: Sequence[int],
    weights: Sequence[float],
) -> float:
    epsilon = 1e-15
    total_weight = math.fsum(weights)
    return (
        -math.fsum(
            weight
            * (
                label * math.log(min(1.0 - epsilon, max(epsilon, probability)))
                + (1 - label) * math.log(min(1.0 - epsilon, max(epsilon, 1.0 - probability)))
            )
            for probability, label, weight in zip(probabilities, labels, weights, strict=True)
        )
        / total_weight
    )


def train_logistic(
    train: SurgeonMatrix,
    *,
    config: LogisticConfig = DEFAULT_LOGISTIC_CONFIG,
    validation: SurgeonMatrix | None = None,
) -> LogisticSurgeonModel:
    """Train a class-weighted deterministic L2 logistic baseline."""

    x, raw_y, sample_weights = _measured_rows(train)
    labels = _binary_labels(raw_y)
    positives = sum(labels)
    negatives = len(labels) - positives
    positive_weight = negatives / positives if config.class_weight == "balanced" else 1.0
    weights = tuple(
        sample_weight * (positive_weight if label else 1.0)
        for sample_weight, label in zip(sample_weights, labels, strict=True)
    )
    total_weight = math.fsum(weights)
    width = len(train.feature_names)
    coefficients = [0.0] * width
    prior = min(1.0 - 1e-6, max(1e-6, positives / len(labels)))
    intercept = math.log(prior / (1.0 - prior))
    previous = math.inf
    epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        probabilities = [_sigmoid(intercept + _dot(coefficients, row)) for row in x]
        residuals = [
            probability - label for probability, label in zip(probabilities, labels, strict=True)
        ]
        grad_intercept = (
            math.fsum(
                weight * residual for weight, residual in zip(weights, residuals, strict=True)
            )
            / total_weight
        )
        gradients = [
            (
                math.fsum(
                    weight * residual * row[index]
                    for row, residual, weight in zip(x, residuals, weights, strict=True)
                )
                / total_weight
                + 2.0 * config.alpha * coefficients[index]
            )
            for index in range(width)
        ]
        intercept -= config.learning_rate * grad_intercept
        coefficients = [
            value - config.learning_rate * gradient
            for value, gradient in zip(coefficients, gradients, strict=True)
        ]
        probabilities = [_sigmoid(intercept + _dot(coefficients, row)) for row in x]
        loss = _log_loss(probabilities, labels, weights) + config.alpha * math.fsum(
            value * value for value in coefficients
        )
        epochs = epoch
        if abs(previous - loss) <= config.tolerance:
            break
        previous = loss

    validation_loss: float | None = None
    if validation is not None:
        val_x, val_y, val_weights = _measured_rows(validation)
        if any(value not in (0.0, 1.0) for value in val_y):
            raise SurgeonModelError("validation classification targets must be 0/1")
        val_labels = tuple(int(value) for value in val_y)
        validation_loss = _log_loss(
            tuple(_sigmoid(intercept + _dot(coefficients, row)) for row in val_x),
            val_labels,
            val_weights,
        )
    return LogisticSurgeonModel(
        train.feature_names,
        train.target_name,
        tuple(coefficients),
        intercept,
        config,
        epochs,
        positive_weight,
        validation_loss,
    )


def select_logistic_model(
    train: SurgeonMatrix,
    validation: SurgeonMatrix,
    configs: Iterable[LogisticConfig],
) -> LogisticSurgeonModel:
    """Select logistic hyperparameters without accepting or reading test labels."""

    candidates = tuple(configs)
    if not candidates:
        raise SurgeonModelError("logistic model selection requires configurations")
    models = tuple(
        train_logistic(train, config=config, validation=validation) for config in candidates
    )
    return min(
        models,
        key=lambda model: (
            math.inf if model.validation_log_loss is None else model.validation_log_loss,
            model.config.alpha,
        ),
    )


@dataclass(frozen=True, slots=True)
class LightGBMConfig:
    task: ModelTask
    num_leaves: int = 31
    learning_rate: float = 0.05
    max_rounds: int = 1000
    early_stopping_rounds: int = 50
    num_threads: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_leaves < 2 or self.max_rounds <= 0 or self.early_stopping_rounds <= 0:
            raise SurgeonModelError("LightGBM tree/round limits are invalid")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise SurgeonModelError("LightGBM learning rate must be finite and positive")
        if self.num_threads <= 0 or self.num_threads > 32:
            raise SurgeonModelError("LightGBM num_threads must be within 1..32")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 31:
            raise SurgeonModelError("LightGBM seed must fit a non-negative 31-bit integer")


def _lightgbm_rows(
    rows: Sequence[Sequence[float]],
    *,
    width: int,
) -> Any:
    if width <= 0:
        raise SurgeonModelError("LightGBM requires at least one input feature")
    if not rows:
        return np.empty((0, width), dtype=np.float64)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or int(matrix.shape[1]) != width:
        raise SurgeonModelError("LightGBM input rows have incompatible feature width")
    if not bool(np.isfinite(matrix).all()):
        raise SurgeonModelError("LightGBM input rows must be finite")
    return matrix


def _lightgbm_feature_names(width: int) -> list[str]:
    # The public preprocessing schema deliberately uses readable names such as
    # `num:weight_l1_norm` and `cat:model_family=llama`. LightGBM's model-string
    # format rejects several punctuation characters, so keep stable backend-only
    # names while preserving the semantic schema on LightGBMSurgeonModel.
    return [f"feature_{index}" for index in range(width)]


@dataclass(frozen=True, slots=True)
class LightGBMSurgeonModel:
    task: ModelTask
    feature_names: tuple[str, ...]
    target_name: str
    model_string: str
    config: LightGBMConfig
    best_iteration: int
    positive_weight: float
    schema_version: int = MODEL_SCHEMA_VERSION

    def predict(self, rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
        if not rows:
            return ()
        lightgbm = _lightgbm()
        booster = lightgbm.Booster(model_str=self.model_string)
        matrix = _lightgbm_rows(rows, width=len(self.feature_names))
        raw = booster.predict(matrix, num_iteration=self.best_iteration)
        return tuple(float(value) for value in raw)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "lightgbm",
            "task": self.task.value,
            "feature_names": list(self.feature_names),
            "target_name": self.target_name,
            "model_string": self.model_string,
            "best_iteration": self.best_iteration,
            "positive_weight": self.positive_weight,
            "config": {
                "num_leaves": self.config.num_leaves,
                "learning_rate": self.config.learning_rate,
                "max_rounds": self.config.max_rounds,
                "early_stopping_rounds": self.config.early_stopping_rounds,
                "num_threads": self.config.num_threads,
                "seed": self.config.seed,
            },
        }

    @classmethod
    def from_record(cls, value: dict[str, object]) -> LightGBMSurgeonModel:
        try:
            if (
                value.get("kind") != "lightgbm"
                or value.get("schema_version") != MODEL_SCHEMA_VERSION
            ):
                raise SurgeonModelError("persisted LightGBM kind/schema is incompatible")
            task = ModelTask(str(value["task"]))
            config_raw = cast(dict[str, object], value["config"])
            config = LightGBMConfig(
                task,
                int(cast(int, config_raw["num_leaves"])),
                float(cast(float, config_raw["learning_rate"])),
                int(cast(int, config_raw["max_rounds"])),
                int(cast(int, config_raw["early_stopping_rounds"])),
                int(cast(int, config_raw["num_threads"])),
                int(cast(int, config_raw["seed"])),
            )
            return cls(
                task,
                tuple(cast(list[str], value["feature_names"])),
                str(value["target_name"]),
                str(value["model_string"]),
                config,
                int(cast(int, value["best_iteration"])),
                float(cast(float, value["positive_weight"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SurgeonModelError("persisted LightGBM model fields are malformed") from error


def _lightgbm() -> Any:
    try:
        return import_module("lightgbm")
    except ImportError as error:
        raise SurgeonDependencyError(
            "LightGBM surgeon baselines require the optional 'lightgbm' package"
        ) from error


def train_lightgbm(
    train: SurgeonMatrix,
    validation: SurgeonMatrix,
    *,
    config: LightGBMConfig,
) -> LightGBMSurgeonModel:
    """Train bounded deterministic LightGBM with validation-only early stopping."""

    lightgbm = _lightgbm()
    x, y, weights = _measured_rows(train)
    val_x, val_y, val_weights = _measured_rows(validation)
    width = len(train.feature_names)
    if validation.feature_names != train.feature_names:
        raise SurgeonModelError("LightGBM train/validation feature schemas differ")
    train_matrix = _lightgbm_rows(x, width=width)
    validation_matrix = _lightgbm_rows(val_x, width=width)
    backend_feature_names = _lightgbm_feature_names(width)

    positive_weight = 1.0
    objective = "regression"
    metric = "l2"
    if config.task is ModelTask.CLASSIFICATION:
        labels = _binary_labels(y)
        positives = sum(labels)
        negatives = len(labels) - positives
        positive_weight = negatives / positives
        objective = "binary"
        metric = "auc"

    params: dict[str, object] = {
        "objective": objective,
        "metric": metric,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "num_threads": config.num_threads,
        "seed": config.seed,
        "feature_fraction_seed": config.seed,
        "bagging_seed": config.seed,
        "data_random_seed": config.seed,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if config.task is ModelTask.CLASSIFICATION:
        params["scale_pos_weight"] = positive_weight

    training = lightgbm.Dataset(
        train_matrix,
        label=list(y),
        weight=list(weights),
        feature_name=backend_feature_names,
        free_raw_data=False,
    )
    validating = lightgbm.Dataset(
        validation_matrix,
        label=list(val_y),
        weight=list(val_weights),
        reference=training,
        feature_name=backend_feature_names,
        free_raw_data=False,
    )
    booster = lightgbm.train(
        params,
        training,
        num_boost_round=config.max_rounds,
        valid_sets=[validating],
        valid_names=["validation"],
        callbacks=[lightgbm.early_stopping(config.early_stopping_rounds, verbose=False)],
    )
    model_string = str(booster.model_to_string(num_iteration=booster.best_iteration))
    reloaded = lightgbm.Booster(model_str=model_string)
    original = tuple(
        float(value)
        for value in booster.predict(
            validation_matrix,
            num_iteration=booster.best_iteration,
        )
    )
    restored = tuple(
        float(value)
        for value in reloaded.predict(
            validation_matrix,
            num_iteration=booster.best_iteration,
        )
    )
    if len(original) != len(restored) or any(
        not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
        for left, right in zip(original, restored, strict=True)
    ):
        raise SurgeonModelError("reloaded LightGBM model does not reproduce validation predictions")
    return LightGBMSurgeonModel(
        config.task,
        train.feature_names,
        train.target_name,
        model_string,
        config,
        int(booster.best_iteration),
        positive_weight,
    )


@dataclass(frozen=True, slots=True)
class MLPConfig:
    task: ModelTask
    hidden_sizes: tuple[int, ...] = (64, 32)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 20
    dropout: float = 0.0
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.hidden_sizes or any(size <= 0 or size > 512 for size in self.hidden_sizes):
            raise SurgeonModelError("MLP hidden sizes must be within 1..512")
        if sum(self.hidden_sizes) > 1024:
            raise SurgeonModelError("MLP hidden width budget exceeds 1024 units")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise SurgeonModelError("MLP optimizer settings are invalid")
        if self.batch_size <= 0 or self.max_epochs <= 0 or self.patience <= 0:
            raise SurgeonModelError("MLP training limits must be positive")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise SurgeonModelError("MLP dropout must be finite and within [0, 1)")
        if self.device not in {"cpu", "cuda"}:
            raise SurgeonModelError("MLP device must be 'cpu' or 'cuda'")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 31:
            raise SurgeonModelError("MLP seed must fit a non-negative 31-bit integer")


@dataclass(frozen=True, slots=True)
class MLPSurgeonModel:
    task: ModelTask
    feature_names: tuple[str, ...]
    target_name: str
    hidden_sizes: tuple[int, ...]
    state_base64: str
    config: MLPConfig
    epochs: int
    parameter_count: int
    peak_vram_bytes: int
    schema_version: int = MODEL_SCHEMA_VERSION

    def predict(self, rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
        torch, nn = _torch_modules()
        model = _build_torch_mlp(
            nn, len(self.feature_names), self.hidden_sizes, dropout=self.config.dropout
        )
        state_bytes = base64.b64decode(self.state_base64.encode("ascii"))
        state = torch.load(io.BytesIO(state_bytes), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor([list(row) for row in rows], dtype=torch.float32)
            output = model(tensor).squeeze(-1)
            if self.task is ModelTask.CLASSIFICATION:
                output = torch.sigmoid(output)
            return tuple(float(value) for value in output.tolist())

    def predict_stochastic(
        self,
        rows: Sequence[Sequence[float]],
        *,
        passes: int,
        seed: int,
    ) -> tuple[tuple[float, ...], ...]:
        """Run bounded deterministic Monte Carlo dropout inference on CPU."""

        if self.config.dropout <= 0.0:
            raise SurgeonModelError("stochastic MLP inference requires configured dropout")
        if passes < 2 or passes > 10_000:
            raise SurgeonModelError("stochastic MLP passes must be within 2..10000")
        if isinstance(seed, bool) or not 0 <= seed < 1 << 31:
            raise SurgeonModelError("stochastic MLP seed must be a non-negative 31-bit integer")
        torch, nn = _torch_modules()
        model = _build_torch_mlp(
            nn, len(self.feature_names), self.hidden_sizes, dropout=self.config.dropout
        )
        state_bytes = base64.b64decode(self.state_base64.encode("ascii"))
        state = torch.load(io.BytesIO(state_bytes), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.train()
        tensor = torch.tensor([list(row) for row in rows], dtype=torch.float32)
        torch.manual_seed(seed)
        outputs: list[tuple[float, ...]] = []
        with torch.no_grad():
            for _ in range(passes):
                output = model(tensor).squeeze(-1)
                if self.task is ModelTask.CLASSIFICATION:
                    output = torch.sigmoid(output)
                outputs.append(tuple(float(value) for value in output.tolist()))
        return tuple(outputs)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": "mlp",
            "task": self.task.value,
            "feature_names": list(self.feature_names),
            "target_name": self.target_name,
            "hidden_sizes": list(self.hidden_sizes),
            "state_base64": self.state_base64,
            "epochs": self.epochs,
            "parameter_count": self.parameter_count,
            "peak_vram_bytes": self.peak_vram_bytes,
            "config": {
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "batch_size": self.config.batch_size,
                "max_epochs": self.config.max_epochs,
                "patience": self.config.patience,
                "dropout": self.config.dropout,
                "device": self.config.device,
                "seed": self.config.seed,
            },
        }

    @classmethod
    def from_record(cls, value: dict[str, object]) -> MLPSurgeonModel:
        try:
            if value.get("kind") != "mlp" or value.get("schema_version") != MODEL_SCHEMA_VERSION:
                raise SurgeonModelError("persisted MLP kind/schema is incompatible")
            task = ModelTask(str(value["task"]))
            hidden = tuple(int(item) for item in cast(list[int], value["hidden_sizes"]))
            config_raw = cast(dict[str, object], value["config"])
            config = MLPConfig(
                task=task,
                hidden_sizes=hidden,
                learning_rate=float(cast(float, config_raw["learning_rate"])),
                weight_decay=float(cast(float, config_raw["weight_decay"])),
                batch_size=int(cast(int, config_raw["batch_size"])),
                max_epochs=int(cast(int, config_raw["max_epochs"])),
                patience=int(cast(int, config_raw["patience"])),
                dropout=float(cast(float, config_raw.get("dropout", 0.0))),
                device=str(config_raw["device"]),
                seed=int(cast(int, config_raw["seed"])),
            )
            return cls(
                task,
                tuple(cast(list[str], value["feature_names"])),
                str(value["target_name"]),
                hidden,
                str(value["state_base64"]),
                config,
                int(cast(int, value["epochs"])),
                int(cast(int, value["parameter_count"])),
                int(cast(int, value["peak_vram_bytes"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SurgeonModelError("persisted MLP model fields are malformed") from error


def _torch_modules() -> tuple[Any, Any]:
    try:
        torch = import_module("torch")
        nn = import_module("torch.nn")
    except ImportError as error:
        raise SurgeonDependencyError(
            "MLP surgeon baseline requires PyTorch (install the existing 'hf' extra)"
        ) from error
    return torch, nn


def _build_torch_mlp(
    nn: Any, width: int, hidden_sizes: Sequence[int], *, dropout: float = 0.0
) -> Any:
    layers: list[Any] = []
    previous = width
    for hidden in hidden_sizes:
        layers.extend((nn.Linear(previous, hidden), nn.ReLU()))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        previous = hidden
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def train_mlp(
    train: SurgeonMatrix,
    validation: SurgeonMatrix,
    *,
    config: MLPConfig,
) -> MLPSurgeonModel:
    """Train a bounded deterministic tabular MLP with validation early stopping."""

    torch, nn = _torch_modules()
    if config.device == "cuda" and not bool(torch.cuda.is_available()):
        raise SurgeonModelError("MLP CUDA device requested but CUDA is unavailable")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if bool(torch.cuda.is_available()):
        torch.cuda.manual_seed_all(config.seed)
    with suppress(AttributeError):
        torch.use_deterministic_algorithms(True, warn_only=True)

    train_x, train_y, train_weights = _measured_rows(train)
    val_x, val_y, val_weights = _measured_rows(validation)
    device = torch.device(config.device)
    model = _build_torch_mlp(
        nn,
        len(train.feature_names),
        config.hidden_sizes,
        dropout=config.dropout,
    ).to(device)
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    if config.task is ModelTask.CLASSIFICATION:
        _binary_labels(train_y)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    else:
        loss_fn = nn.MSELoss(reduction="none")

    x_tensor = torch.tensor([list(row) for row in train_x], dtype=torch.float32)
    y_tensor = torch.tensor(list(train_y), dtype=torch.float32)
    w_tensor = torch.tensor(list(train_weights), dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor, w_tensor)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_tensor = torch.tensor([list(row) for row in val_x], dtype=torch.float32, device=device)
    val_target = torch.tensor(list(val_y), dtype=torch.float32, device=device)
    val_weight = torch.tensor(list(val_weights), dtype=torch.float32, device=device)

    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_loss = math.inf
    best_state: bytes | None = None
    stale = 0
    epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch_x, batch_y, batch_weight in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_weight = batch_weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_x).squeeze(-1)
            losses = loss_fn(predictions, batch_y)
            loss = (losses * batch_weight).sum() / batch_weight.sum()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_tensor).squeeze(-1)
            val_losses = loss_fn(val_predictions, val_target)
            val_loss = float((val_losses * val_weight).sum().item() / val_weight.sum().item())
        epochs = epoch
        if val_loss < best_loss - 1e-10:
            best_loss = val_loss
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            best_state = buffer.getvalue()
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is None:
        raise SurgeonModelError("MLP training did not produce a finite validation checkpoint")
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if config.device == "cuda" else 0
    return MLPSurgeonModel(
        config.task,
        train.feature_names,
        train.target_name,
        config.hidden_sizes,
        base64.b64encode(best_state).decode("ascii"),
        config,
        epochs,
        parameter_count,
        peak_vram,
    )


type SerializableModel = (
    LinearSurgeonModel | LogisticSurgeonModel | LightGBMSurgeonModel | MLPSurgeonModel
)


def model_to_json(model: SerializableModel) -> str:
    """Serialize classical/tree models canonically for immutable registry storage."""

    return json.dumps(model.to_record(), sort_keys=True, separators=(",", ":"))


def model_from_json(payload: str) -> SerializableModel:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise SurgeonModelError("surgeon model payload is invalid JSON") from error
    if not isinstance(value, dict):
        raise SurgeonModelError("surgeon model payload must be an object")
    record = cast(dict[str, object], value)
    kind = record.get("kind")
    if kind == "linear":
        return LinearSurgeonModel.from_record(record)
    if kind == "logistic":
        return LogisticSurgeonModel.from_record(record)
    if kind == "lightgbm":
        return LightGBMSurgeonModel.from_record(record)
    if kind == "mlp":
        return MLPSurgeonModel.from_record(record)
    raise SurgeonModelError(f"unknown persisted surgeon model kind {kind!r}")

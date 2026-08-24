"""Optional fixed-budget MLP uncertainty comparison."""

from __future__ import annotations

import base64
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.surgeon.matrix import SurgeonMatrix
from modelsurgeon.surgeon.models import MLPConfig, ModelTask, train_mlp

MLP_UNCERTAINTY_SCHEMA_VERSION: Final[int] = 1


class MLPUncertaintyError(ValueError):
    """Raised when MLP uncertainty evidence violates its bounded contract."""


class MLPUncertaintyMethod(StrEnum):
    DEEP_ENSEMBLE = "deep_ensemble"
    MC_DROPOUT = "mc_dropout"


@dataclass(frozen=True, slots=True)
class MLPUncertaintyBudget:
    enabled: bool = True
    max_trainings_per_method: int = 5
    max_stochastic_passes: int = 20
    max_cpu_seconds_per_method: float = 3600.0
    max_prediction_values: int = 1_000_000

    def __post_init__(self) -> None:
        if self.max_trainings_per_method <= 0 or self.max_stochastic_passes < 2:
            raise MLPUncertaintyError("MLP uncertainty training/pass budgets are invalid")
        if (
            not math.isfinite(self.max_cpu_seconds_per_method)
            or self.max_cpu_seconds_per_method <= 0.0
        ):
            raise MLPUncertaintyError("MLP uncertainty CPU budget must be finite and positive")
        if self.max_prediction_values <= 0:
            raise MLPUncertaintyError("MLP uncertainty prediction budget must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "max_trainings_per_method": self.max_trainings_per_method,
            "max_stochastic_passes": self.max_stochastic_passes,
            "max_cpu_seconds_per_method": self.max_cpu_seconds_per_method,
            "max_prediction_values": self.max_prediction_values,
        }


DEFAULT_MLP_UNCERTAINTY_BUDGET: Final[MLPUncertaintyBudget] = MLPUncertaintyBudget()


@dataclass(frozen=True, slots=True)
class MLPUncertaintyValue:
    point: float
    lower: float
    upper: float
    uncertainty: float
    schema_version: int = MLP_UNCERTAINTY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MLP_UNCERTAINTY_SCHEMA_VERSION:
            raise MLPUncertaintyError("unsupported MLP uncertainty value schema")
        if any(
            not math.isfinite(value)
            for value in (self.point, self.lower, self.upper, self.uncertainty)
        ):
            raise MLPUncertaintyError("MLP uncertainty values must be finite")
        if not self.lower <= self.point <= self.upper or self.uncertainty < 0.0:
            raise MLPUncertaintyError("MLP uncertainty interval is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class MLPMethodEvidence:
    method: MLPUncertaintyMethod
    predictions: tuple[MLPUncertaintyValue, ...]
    training_count: int
    stochastic_passes: int
    cpu_seconds: float
    model_bytes: int
    technique_version: str


@dataclass(frozen=True, slots=True)
class MLPUncertaintyScore:
    method: MLPUncertaintyMethod
    coverage: float
    target_coverage: float
    calibration_error: float
    active_selection_lift: float | None
    selected_count: int
    mean_interval_width: float
    evidence: MLPMethodEvidence

    def to_record(self) -> dict[str, object]:
        return {
            "method": self.method.value,
            "coverage": self.coverage,
            "target_coverage": self.target_coverage,
            "calibration_error": self.calibration_error,
            "active_selection_lift": self.active_selection_lift,
            "selected_count": self.selected_count,
            "mean_interval_width": self.mean_interval_width,
            "cost": {
                "training_count": self.evidence.training_count,
                "stochastic_passes": self.evidence.stochastic_passes,
                "cpu_seconds": self.evidence.cpu_seconds,
                "model_bytes": self.evidence.model_bytes,
                "prediction_value_count": len(self.evidence.predictions),
            },
            "technique_version": self.evidence.technique_version,
            "predictions": [item.to_record() for item in self.evidence.predictions],
        }


@dataclass(frozen=True, slots=True)
class MLPUncertaintyStudy:
    selected_method: MLPUncertaintyMethod
    candidates: tuple[MLPUncertaintyScore, ...]
    confidence: float
    top_fraction: float
    budget: MLPUncertaintyBudget
    schema_version: int = MLP_UNCERTAINTY_SCHEMA_VERSION

    @property
    def selected(self) -> MLPUncertaintyScore:
        return next(item for item in self.candidates if item.method is self.selected_method)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selected_method": self.selected_method.value,
            "selection_rule": "calibration_then_active_lift_then_cpu_bytes_method",
            "confidence": self.confidence,
            "top_fraction": self.top_fraction,
            "budget": self.budget.to_record(),
            "candidates": [item.to_record() for item in self.candidates],
        }


def estimate_mlp_uncertainty(
    passes: Sequence[Sequence[float]],
    *,
    confidence: float,
    max_prediction_values: int,
) -> tuple[MLPUncertaintyValue, ...]:
    values = tuple(tuple(row) for row in passes)
    if len(values) < 2 or not values[0]:
        raise MLPUncertaintyError("MLP uncertainty requires at least two non-empty passes")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise MLPUncertaintyError("MLP uncertainty pass arrays must align")
    if len(values) * width > max_prediction_values:
        raise MLPUncertaintyError("MLP uncertainty exceeds the prediction memory budget")
    if any(not math.isfinite(value) for row in values for value in row):
        raise MLPUncertaintyError("MLP uncertainty predictions must be finite")
    alpha = (1.0 - confidence) / 2.0
    output: list[MLPUncertaintyValue] = []
    for index in range(width):
        column = tuple(row[index] for row in values)
        point = math.fsum(column) / len(column)
        variance = math.fsum((value - point) ** 2 for value in column) / (len(column) - 1)
        output.append(
            MLPUncertaintyValue(
                point,
                min(point, _percentile(column, alpha)),
                max(point, _percentile(column, 1.0 - alpha)),
                math.sqrt(variance),
            )
        )
    return tuple(output)


def compare_mlp_uncertainty(
    validation_targets: Sequence[float],
    evidence: Sequence[MLPMethodEvidence],
    *,
    confidence: float = 0.9,
    top_fraction: float = 0.1,
    budget: MLPUncertaintyBudget = DEFAULT_MLP_UNCERTAINTY_BUDGET,
) -> MLPUncertaintyStudy | None:
    """Report calibration and active-selection lift, or return None when disabled."""

    if not budget.enabled:
        return None
    if not 0.0 < confidence < 1.0 or not 0.0 < top_fraction <= 1.0:
        raise MLPUncertaintyError("MLP uncertainty confidence/fraction bounds are invalid")
    if not validation_targets or any(not math.isfinite(value) for value in validation_targets):
        raise MLPUncertaintyError("MLP uncertainty validation targets must be finite")
    methods = tuple(item.method for item in evidence)
    if len(evidence) != 2 or set(methods) != set(MLPUncertaintyMethod):
        raise MLPUncertaintyError("MLP comparison requires deep ensemble and MC dropout")
    scores: list[MLPUncertaintyScore] = []
    selected_count = max(1, math.ceil(len(validation_targets) * top_fraction))
    for item in evidence:
        if len(item.predictions) != len(validation_targets):
            raise MLPUncertaintyError("MLP uncertainty predictions must align with targets")
        if item.training_count > budget.max_trainings_per_method:
            raise MLPUncertaintyError(f"{item.method.value} exceeds the training budget")
        if item.stochastic_passes > budget.max_stochastic_passes:
            raise MLPUncertaintyError(f"{item.method.value} exceeds the pass budget")
        if item.cpu_seconds > budget.max_cpu_seconds_per_method:
            raise MLPUncertaintyError(f"{item.method.value} exceeds the CPU budget")
        errors = tuple(
            abs(target - prediction.point)
            for target, prediction in zip(validation_targets, item.predictions, strict=True)
        )
        overall_error = math.fsum(errors) / len(errors)
        ranked = sorted(
            range(len(errors)), key=lambda index: (-item.predictions[index].uncertainty, index)
        )[:selected_count]
        selected_error = math.fsum(errors[index] for index in ranked) / selected_count
        coverage = sum(
            prediction.lower <= target <= prediction.upper
            for target, prediction in zip(validation_targets, item.predictions, strict=True)
        ) / len(validation_targets)
        scores.append(
            MLPUncertaintyScore(
                item.method,
                coverage,
                confidence,
                abs(coverage - confidence),
                None if overall_error == 0.0 else selected_error / overall_error,
                selected_count,
                math.fsum(value.upper - value.lower for value in item.predictions)
                / len(item.predictions),
                item,
            )
        )
    canonical = tuple(sorted(scores, key=lambda item: item.method.value))
    selected = min(
        canonical,
        key=lambda item: (
            item.calibration_error,
            math.inf if item.active_selection_lift is None else -item.active_selection_lift,
            item.evidence.cpu_seconds,
            item.evidence.model_bytes,
            item.method.value,
        ),
    )
    return MLPUncertaintyStudy(selected.method, canonical, confidence, top_fraction, budget)


@dataclass(frozen=True, slots=True)
class MLPUncertaintyRunConfig:
    budget: MLPUncertaintyBudget = DEFAULT_MLP_UNCERTAINTY_BUDGET
    confidence: float = 0.9
    top_fraction: float = 0.1
    ensemble_members: int = 5
    dropout_passes: int = 20
    dropout: float = 0.2
    hidden_sizes: tuple[int, ...] = (64, 32)
    max_epochs: int = 200
    patience: int = 20
    batch_size: int = 128
    device: str = "cpu"
    seed: int = 0


DEFAULT_MLP_UNCERTAINTY_RUN_CONFIG: Final[MLPUncertaintyRunConfig] = MLPUncertaintyRunConfig()


def run_mlp_uncertainty_study(
    train: SurgeonMatrix,
    validation: SurgeonMatrix,
    *,
    config: MLPUncertaintyRunConfig = DEFAULT_MLP_UNCERTAINTY_RUN_CONFIG,
) -> MLPUncertaintyStudy | None:
    """Train deep-ensemble and MC-dropout evidence when explicitly enabled."""

    if not config.budget.enabled:
        return None
    if (
        config.ensemble_members < 2
        or config.ensemble_members > config.budget.max_trainings_per_method
    ):
        raise MLPUncertaintyError("deep ensemble members exceed the training budget")
    if not 2 <= config.dropout_passes <= config.budget.max_stochastic_passes:
        raise MLPUncertaintyError("MC dropout passes exceed the stochastic-pass budget")
    ensemble_start = time.process_time()
    ensemble_models = tuple(
        train_mlp(
            train,
            validation,
            config=_model_config(config, seed=config.seed + index, dropout=0.0),
        )
        for index in range(config.ensemble_members)
    )
    ensemble_passes = tuple(model.predict(validation.values) for model in ensemble_models)
    ensemble_cpu = time.process_time() - ensemble_start
    dropout_start = time.process_time()
    dropout_model = train_mlp(
        train,
        validation,
        config=_model_config(config, seed=config.seed + 10_000, dropout=config.dropout),
    )
    dropout_passes = dropout_model.predict_stochastic(
        validation.values, passes=config.dropout_passes, seed=config.seed + 20_000
    )
    dropout_cpu = time.process_time() - dropout_start
    evidence = (
        MLPMethodEvidence(
            MLPUncertaintyMethod.DEEP_ENSEMBLE,
            estimate_mlp_uncertainty(
                ensemble_passes,
                confidence=config.confidence,
                max_prediction_values=config.budget.max_prediction_values,
            ),
            config.ensemble_members,
            config.ensemble_members,
            ensemble_cpu,
            sum(len(base64.b64decode(model.state_base64)) for model in ensemble_models),
            "pytorch-deep-ensemble-v1",
        ),
        MLPMethodEvidence(
            MLPUncertaintyMethod.MC_DROPOUT,
            estimate_mlp_uncertainty(
                dropout_passes,
                confidence=config.confidence,
                max_prediction_values=config.budget.max_prediction_values,
            ),
            1,
            config.dropout_passes,
            dropout_cpu,
            len(base64.b64decode(dropout_model.state_base64)),
            "pytorch-mc-dropout-v1",
        ),
    )
    return compare_mlp_uncertainty(
        validation.target_values,
        evidence,
        confidence=config.confidence,
        top_fraction=config.top_fraction,
        budget=config.budget,
    )


def _model_config(config: MLPUncertaintyRunConfig, *, seed: int, dropout: float) -> MLPConfig:
    return MLPConfig(
        task=ModelTask.REGRESSION,
        hidden_sizes=config.hidden_sizes,
        batch_size=config.batch_size,
        max_epochs=config.max_epochs,
        patience=config.patience,
        dropout=dropout,
        device=config.device,
        seed=seed,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

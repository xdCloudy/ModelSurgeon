"""Reconciled local feature attribution for supported surgeon models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from modelsurgeon.surgeon.models import (
    LightGBMSurgeonModel,
    LinearSurgeonModel,
    LogisticSurgeonModel,
    MLPSurgeonModel,
)

ATTRIBUTION_SCHEMA_VERSION: Final[int] = 1


class AttributionError(ValueError):
    """Raised when local contributions cannot satisfy their prediction contract."""


@dataclass(frozen=True, slots=True)
class FeatureProvenance:
    feature_name: str
    source_kind: str
    source_name: str
    category: str | None = None

    def __post_init__(self) -> None:
        if not self.feature_name or not self.source_kind or not self.source_name:
            raise AttributionError("feature attribution provenance cannot be blank")

    def to_record(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "source_kind": self.source_kind,
            "source_name": self.source_name,
            "category": self.category,
        }


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    provenance: FeatureProvenance
    input_value: float
    contribution: float
    missing: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.input_value) or not math.isfinite(self.contribution):
            raise AttributionError("feature attribution values must be finite")

    def to_record(self) -> dict[str, object]:
        return {
            "provenance": self.provenance.to_record(),
            "input_value": self.input_value,
            "contribution": self.contribution,
            "missing": self.missing,
        }


@dataclass(frozen=True, slots=True)
class PredictionAttribution:
    prediction: float
    bias: float
    contributions: tuple[FeatureContribution, ...]
    reconstructed_prediction: float
    absolute_reconciliation_error: float

    def __post_init__(self) -> None:
        values = (
            self.prediction,
            self.bias,
            self.reconstructed_prediction,
            self.absolute_reconciliation_error,
        )
        if any(not math.isfinite(value) for value in values):
            raise AttributionError("prediction attribution values must be finite")
        if self.absolute_reconciliation_error < 0:
            raise AttributionError("attribution reconciliation error cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "prediction": self.prediction,
            "bias": self.bias,
            "reconstructed_prediction": self.reconstructed_prediction,
            "absolute_reconciliation_error": self.absolute_reconciliation_error,
            "contributions": [item.to_record() for item in self.contributions],
        }


@dataclass(frozen=True, slots=True)
class AttributionReport:
    model_kind: str
    target_name: str
    technique: str
    output_space: str
    tolerance: float
    predictions: tuple[PredictionAttribution, ...]
    schema_version: int = ATTRIBUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != ATTRIBUTION_SCHEMA_VERSION
            or not self.model_kind
            or not self.target_name
            or not self.technique
            or not self.output_space
            or not self.predictions
            or not math.isfinite(self.tolerance)
            or self.tolerance < 0
        ):
            raise AttributionError("attribution report identity or limits are invalid")
        if any(item.absolute_reconciliation_error > self.tolerance for item in self.predictions):
            raise AttributionError("feature contributions do not reconcile with predictions")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_kind": self.model_kind,
            "target_name": self.target_name,
            "technique": self.technique,
            "output_space": self.output_space,
            "tolerance": self.tolerance,
            "predictions": [item.to_record() for item in self.predictions],
        }


@dataclass(frozen=True, slots=True)
class AttributionUnavailable:
    model_kind: str
    reason: str
    available_fallbacks: tuple[str, ...] = ("permutation",)
    schema_version: int = ATTRIBUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != ATTRIBUTION_SCHEMA_VERSION
            or not self.model_kind
            or not self.reason
            or not self.available_fallbacks
            or self.available_fallbacks != tuple(sorted(set(self.available_fallbacks)))
        ):
            raise AttributionError("attribution fallback record is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_kind": self.model_kind,
            "available": False,
            "reason": self.reason,
            "available_fallbacks": list(self.available_fallbacks),
        }


type AttributionResult = AttributionReport | AttributionUnavailable


def _provenance(name: str) -> FeatureProvenance:
    if name.startswith("num:") and len(name) > 4:
        return FeatureProvenance(name, "numeric", name[4:])
    if name.startswith("missing:") and len(name) > 8:
        return FeatureProvenance(name, "missing_indicator", name[8:])
    if name.startswith("cat:") and "=" in name[4:]:
        source, category = name[4:].split("=", 1)
        if source and category:
            return FeatureProvenance(name, "categorical", source, category)
    return FeatureProvenance(name, "derived", name)


def _missing_by_feature(feature_names: tuple[str, ...], row: Sequence[float]) -> dict[str, bool]:
    missing_sources = {
        name[8:]
        for name, value in zip(feature_names, row, strict=True)
        if name.startswith("missing:") and value > 0.5
    }
    return {
        name: (
            (name.startswith("missing:") and value > 0.5)
            or (name.startswith("num:") and name[4:] in missing_sources)
        )
        for name, value in zip(feature_names, row, strict=True)
    }


def _validated_rows(
    feature_names: tuple[str, ...], rows: Sequence[Sequence[float]]
) -> tuple[tuple[float, ...], ...]:
    if not rows:
        raise AttributionError("feature attribution requires at least one row")
    resolved: list[tuple[float, ...]] = []
    for row in rows:
        values = tuple(float(value) for value in row)
        if len(values) != len(feature_names) or any(not math.isfinite(value) for value in values):
            raise AttributionError("attribution rows must be finite and match feature schema")
        resolved.append(values)
    return tuple(resolved)


def _prediction(
    prediction: float,
    bias: float,
    feature_names: tuple[str, ...],
    row: tuple[float, ...],
    contributions: Sequence[float],
) -> PredictionAttribution:
    if len(contributions) != len(feature_names):
        raise AttributionError("contributions do not match the feature schema")
    missing = _missing_by_feature(feature_names, row)
    values = tuple(
        FeatureContribution(_provenance(name), value, float(contribution), missing[name])
        for name, value, contribution in zip(feature_names, row, contributions, strict=True)
    )
    reconstructed = math.fsum((bias, *(item.contribution for item in values)))
    return PredictionAttribution(
        prediction,
        bias,
        values,
        reconstructed,
        abs(prediction - reconstructed),
    )


def _linear(
    model: LinearSurgeonModel | LogisticSurgeonModel,
    rows: tuple[tuple[float, ...], ...],
    tolerance: float,
) -> AttributionReport:
    predictions = tuple(
        _prediction(
            model.intercept + math.fsum(
                coefficient * value
                for coefficient, value in zip(model.coefficients, row, strict=True)
            ),
            model.intercept,
            model.feature_names,
            row,
            tuple(
                coefficient * value
                for coefficient, value in zip(model.coefficients, row, strict=True)
            ),
        )
        for row in rows
    )
    logistic = isinstance(model, LogisticSurgeonModel)
    return AttributionReport(
        "logistic" if logistic else "linear",
        model.target_name,
        "exact_linear_contribution",
        "raw_logit" if logistic else "prediction",
        tolerance,
        predictions,
    )


def _tree(
    model: LightGBMSurgeonModel,
    rows: tuple[tuple[float, ...], ...],
    tolerance: float,
) -> AttributionReport:
    from modelsurgeon.surgeon import models

    lightgbm = models._lightgbm()
    booster = lightgbm.Booster(model_str=model.model_string)
    matrix = models._lightgbm_rows(rows, width=len(model.feature_names))
    raw_predictions = booster.predict(
        matrix,
        num_iteration=model.best_iteration,
        raw_score=True,
    )
    raw_contributions = booster.predict(
        matrix,
        num_iteration=model.best_iteration,
        pred_contrib=True,
    )
    predictions: list[PredictionAttribution] = []
    for row, prediction, contribution_row in zip(
        rows, raw_predictions, raw_contributions, strict=True
    ):
        values = tuple(float(value) for value in contribution_row)
        if len(values) != len(model.feature_names) + 1:
            raise AttributionError("LightGBM returned an incompatible contribution width")
        predictions.append(
            _prediction(
                float(prediction),
                values[-1],
                model.feature_names,
                row,
                values[:-1],
            )
        )
    return AttributionReport(
        "lightgbm",
        model.target_name,
        "lightgbm_tree_shap",
        "raw_score",
        tolerance,
        tuple(predictions),
    )


def attribute_predictions(
    model: object,
    rows: Sequence[Sequence[float]],
    *,
    tolerance: float = 1e-8,
) -> AttributionResult:
    """Return reconciled local contributions or an explicit supported fallback."""

    if not math.isfinite(tolerance) or tolerance < 0:
        raise AttributionError("attribution tolerance must be finite and non-negative")
    if isinstance(model, (LinearSurgeonModel, LogisticSurgeonModel)):
        return _linear(model, _validated_rows(model.feature_names, rows), tolerance)
    if isinstance(model, LightGBMSurgeonModel):
        return _tree(model, _validated_rows(model.feature_names, rows), tolerance)
    if isinstance(model, MLPSurgeonModel):
        return AttributionUnavailable(
            "mlp",
            "exact local additive attribution is unavailable for MLP surgeon models",
        )
    record = getattr(model, "to_record", None)
    kind = type(model).__name__
    if callable(record):
        candidate = record()
        if isinstance(candidate, dict) and isinstance(candidate.get("kind"), str):
            kind = candidate["kind"]
    return AttributionUnavailable(
        kind,
        "model type has no registered local attribution implementation",
    )

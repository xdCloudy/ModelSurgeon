"""Bounded, leakage-aware predictors for hardware deployment costs.

The predictor contract deliberately keeps the scientific result small and
auditable.  It trains a deterministic log-linear model, calibrates an
absolute-residual interval on validation examples, compares it with a mean
and artifact-size analytic baseline, and refuses unseen context by default.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter_ns
from typing import Final, cast

from modelsurgeon.datasets.hardware_cost import (
    HardwareCostDataset,
    HardwareCostExample,
    HardwareCostMetric,
    HardwareCostOutcome,
)

COST_PREDICTOR_SCHEMA_VERSION: Final[int] = 1
COST_PREDICTOR_PROTOCOL_REVISION: Final[str] = "hardware-cost-predictor-v1"
COST_PREDICTOR_TARGETS: Final[tuple[str, ...]] = (
    "decode_tokens_per_second",
    "latency_seconds",
    "load_time_seconds",
    "peak_ram_bytes",
    "peak_vram_bytes",
    "prefill_tokens_per_second",
)
_UNKNOWN_CATEGORY: Final[str] = "<unknown>"
_BASE_CATEGORICAL_FIELDS: Final[tuple[str, ...]] = (
    "architecture_state_id",
    "hardware_context_id",
    "hardware_profile_id",
    "model_family",
    "runtime_id",
    "runtime_revision",
)
_SIZE_SCALED_TARGETS: Final[frozenset[str]] = frozenset(
    {"load_time_seconds", "latency_seconds", "peak_ram_bytes", "peak_vram_bytes"}
)
_SHA256: Final[str] = "0123456789abcdef"


class CostPredictorError(ValueError):
    """Raised when a predictor contract or persisted card is unsafe."""


class CostPredictorOutcome(StrEnum):
    """Evidence state for one target predictor."""

    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class CostPredictionStatus(StrEnum):
    """Safe inference state for one requested target."""

    PREDICTED = "predicted"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CostPredictorError(f"{label} is required")
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CostPredictorError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        raise CostPredictorError(
            f"{label} must be finite and {'non-negative' if nonnegative else 'valid'}"
        )
    return result


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(character not in _SHA256 for character in result):
        raise CostPredictorError(f"{label} must be a lowercase SHA-256")
    return result


@dataclass(frozen=True, slots=True)
class CostPredictorConfig:
    """Preregistered, bounded training and evaluation controls."""

    target_names: tuple[str, ...] = COST_PREDICTOR_TARGETS
    minimum_train_examples: int = 3
    confidence: float = 0.95
    ridge: float = 1e-3
    seed: int = 0
    heldout_dimensions: tuple[str, ...] = (
        "hardware_profile_id",
        "model_revision",
        "source_artifact_digest",
    )

    def __post_init__(self) -> None:
        if self.target_names != tuple(sorted(set(self.target_names))):
            raise CostPredictorError("predictor targets must be unique and canonical")
        if not self.target_names or any(
            target not in COST_PREDICTOR_TARGETS for target in self.target_names
        ):
            raise CostPredictorError("predictor targets contain an unsupported metric")
        if self.minimum_train_examples < 2:
            raise CostPredictorError("predictors require at least two training examples")
        if not 0.5 < self.confidence < 1.0 or not math.isfinite(self.confidence):
            raise CostPredictorError("predictor confidence must be finite and in (0.5, 1)")
        if self.ridge < 0 or not math.isfinite(self.ridge):
            raise CostPredictorError("predictor ridge must be finite and non-negative")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64:
            raise CostPredictorError("predictor seed must be unsigned 64-bit")
        allowed = {
            "model_revision",
            "source_artifact_digest",
            "hardware_profile_id",
            "hardware_context_id",
            "runtime_id",
        }
        if not self.heldout_dimensions or any(
            item not in allowed for item in self.heldout_dimensions
        ):
            raise CostPredictorError("held-out dimensions are unsupported")
        if self.heldout_dimensions != tuple(sorted(set(self.heldout_dimensions))):
            raise CostPredictorError("held-out dimensions must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "target_names": list(self.target_names),
            "minimum_train_examples": self.minimum_train_examples,
            "confidence": self.confidence,
            "ridge": self.ridge,
            "seed": self.seed,
            "heldout_dimensions": list(self.heldout_dimensions),
            "model_families": ["log_ridge"],
            "baselines": ["mean", "artifact_size_analytic"],
            "metrics": [
                "coverage",
                "mae",
                "rmse",
                "rank_correlation",
                "p95_absolute_error",
                "model_size_bytes",
                "inference_cost_microseconds",
            ],
        }


@dataclass(frozen=True, slots=True)
class CostFeatureSchema:
    """Train-only feature vocabulary and numeric support bounds."""

    categorical_fields: tuple[str, ...]
    categorical_values: tuple[tuple[str, tuple[str, ...]], ...]
    numeric_fields: tuple[str, ...]
    numeric_bounds: tuple[tuple[str, float, float], ...]
    schema_version: int = COST_PREDICTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COST_PREDICTOR_SCHEMA_VERSION:
            raise CostPredictorError("unsupported predictor feature schema")
        if self.categorical_fields != tuple(sorted(set(self.categorical_fields))):
            raise CostPredictorError("categorical feature fields must be unique and canonical")
        names = tuple(name for name, _ in self.categorical_values)
        if names != self.categorical_fields:
            raise CostPredictorError("categorical feature values must align with fields")
        if any(
            values != tuple(sorted(set(values))) or _UNKNOWN_CATEGORY not in values
            for _, values in self.categorical_values
        ):
            raise CostPredictorError("categorical values must be canonical and include unknown")
        if self.numeric_fields != tuple(sorted(set(self.numeric_fields))):
            raise CostPredictorError("numeric feature fields must be unique and canonical")
        bound_names = tuple(name for name, _, _ in self.numeric_bounds)
        if bound_names != self.numeric_fields:
            raise CostPredictorError("numeric feature bounds must align with fields")
        if any(
            not math.isfinite(low) or not math.isfinite(high) or low > high
            for _, low, high in self.numeric_bounds
        ):
            raise CostPredictorError("numeric feature bounds are invalid")

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = list(self.numeric_fields)
        for field, values in self.categorical_values:
            names.extend(f"cat:{field}={value}" for value in values)
        return tuple(names)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "categorical_fields": list(self.categorical_fields),
            "categorical_values": {name: list(values) for name, values in self.categorical_values},
            "numeric_fields": list(self.numeric_fields),
            "numeric_bounds": {
                name: {"low": low, "high": high} for name, low, high in self.numeric_bounds
            },
            "feature_names": list(self.feature_names),
        }


@dataclass(frozen=True, slots=True)
class CostPrediction:
    target_name: str
    status: CostPredictionStatus
    value: float | None
    interval_low: float | None
    interval_high: float | None
    out_of_distribution: bool
    predictor_id: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is CostPredictionStatus.PREDICTED:
            if self.value is None or self.interval_low is None or self.interval_high is None:
                raise CostPredictorError("predicted results require value and interval")
            if self.interval_low > self.value or self.value > self.interval_high:
                raise CostPredictorError("prediction interval must contain the prediction")
            for value in (self.value, self.interval_low, self.interval_high):
                if not math.isfinite(value) or value < 0:
                    raise CostPredictorError("predictions must be finite and non-negative")
        elif (
            self.value is not None
            or self.interval_low is not None
            or self.interval_high is not None
        ):
            raise CostPredictorError("non-predicted results cannot carry numeric values")
        if self.status is CostPredictionStatus.OUT_OF_DISTRIBUTION and not self.out_of_distribution:
            raise CostPredictorError("out-of-distribution status must be flagged")
        if self.status is not CostPredictionStatus.PREDICTED and not self.reason:
            raise CostPredictorError("non-predicted results require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "target_name": self.target_name,
            "status": self.status.value,
            "value": self.value,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "out_of_distribution": self.out_of_distribution,
            "predictor_id": self.predictor_id,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CostPredictorScore:
    """Held-out regression and interval metrics for one predictor or baseline."""

    coverage: float | None
    mae: float | None
    rmse: float | None
    rank_correlation: float | None
    p95_absolute_error: float | None
    model_size_bytes: int
    inference_cost_microseconds: float
    sample_count: int
    reason: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("coverage", self.coverage),
            ("mae", self.mae),
            ("rmse", self.rmse),
            ("rank_correlation", self.rank_correlation),
            ("p95_absolute_error", self.p95_absolute_error),
            ("inference_cost_microseconds", self.inference_cost_microseconds),
        ):
            if value is not None and not math.isfinite(value):
                raise CostPredictorError(f"{name} must be finite when present")
        if self.coverage is not None and not 0 <= self.coverage <= 1:
            raise CostPredictorError("coverage must be within [0, 1]")
        if (
            self.model_size_bytes < 0
            or self.inference_cost_microseconds < 0
            or self.sample_count < 0
        ):
            raise CostPredictorError("score sizes and counts cannot be negative")
        if self.sample_count == 0 and self.reason is None:
            raise CostPredictorError("empty scores require an explicit reason")

    def to_record(self) -> dict[str, object]:
        return {
            "coverage": self.coverage,
            "mae": self.mae,
            "rmse": self.rmse,
            "rank_correlation": self.rank_correlation,
            "p95_absolute_error": self.p95_absolute_error,
            "model_size_bytes": self.model_size_bytes,
            "inference_cost_microseconds": self.inference_cost_microseconds,
            "sample_count": self.sample_count,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CostPredictorEvaluation:
    target_name: str
    outcome: CostPredictorOutcome
    predictor: CostPredictorScore | None
    mean_baseline: CostPredictorScore | None
    analytic_baseline: CostPredictorScore | None
    train_count: int
    calibration_count: int
    test_count: int
    beats_both_baselines: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if min(self.train_count, self.calibration_count, self.test_count) < 0:
            raise CostPredictorError("evaluation counts cannot be negative")
        if self.outcome is CostPredictorOutcome.MEASURED and not self.beats_both_baselines:
            raise CostPredictorError("measured predictor claims must beat both baselines")
        if self.outcome is CostPredictorOutcome.NEGATIVE_RESULT and self.beats_both_baselines:
            raise CostPredictorError("negative predictor results cannot claim a win")
        if self.outcome is not CostPredictorOutcome.MEASURED and self.reason is None:
            raise CostPredictorError("non-measured evaluations require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "target_name": self.target_name,
            "outcome": self.outcome.value,
            "predictor": None if self.predictor is None else self.predictor.to_record(),
            "mean_baseline": None if self.mean_baseline is None else self.mean_baseline.to_record(),
            "analytic_baseline": None
            if self.analytic_baseline is None
            else self.analytic_baseline.to_record(),
            "train_count": self.train_count,
            "calibration_count": self.calibration_count,
            "test_count": self.test_count,
            "beats_both_baselines": self.beats_both_baselines,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CostPredictorModel:
    """Immutable log-linear predictor with validation-calibrated residual radius."""

    target_name: str
    feature_names: tuple[str, ...]
    numeric_means: tuple[float, ...]
    numeric_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    residual_radius: float
    training_example_ids: tuple[str, ...]
    calibration_example_ids: tuple[str, ...]
    schema_version: int = COST_PREDICTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COST_PREDICTOR_SCHEMA_VERSION:
            raise CostPredictorError("unsupported cost predictor model schema")
        if len(self.numeric_means) != len(self.numeric_scales) or len(self.coefficients) != len(
            self.feature_names
        ):
            raise CostPredictorError("predictor model vectors do not align")
        if any(scale <= 0 or not math.isfinite(scale) for scale in self.numeric_scales):
            raise CostPredictorError("predictor numeric scales must be finite and positive")
        values = (*self.numeric_means, *self.coefficients, self.intercept, self.residual_radius)
        if any(not math.isfinite(value) for value in values) or self.residual_radius < 0:
            raise CostPredictorError("predictor model parameters must be finite")
        if not self.training_example_ids or not self.calibration_example_ids:
            raise CostPredictorError("predictor models require training and calibration evidence")

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "feature_names": list(self.feature_names),
            "numeric_means": list(self.numeric_means),
            "numeric_scales": list(self.numeric_scales),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "residual_radius": self.residual_radius,
            "training_example_ids": list(self.training_example_ids),
            "calibration_example_ids": list(self.calibration_example_ids),
        }

    @property
    def predictor_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"cost_predictor_{digest}"

    @property
    def model_size_bytes(self) -> int:
        return len(_canonical(self._identity_record()).encode("utf-8"))

    @property
    def inference_cost_microseconds(self) -> float:
        # Stable operation-count proxy; wall-clock timing is retained separately
        # by the smoke test and is intentionally not part of the identity.
        return float(2 + len(self.coefficients) * 2)

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["predictor_id"] = self.predictor_id
        record["model_size_bytes"] = self.model_size_bytes
        record["inference_cost_microseconds"] = self.inference_cost_microseconds
        return record


@dataclass(frozen=True, slots=True)
class CostPredictorStudy:
    """Complete retained predictor card, including negative and unsupported cells."""

    dataset_id: str
    config: CostPredictorConfig
    feature_schema: CostFeatureSchema
    models: tuple[CostPredictorModel, ...]
    evaluations: tuple[CostPredictorEvaluation, ...]
    protocol_revision: str = COST_PREDICTOR_PROTOCOL_REVISION
    schema_version: int = COST_PREDICTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.dataset_id, "dataset ID")
        if self.schema_version != COST_PREDICTOR_SCHEMA_VERSION:
            raise CostPredictorError("unsupported predictor study schema")
        if self.protocol_revision != COST_PREDICTOR_PROTOCOL_REVISION:
            raise CostPredictorError("unsupported predictor protocol revision")
        model_targets = tuple(model.target_name for model in self.models)
        evaluation_targets = tuple(item.target_name for item in self.evaluations)
        if model_targets != tuple(sorted(set(model_targets))):
            raise CostPredictorError("predictor models must be unique and canonical")
        if evaluation_targets != tuple(sorted(set(evaluation_targets))):
            raise CostPredictorError("predictor evaluations must be unique and canonical")

    @property
    def outcome(self) -> CostPredictorOutcome:
        outcomes = tuple(item.outcome for item in self.evaluations)
        if any(item is CostPredictorOutcome.MEASURED for item in outcomes):
            return CostPredictorOutcome.MEASURED
        if any(item is CostPredictorOutcome.NEGATIVE_RESULT for item in outcomes):
            return CostPredictorOutcome.NEGATIVE_RESULT
        return CostPredictorOutcome.UNSUPPORTED

    @property
    def study_id(self) -> str:
        identity = {
            "schema_version": self.schema_version,
            "protocol_revision": self.protocol_revision,
            "dataset_id": self.dataset_id,
            "config": self.config.to_record(),
            "feature_schema": self.feature_schema.to_record(),
            "models": [model.to_record() for model in self.models],
            "evaluations": [item.to_record() for item in self.evaluations],
        }
        digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        return f"cost_predictor_study_{digest}"

    def model(self, target_name: str) -> CostPredictorModel:
        for model in self.models:
            if model.target_name == target_name:
                return model
        raise KeyError(target_name)

    def predict(
        self,
        example: HardwareCostExample,
        target_name: str,
        *,
        allow_out_of_distribution: bool = False,
    ) -> CostPrediction:
        try:
            model = self.model(target_name)
        except KeyError:
            return CostPrediction(
                target_name,
                CostPredictionStatus.UNSUPPORTED,
                None,
                None,
                None,
                False,
                None,
                "no predictor was retained for this target",
            )
        vector, reasons = _encode_features(self.feature_schema, example)
        if reasons and not allow_out_of_distribution:
            return CostPrediction(
                target_name,
                CostPredictionStatus.OUT_OF_DISTRIBUTION,
                None,
                None,
                None,
                True,
                model.predictor_id,
                "; ".join(reasons),
            )
        if not vector:
            return CostPrediction(
                target_name,
                CostPredictionStatus.UNKNOWN,
                None,
                None,
                None,
                bool(reasons),
                model.predictor_id,
                "feature encoding produced no usable values",
            )
        normalized = _normalize(vector, model.numeric_means, model.numeric_scales)
        log_value = model.intercept + math.fsum(
            coefficient * feature
            for coefficient, feature in zip(model.coefficients, normalized, strict=True)
        )
        value = max(0.0, math.expm1(log_value))
        low = max(0.0, math.expm1(log_value - model.residual_radius))
        high = max(value, math.expm1(log_value + model.residual_radius))
        if reasons:
            low = 0.0
            high = max(
                value,
                math.expm1(min(700.0, log_value + model.residual_radius + 4.0)),
            )
        status = CostPredictionStatus.PREDICTED
        return CostPrediction(
            target_name,
            status,
            value,
            low,
            high,
            bool(reasons),
            model.predictor_id,
            "; ".join(reasons) if reasons else None,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "protocol_revision": self.protocol_revision,
            "dataset_id": self.dataset_id,
            "outcome": self.outcome.value,
            "config": self.config.to_record(),
            "feature_schema": self.feature_schema.to_record(),
            "models": [model.to_record() for model in self.models],
            "evaluations": [item.to_record() for item in self.evaluations],
        }


@dataclass(frozen=True, slots=True)
class _RawFeatures:
    categorical: tuple[tuple[str, str], ...]
    numeric: tuple[tuple[str, float], ...]


def _raw_features(example: HardwareCostExample) -> _RawFeatures:
    categorical = {
        "model_family": example.model_family,
        "architecture_state_id": example.architecture_state_id,
        "hardware_profile_id": example.hardware_profile_id,
        "hardware_context_id": example.hardware_context_id,
        "runtime_id": example.runtime_id,
        "runtime_revision": example.runtime_revision,
    }
    numeric: dict[str, float] = {}
    if example.artifact_size_bytes is not None:
        numeric["log_artifact_size_bytes"] = math.log1p(example.artifact_size_bytes)
    for name, raw in sorted(example.runtime_settings.items()):
        if isinstance(raw, bool):
            categorical[f"runtime_setting:{name}"] = str(raw).lower()
        elif (
            isinstance(raw, (int, float))
            and not isinstance(raw, bool)
            and math.isfinite(float(raw))
        ):
            numeric[f"runtime_setting:{name}"] = float(raw)
        elif isinstance(raw, str) and raw.strip():
            categorical[f"runtime_setting:{name}"] = raw
    return _RawFeatures(tuple(sorted(categorical.items())), tuple(sorted(numeric.items())))


def _build_feature_schema(examples: Sequence[HardwareCostExample]) -> CostFeatureSchema:
    if not examples:
        raise CostPredictorError("feature schema requires measured examples")
    categorical_fields = set(_BASE_CATEGORICAL_FIELDS)
    categorical_values: dict[str, set[str]] = {field: set() for field in categorical_fields}
    numeric_fields = {"log_artifact_size_bytes"}
    numeric_values: dict[str, list[float]] = {"log_artifact_size_bytes": []}
    for example in examples:
        raw = _raw_features(example)
        for name, value in raw.categorical:
            categorical_fields.add(name)
            categorical_values.setdefault(name, set()).add(value)
        for name, numeric_value in raw.numeric:
            numeric_fields.add(name)
            numeric_values.setdefault(name, []).append(numeric_value)
    categorical = tuple(sorted(categorical_fields))
    values = tuple(
        (field, tuple(sorted(categorical_values.get(field, set()) | {_UNKNOWN_CATEGORY})))
        for field in categorical
    )
    numeric = tuple(sorted(numeric_fields))
    bounds = tuple(
        (
            field,
            min(numeric_values.get(field, [0.0])),
            max(numeric_values.get(field, [0.0])),
        )
        for field in numeric
    )
    return CostFeatureSchema(categorical, values, numeric, bounds)


def _encode_features(
    schema: CostFeatureSchema,
    example: HardwareCostExample,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    raw = _raw_features(example)
    categorical = dict(raw.categorical)
    numeric = dict(raw.numeric)
    reasons: list[str] = []
    output: list[float] = []
    for field in schema.numeric_fields:
        if field not in numeric:
            if field == "log_artifact_size_bytes":
                reasons.append("artifact size is unavailable")
            output.append(0.0)
            continue
        value = numeric[field]
        output.append(value)
        bound = next(item for item in schema.numeric_bounds if item[0] == field)
        if value < bound[1] or value > bound[2]:
            reasons.append(f"{field} is outside the training range")
    for field, values in schema.categorical_values:
        actual = categorical.get(field, _UNKNOWN_CATEGORY)
        if actual not in values or actual == _UNKNOWN_CATEGORY:
            reasons.append(f"unseen {field}: {actual}")
            actual = _UNKNOWN_CATEGORY
        output.extend(1.0 if actual == value else 0.0 for value in values)
    return tuple(output), tuple(sorted(set(reasons)))


def _normalize(
    values: Sequence[float], means: Sequence[float], scales: Sequence[float]
) -> tuple[float, ...]:
    return tuple(
        (value - mean) / scale if index < len(means) else value
        for index, (value, mean, scale) in enumerate(zip(values, means, scales, strict=False))
    )


def _fit_linear(
    rows: Sequence[Sequence[float]], targets: Sequence[float], ridge: float
) -> tuple[float, tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    if len(rows) != len(targets) or not rows:
        raise CostPredictorError("linear fitting requires aligned non-empty rows")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise CostPredictorError("linear fitting rows have inconsistent widths")
    means = tuple(sum(row[index] for row in rows) / len(rows) for index in range(width))
    scales = tuple(
        max(math.sqrt(sum((row[index] - means[index]) ** 2 for row in rows) / len(rows)), 1.0)
        for index in range(width)
    )
    normalized = [_normalize(row, means, scales) for row in rows]
    augmented = [[1.0, *row] for row in normalized]
    dimension = width + 1
    matrix = [[0.0 for _ in range(dimension + 1)] for _ in range(dimension)]
    for augmented_row, target in zip(augmented, targets, strict=True):
        for left in range(dimension):
            matrix[left][dimension] += augmented_row[left] * target
            for right in range(dimension):
                matrix[left][right] += augmented_row[left] * augmented_row[right]
    for index in range(1, dimension):
        matrix[index][index] += ridge
    for pivot in range(dimension):
        pivot_row = max(range(pivot, dimension), key=lambda row: abs(matrix[row][pivot]))
        if abs(matrix[pivot_row][pivot]) < 1e-12:
            matrix[pivot][pivot] += 1e-8
            pivot_row = pivot
        matrix[pivot], matrix[pivot_row] = matrix[pivot_row], matrix[pivot]
        scale = matrix[pivot][pivot]
        for column in range(pivot, dimension + 1):
            matrix[pivot][column] /= scale
        for elimination_row in range(dimension):
            if elimination_row == pivot:
                continue
            factor = matrix[elimination_row][pivot]
            for column in range(pivot, dimension + 1):
                matrix[elimination_row][column] -= factor * matrix[pivot][column]
    solution = tuple(matrix[index][dimension] for index in range(dimension))
    if any(not math.isfinite(value) for value in solution):
        raise CostPredictorError("linear fitting produced non-finite coefficients")
    return solution[0], solution[1:], means, scales


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise CostPredictorError("quantiles require observations")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _rank(values: Sequence[float]) -> tuple[float, ...]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        value = (position + end - 1) / 2.0
        for index in range(position, end):
            output[ordered[index][0]] = value
        position = end
    return tuple(output)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return None if denominator == 0 else numerator / denominator


def _metric(example: HardwareCostExample, target_name: str) -> HardwareCostMetric | None:
    return next((item for item in example.metrics if item.name == target_name), None)


def _target(example: HardwareCostExample, target_name: str) -> float:
    metric = _metric(example, target_name)
    if metric is None:
        raise CostPredictorError(f"example is missing target metric {target_name}")
    return metric.median


def _mean_baseline(target_name: str, train: Sequence[HardwareCostExample]) -> float:
    return sum(_target(example, target_name) for example in train) / len(train)


def _analytic_baseline(
    target_name: str,
    example: HardwareCostExample,
    train: Sequence[HardwareCostExample],
) -> float:
    values = [_target(item, target_name) for item in train]
    baseline = _quantile(values, 0.5)
    if example.artifact_size_bytes is None:
        raise CostPredictorError("analytic baseline requires artifact size")
    sizes = [item.artifact_size_bytes for item in train if item.artifact_size_bytes is not None]
    median_size = _quantile([float(size) for size in sizes], 0.5)
    ratio = float(example.artifact_size_bytes) / median_size
    if target_name in _SIZE_SCALED_TARGETS:
        return max(0.0, baseline * ratio)
    return max(0.0, baseline / max(ratio, 1e-12))


def _score(
    actual: Sequence[float],
    predicted: Sequence[float],
    intervals: Sequence[tuple[float, float]] | None,
    *,
    model_size_bytes: int,
    inference_cost_microseconds: float,
) -> CostPredictorScore:
    if not actual:
        return CostPredictorScore(
            None,
            None,
            None,
            None,
            None,
            model_size_bytes,
            inference_cost_microseconds,
            0,
            "no held-out observations",
        )
    errors = [abs(left - right) for left, right in zip(actual, predicted, strict=True)]
    mae = sum(errors) / len(errors)
    rmse = math.sqrt(
        sum((left - right) ** 2 for left, right in zip(actual, predicted, strict=True))
        / len(actual)
    )
    coverage = None
    if intervals is not None:
        coverage = sum(
            low <= value <= high for value, (low, high) in zip(actual, intervals, strict=True)
        ) / len(actual)
    return CostPredictorScore(
        coverage,
        mae,
        rmse,
        _correlation(_rank(actual), _rank(predicted)),
        _quantile(errors, 0.95),
        model_size_bytes,
        inference_cost_microseconds,
        len(actual),
    )


def _partition_examples(
    dataset: HardwareCostDataset,
) -> tuple[
    tuple[HardwareCostExample, ...],
    tuple[HardwareCostExample, ...],
    tuple[HardwareCostExample, ...],
]:
    mapping = {group: "train" for group in dataset.split.train_group_ids}
    mapping.update({group: "validation" for group in dataset.split.validation_group_ids})
    mapping.update({group: "test" for group in dataset.split.test_group_ids})
    partitions: dict[str, list[HardwareCostExample]] = {"train": [], "validation": [], "test": []}
    for example in dataset.examples:
        partition = mapping.get(example.lineage_group_id)
        if partition is None:
            raise CostPredictorError(
                f"example lineage is absent from predictor split: {example.lineage_group_id}"
            )
        if example.outcome is HardwareCostOutcome.MEASURED:
            partitions[partition].append(example)
    return tuple(partitions["train"]), tuple(partitions["validation"]), tuple(partitions["test"])


def _leakage_reasons(
    train: Sequence[HardwareCostExample],
    validation: Sequence[HardwareCostExample],
    test: Sequence[HardwareCostExample],
    dimensions: Sequence[str],
) -> tuple[str, ...]:
    partitions = (train, validation, test)
    reasons: list[str] = []
    for dimension in dimensions:
        values = [
            {cast(str, getattr(example, dimension)) for example in partition}
            for partition in partitions
        ]
        if values[0] & values[1] or values[0] & values[2] or values[1] & values[2]:
            reasons.append(f"{dimension} overlaps predictor partitions")
    return tuple(reasons)


def fit_cost_predictors(
    dataset: HardwareCostDataset,
    *,
    config: CostPredictorConfig | None = None,
) -> CostPredictorStudy:
    """Fit all requested target predictors and retain every evidence boundary."""

    resolved_config = CostPredictorConfig() if config is None else config
    train, validation, test = _partition_examples(dataset)
    feature_schema = (
        _build_feature_schema(train)
        if train
        else CostFeatureSchema(
            _BASE_CATEGORICAL_FIELDS,
            tuple((field, (_UNKNOWN_CATEGORY,)) for field in _BASE_CATEGORICAL_FIELDS),
            ("log_artifact_size_bytes",),
            (("log_artifact_size_bytes", 0.0, 0.0),),
        )
    )
    leakage = _leakage_reasons(train, validation, test, resolved_config.heldout_dimensions)
    models: list[CostPredictorModel] = []
    evaluations: list[CostPredictorEvaluation] = []
    for target_name in resolved_config.target_names:
        if leakage:
            evaluations.append(
                CostPredictorEvaluation(
                    target_name,
                    CostPredictorOutcome.FAILED,
                    None,
                    None,
                    None,
                    len(train),
                    len(validation),
                    len(test),
                    False,
                    "; ".join(leakage),
                )
            )
            continue
        if len(train) < resolved_config.minimum_train_examples or not validation or not test:
            evaluations.append(
                CostPredictorEvaluation(
                    target_name,
                    CostPredictorOutcome.UNSUPPORTED,
                    None,
                    None,
                    None,
                    len(train),
                    len(validation),
                    len(test),
                    False,
                    "insufficient train, calibration, or test evidence",
                )
            )
            continue
        try:
            train_vectors = [_encode_features(feature_schema, example)[0] for example in train]
            validation_vectors = [
                _encode_features(feature_schema, example)[0] for example in validation
            ]
            test_vectors = [_encode_features(feature_schema, example)[0] for example in test]
            train_targets = [math.log1p(_target(example, target_name)) for example in train]
            intercept, coefficients, means, scales = _fit_linear(
                train_vectors, train_targets, resolved_config.ridge
            )
            calibration_actual = tuple(_target(example, target_name) for example in validation)
            log_residuals = tuple(
                abs(
                    math.log1p(actual)
                    - (
                        intercept
                        + math.fsum(
                            coef * value
                            for coef, value in zip(
                                coefficients, _normalize(vector, means, scales), strict=True
                            )
                        )
                    )
                )
                for actual, vector in zip(calibration_actual, validation_vectors, strict=True)
            )
            radius = _quantile(log_residuals, resolved_config.confidence)
            model = CostPredictorModel(
                target_name,
                feature_schema.feature_names,
                means,
                scales,
                coefficients,
                intercept,
                radius,
                tuple(sorted(example.example_id for example in train)),
                tuple(sorted(example.example_id for example in validation)),
            )
            models.append(model)
            predictions = tuple(
                max(
                    0.0,
                    math.expm1(
                        intercept
                        + math.fsum(
                            coef * value
                            for coef, value in zip(
                                coefficients, _normalize(vector, means, scales), strict=True
                            )
                        )
                    ),
                )
                for vector in test_vectors
            )
            actual = tuple(_target(example, target_name) for example in test)
            intervals = tuple(
                (
                    max(
                        0.0,
                        math.expm1(
                            intercept
                            + math.fsum(
                                coef * value
                                for coef, value in zip(
                                    coefficients, _normalize(vector, means, scales), strict=True
                                )
                            )
                            - radius
                        ),
                    ),
                    max(
                        0.0,
                        math.expm1(
                            intercept
                            + math.fsum(
                                coef * value
                                for coef, value in zip(
                                    coefficients, _normalize(vector, means, scales), strict=True
                                )
                            )
                            + radius
                        ),
                    ),
                )
                for vector in test_vectors
            )
            mean_values = tuple(_mean_baseline(target_name, train) for _ in test)
            analytic_values = tuple(
                _analytic_baseline(target_name, example, train) for example in test
            )
            predictor_score = _score(
                actual,
                predictions,
                intervals,
                model_size_bytes=model.model_size_bytes,
                inference_cost_microseconds=model.inference_cost_microseconds,
            )
            mean_score = _score(
                actual, mean_values, None, model_size_bytes=0, inference_cost_microseconds=1.0
            )
            analytic_score = _score(
                actual, analytic_values, None, model_size_bytes=0, inference_cost_microseconds=2.0
            )
            beats = (
                predictor_score.mae is not None
                and mean_score.mae is not None
                and analytic_score.mae is not None
                and predictor_score.mae < mean_score.mae
                and predictor_score.mae < analytic_score.mae
            )
            evaluations.append(
                CostPredictorEvaluation(
                    target_name,
                    CostPredictorOutcome.MEASURED
                    if beats
                    else CostPredictorOutcome.NEGATIVE_RESULT,
                    predictor_score,
                    mean_score,
                    analytic_score,
                    len(train),
                    len(validation),
                    len(test),
                    beats,
                    None if beats else "log-ridge did not beat both preregistered baselines",
                )
            )
        except (CostPredictorError, ArithmeticError, KeyError) as error:
            evaluations.append(
                CostPredictorEvaluation(
                    target_name,
                    CostPredictorOutcome.FAILED,
                    None,
                    None,
                    None,
                    len(train),
                    len(validation),
                    len(test),
                    False,
                    str(error),
                )
            )
    return CostPredictorStudy(
        dataset.dataset_id,
        resolved_config,
        feature_schema,
        tuple(sorted(models, key=lambda item: item.target_name)),
        tuple(sorted(evaluations, key=lambda item: item.target_name)),
    )


def smoke_test_inference(study: CostPredictorStudy, example: HardwareCostExample) -> float:
    """Run one bounded inference pass and return elapsed microseconds for evidence."""

    start = perf_counter_ns()
    for target_name in study.config.target_names:
        study.predict(example, target_name)
    return (perf_counter_ns() - start) / 1000.0

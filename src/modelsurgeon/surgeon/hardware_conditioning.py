"""Hardware-conditioned surgeon utility predictions and held-out ablations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

HARDWARE_CONDITIONING_SCHEMA_VERSION: Final[int] = 1
HARDWARE_PROFILE_FEATURE_SCHEMA_VERSION: Final[int] = 1


class HardwareConditioningError(ValueError):
    """Raised when hardware-conditioned evidence is unsafe or incompatible."""


class HardwarePredictionStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class HardwareStudyStatus(StrEnum):
    MEASURED = "measured"
    NULL_RESULT = "null_result"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class HardwareProfileContext:
    profile_id: str
    runtime: str
    cpu_threads: int
    vram_gb: float
    gpu_available: bool
    offload_fraction: float
    quantization: str
    feature_schema_version: int = HARDWARE_PROFILE_FEATURE_SCHEMA_VERSION
    supported: bool = True

    def __post_init__(self) -> None:
        for label, value in (
            ("profile ID", self.profile_id),
            ("runtime", self.runtime),
            ("quantization", self.quantization),
        ):
            if not isinstance(value, str) or not value.strip():
                raise HardwareConditioningError(f"{label} is required")
        if self.cpu_threads <= 0 or not math.isfinite(self.vram_gb) or self.vram_gb < 0:
            raise HardwareConditioningError("hardware capacity values are invalid")
        if not 0 <= self.offload_fraction <= 1:
            raise HardwareConditioningError("offload fraction must be within [0, 1]")
        if self.feature_schema_version != HARDWARE_PROFILE_FEATURE_SCHEMA_VERSION:
            raise HardwareConditioningError("unsupported hardware feature schema")

    def feature_vector(self) -> tuple[float, ...]:
        return (
            float(self.cpu_threads),
            self.vram_gb,
            float(self.gpu_available),
            self.offload_fraction,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "runtime": self.runtime,
            "cpu_threads": self.cpu_threads,
            "vram_gb": self.vram_gb,
            "gpu_available": self.gpu_available,
            "offload_fraction": self.offload_fraction,
            "quantization": self.quantization,
            "feature_schema_version": self.feature_schema_version,
            "supported": self.supported,
        }


@dataclass(frozen=True, slots=True)
class HardwareConditionedSample:
    candidate_id: str
    lineage_group_id: str
    pre_mutation_features: tuple[float, ...]
    profile: HardwareProfileContext
    quality_target: float
    safety_target: float
    utility_target: float
    post_mutation_runtime_target: float | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.lineage_group_id or not self.pre_mutation_features:
            raise HardwareConditioningError(
                "samples require candidate, lineage, and state features"
            )
        values = (
            *self.pre_mutation_features,
            self.quality_target,
            self.safety_target,
            self.utility_target,
        )
        if any(not math.isfinite(value) for value in values):
            raise HardwareConditioningError("hardware-conditioned sample values must be finite")
        if self.post_mutation_runtime_target is not None and not math.isfinite(
            self.post_mutation_runtime_target
        ):
            raise HardwareConditioningError("post-mutation runtime target must be finite")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "lineage_group_id": self.lineage_group_id,
            "pre_mutation_features": list(self.pre_mutation_features),
            "profile": self.profile.to_record(),
            "quality_target": self.quality_target,
            "safety_target": self.safety_target,
            "utility_target": self.utility_target,
            "post_mutation_runtime_target": self.post_mutation_runtime_target,
        }


def _solve_ridge(
    rows: Sequence[tuple[float, ...]], targets: Sequence[float], ridge: float
) -> tuple[float, ...]:
    if not rows or len(rows) != len(targets):
        raise HardwareConditioningError("training rows and targets must be non-empty and aligned")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise HardwareConditioningError("training rows must have equal width")
    matrix = [[0.0 for _ in range(width + 1)] for _ in range(width)]
    for training_row, target in zip(rows, targets, strict=True):
        for left in range(width):
            for right in range(width):
                matrix[left][right] += training_row[left] * training_row[right]
            matrix[left][width] += training_row[left] * target
    for index in range(1, width):
        matrix[index][index] += ridge
    for pivot in range(width):
        best = max(range(pivot, width), key=lambda row: abs(matrix[row][pivot]))
        if abs(matrix[best][pivot]) < 1e-12:
            raise HardwareConditioningError("hardware-conditioned design matrix is singular")
        matrix[pivot], matrix[best] = matrix[best], matrix[pivot]
        scale = matrix[pivot][pivot]
        matrix[pivot] = [value / scale for value in matrix[pivot]]
        for matrix_row in range(width):
            if matrix_row == pivot:
                continue
            factor = matrix[matrix_row][pivot]
            matrix[matrix_row] = [
                left - factor * right
                for left, right in zip(matrix[matrix_row], matrix[pivot], strict=True)
            ]
    return tuple(matrix[index][width] for index in range(width))


@dataclass(frozen=True, slots=True)
class HardwareConditionedModel:
    feature_count: int
    uses_hardware_context: bool
    quality_coefficients: tuple[float, ...]
    safety_coefficients: tuple[float, ...]
    utility_coefficients: tuple[float, ...]
    training_profiles: tuple[str, ...]
    model_revision: str = "hardware-conditioned-linear-v1"
    schema_version: int = HARDWARE_CONDITIONING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        width = self.feature_count + 1
        if self.schema_version != HARDWARE_CONDITIONING_SCHEMA_VERSION:
            raise HardwareConditioningError("unsupported hardware-conditioned model schema")
        if any(
            len(values) != width
            for values in (
                self.quality_coefficients,
                self.safety_coefficients,
                self.utility_coefficients,
            )
        ):
            raise HardwareConditioningError("model coefficient width does not match feature count")
        if not self.training_profiles:
            raise HardwareConditioningError("model must retain training profile identities")

    def _features(
        self, state: tuple[float, ...], profile: HardwareProfileContext
    ) -> tuple[float, ...]:
        if len(state) != self.feature_count - (4 if self.uses_hardware_context else 0):
            raise HardwareConditioningError("state feature width does not match model")
        return (1.0, *state, *(profile.feature_vector() if self.uses_hardware_context else ()))

    def predict(
        self, state: tuple[float, ...], profile: HardwareProfileContext
    ) -> HardwarePrediction:
        if not profile.supported:
            return HardwarePrediction(
                HardwarePredictionStatus.UNSUPPORTED,
                None,
                None,
                None,
                None,
                "hardware profile is unsupported",
            )
        features = self._features(state, profile)
        return HardwarePrediction(
            HardwarePredictionStatus.SUPPORTED,
            sum(
                left * right
                for left, right in zip(self.quality_coefficients, features, strict=True)
            ),
            sum(
                left * right for left, right in zip(self.safety_coefficients, features, strict=True)
            ),
            sum(
                left * right
                for left, right in zip(self.utility_coefficients, features, strict=True)
            ),
            profile.profile_id,
            None,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_revision": self.model_revision,
            "feature_count": self.feature_count,
            "uses_hardware_context": self.uses_hardware_context,
            "quality_coefficients": list(self.quality_coefficients),
            "safety_coefficients": list(self.safety_coefficients),
            "utility_coefficients": list(self.utility_coefficients),
            "training_profiles": list(self.training_profiles),
        }


@dataclass(frozen=True, slots=True)
class HardwarePrediction:
    status: HardwarePredictionStatus
    quality: float | None
    safety: float | None
    utility: float | None
    profile_id: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class HardwareAblationReport:
    heldout_profiles: tuple[str, ...]
    conditional_mae: float | None
    baseline_mae: float | None
    context_value: float | None
    status: HardwareStudyStatus
    reason: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": HARDWARE_CONDITIONING_SCHEMA_VERSION,
            "heldout_profiles": list(self.heldout_profiles),
            "conditional_mae": self.conditional_mae,
            "baseline_mae": self.baseline_mae,
            "context_value": self.context_value,
            "status": self.status.value,
            "reason": self.reason,
        }


def _model_features(sample: HardwareConditionedSample, include_hardware: bool) -> tuple[float, ...]:
    return (
        1.0,
        *sample.pre_mutation_features,
        *(sample.profile.feature_vector() if include_hardware else ()),
    )


def fit_hardware_conditioned_model(
    samples: tuple[HardwareConditionedSample, ...],
    *,
    include_hardware_context: bool = True,
    excluded_profile_ids: tuple[str, ...] = (),
) -> HardwareConditionedModel:
    """Fit separate quality/safety/utility models using pre-mutation context only."""
    selected = tuple(
        item for item in samples if item.profile.profile_id not in excluded_profile_ids
    )
    if not selected:
        raise HardwareConditioningError("no training samples remain after profile exclusion")
    widths = {len(item.pre_mutation_features) for item in selected}
    if len(widths) != 1:
        raise HardwareConditioningError("pre-mutation feature widths must agree")
    state_width = widths.pop()
    rows = tuple(_model_features(item, include_hardware_context) for item in selected)
    model_width = state_width + (4 if include_hardware_context else 0)
    return HardwareConditionedModel(
        model_width,
        include_hardware_context,
        _solve_ridge(rows, tuple(item.quality_target for item in selected), 1e-6),
        _solve_ridge(rows, tuple(item.safety_target for item in selected), 1e-6),
        _solve_ridge(rows, tuple(item.utility_target for item in selected), 1e-6),
        tuple(sorted({item.profile.profile_id for item in selected})),
    )


def evaluate_hardware_ablation(
    conditional: HardwareConditionedModel,
    baseline: HardwareConditionedModel,
    samples: tuple[HardwareConditionedSample, ...],
    heldout_profile_ids: tuple[str, ...],
) -> HardwareAblationReport:
    """Compare conditional and hardware-agnostic utility error on held-out profiles."""
    heldout = tuple(item for item in samples if item.profile.profile_id in heldout_profile_ids)
    if not heldout:
        return HardwareAblationReport(
            heldout_profile_ids,
            None,
            None,
            None,
            HardwareStudyStatus.UNKNOWN,
            "no held-out profile evidence",
        )
    conditional_errors: list[float] = []
    baseline_errors: list[float] = []
    for item in heldout:
        conditional_prediction = conditional.predict(item.pre_mutation_features, item.profile)
        baseline_prediction = baseline.predict(item.pre_mutation_features, item.profile)
        if conditional_prediction.utility is None or baseline_prediction.utility is None:
            return HardwareAblationReport(
                heldout_profile_ids,
                None,
                None,
                None,
                HardwareStudyStatus.UNSUPPORTED,
                "held-out profile prediction was unsupported",
            )
        conditional_errors.append(abs(conditional_prediction.utility - item.utility_target))
        baseline_errors.append(abs(baseline_prediction.utility - item.utility_target))
    conditional_mae = math.fsum(conditional_errors) / len(conditional_errors)
    baseline_mae = math.fsum(baseline_errors) / len(baseline_errors)
    context_value = baseline_mae - conditional_mae
    status = HardwareStudyStatus.MEASURED if context_value > 0 else HardwareStudyStatus.NULL_RESULT
    return HardwareAblationReport(
        heldout_profile_ids, conditional_mae, baseline_mae, context_value, status
    )

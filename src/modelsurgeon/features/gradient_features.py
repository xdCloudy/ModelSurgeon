"""Per-component gradient norms and first-order removal sensitivity features."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.features.weight_statistics import WeightTensor, _host_values
from modelsurgeon.graph import ComponentId
from modelsurgeon.instrumentation.gradients import GradientSnapshot

GRADIENT_FEATURE_EXTRACTOR_VERSION = "1"


class GradientFeatureError(ValueError):
    """Raised when gradient and weight observations cannot be reconciled safely."""


@dataclass(frozen=True, slots=True)
class GradientFeatures:
    component_id: ComponentId
    batch_count: int
    element_count: int
    weight_storage_dtype: str
    weight_source_device: str
    gradient_storage_dtypes: tuple[str, ...]
    gradient_source_devices: tuple[str, ...]
    gradient_l1_norm: float
    gradient_l2_norm: float
    gradient_max_magnitude: float
    weight_gradient_sum: float
    weight_gradient_abs_sum: float
    weight_gradient_l2_norm: float
    first_order_removal_estimate: float
    first_order_removal_magnitude: float
    aggregation: str = "mean_over_batches"
    compute_dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.batch_count <= 0 or self.element_count <= 0:
            raise GradientFeatureError("gradient features require observed gradients")
        if not self.gradient_storage_dtypes or not self.gradient_source_devices:
            raise GradientFeatureError("gradient provenance cannot be empty")
        numeric = (
            self.gradient_l1_norm,
            self.gradient_l2_norm,
            self.gradient_max_magnitude,
            self.weight_gradient_sum,
            self.weight_gradient_abs_sum,
            self.weight_gradient_l2_norm,
            self.first_order_removal_estimate,
            self.first_order_removal_magnitude,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise GradientFeatureError("gradient feature values must be finite")
        non_negative = (
            self.gradient_l1_norm,
            self.gradient_l2_norm,
            self.gradient_max_magnitude,
            self.weight_gradient_abs_sum,
            self.weight_gradient_l2_norm,
            self.first_order_removal_magnitude,
        )
        if any(value < 0.0 for value in non_negative):
            raise GradientFeatureError("gradient magnitudes cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "batch_count": self.batch_count,
            "element_count": self.element_count,
            "weight_storage_dtype": self.weight_storage_dtype,
            "weight_source_device": self.weight_source_device,
            "gradient_storage_dtypes": list(self.gradient_storage_dtypes),
            "gradient_source_devices": list(self.gradient_source_devices),
            "gradient_l1_norm": self.gradient_l1_norm,
            "gradient_l2_norm": self.gradient_l2_norm,
            "gradient_max_magnitude": self.gradient_max_magnitude,
            "weight_gradient_sum": self.weight_gradient_sum,
            "weight_gradient_abs_sum": self.weight_gradient_abs_sum,
            "weight_gradient_l2_norm": self.weight_gradient_l2_norm,
            "first_order_removal_estimate": self.first_order_removal_estimate,
            "first_order_removal_magnitude": self.first_order_removal_magnitude,
            "aggregation": self.aggregation,
            "compute_dtype": self.compute_dtype,
        }

    def feature_records(self) -> tuple[FeatureRecord, ...]:
        precision = PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION,
            self.weight_storage_dtype,
            self.compute_dtype,
        )
        metadata = (
            ("batch_count", self.batch_count),
            ("element_count", self.element_count),
            ("aggregation", self.aggregation),
            ("weight_source_device", self.weight_source_device),
            ("gradient_storage_dtypes", ",".join(self.gradient_storage_dtypes)),
            ("gradient_source_devices", ",".join(self.gradient_source_devices)),
        )
        values = (
            ("gradient_l1_norm", self.gradient_l1_norm),
            ("gradient_l2_norm", self.gradient_l2_norm),
            ("gradient_max_magnitude", self.gradient_max_magnitude),
            ("weight_gradient_sum", self.weight_gradient_sum),
            ("weight_gradient_abs_sum", self.weight_gradient_abs_sum),
            ("weight_gradient_l2_norm", self.weight_gradient_l2_norm),
            ("first_order_removal_estimate", self.first_order_removal_estimate),
            ("first_order_removal_magnitude", self.first_order_removal_magnitude),
        )
        return tuple(
            FeatureRecord(
                self.component_id,
                name,
                FeatureKind.SCALAR,
                value,
                self.compute_dtype,
                "gradient_features",
                GRADIENT_FEATURE_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            )
            for name, value in values
        )


@dataclass(frozen=True, slots=True)
class GradientFeatureOutcome:
    features: GradientFeatures | None
    missing_reason: str | None

    def __post_init__(self) -> None:
        if (self.features is None) == (self.missing_reason is None):
            raise GradientFeatureError(
                "gradient outcome requires either features or a missing reason"
            )

    @property
    def available(self) -> bool:
        return self.features is not None

    def to_record(self) -> dict[str, object]:
        return {
            "available": self.available,
            "missing_reason": self.missing_reason,
            "features": None if self.features is None else self.features.to_record(),
        }


def extract_gradient_features(
    component_id: ComponentId,
    weight: WeightTensor,
    gradients: Iterable[GradientSnapshot],
) -> GradientFeatureOutcome:
    """Aggregate analytic first-order features over observed calibration batches."""

    weight_values, weight_shape, weight_dtype, weight_device = _host_values(weight)
    snapshots = tuple(gradients)
    if not snapshots:
        return GradientFeatureOutcome(None, "no gradients observed for selected component")

    l1_values: list[float] = []
    l2_values: list[float] = []
    max_values: list[float] = []
    dot_values: list[float] = []
    abs_dot_values: list[float] = []
    wg_l2_values: list[float] = []
    removal_values: list[float] = []
    removal_magnitudes: list[float] = []
    gradient_dtypes: set[str] = set()
    gradient_devices: set[str] = set()

    for snapshot in snapshots:
        if snapshot.component_id != component_id:
            raise GradientFeatureError("gradient snapshot component identity does not match")
        if snapshot.shape != weight_shape or len(snapshot.values) != len(weight_values):
            raise GradientFeatureError("gradient snapshot shape does not match weight tensor")
        products = tuple(
            weight_value * gradient_value
            for weight_value, gradient_value in zip(
                weight_values,
                snapshot.values,
                strict=True,
            )
        )
        gradient_squares = math.fsum(value * value for value in snapshot.values)
        dot = math.fsum(products)
        l1_values.append(math.fsum(abs(value) for value in snapshot.values))
        l2_values.append(math.sqrt(gradient_squares))
        max_values.append(max(abs(value) for value in snapshot.values))
        dot_values.append(dot)
        abs_dot_values.append(math.fsum(abs(value) for value in products))
        wg_l2_values.append(math.sqrt(math.fsum(value * value for value in products)))
        removal = -dot
        removal_values.append(removal)
        removal_magnitudes.append(abs(removal))
        gradient_dtypes.add(snapshot.storage_dtype)
        gradient_devices.add(snapshot.source_device)

    batch_count = len(snapshots)

    def average(values: list[float]) -> float:
        return math.fsum(values) / batch_count

    return GradientFeatureOutcome(
        GradientFeatures(
            component_id=component_id,
            batch_count=batch_count,
            element_count=len(weight_values),
            weight_storage_dtype=weight_dtype,
            weight_source_device=weight_device,
            gradient_storage_dtypes=tuple(sorted(gradient_dtypes)),
            gradient_source_devices=tuple(sorted(gradient_devices)),
            gradient_l1_norm=average(l1_values),
            gradient_l2_norm=average(l2_values),
            gradient_max_magnitude=average(max_values),
            weight_gradient_sum=average(dot_values),
            weight_gradient_abs_sum=average(abs_dot_values),
            weight_gradient_l2_norm=average(wg_l2_values),
            first_order_removal_estimate=average(removal_values),
            first_order_removal_magnitude=average(removal_magnitudes),
        ),
        None,
    )

"""Basic deterministic per-tensor weight statistics with bounded host snapshots."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast

from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentId

WEIGHT_STATISTICS_EXTRACTOR_VERSION = "1"


class WeightStatisticsError(ValueError):
    """Raised when a weight tensor cannot be safely summarized."""


class _DetachedTensor(Protocol):
    shape: object
    dtype: object
    device: object

    def numel(self) -> int: ...

    def cpu(self) -> _DetachedTensor: ...

    def double(self) -> _DetachedTensor: ...

    def reshape(self, *shape: int) -> _DetachedTensor: ...

    def tolist(self) -> object: ...


class WeightTensor(Protocol):
    """Minimal PyTorch-compatible surface needed by the extractor."""

    def detach(self) -> _DetachedTensor: ...


@dataclass(frozen=True, slots=True)
class WeightStatistics:
    component_id: ComponentId
    count: int
    shape: tuple[int, ...]
    storage_dtype: str
    source_device: str
    minimum: float
    maximum: float
    mean: float
    variance: float
    standard_deviation: float
    l1_norm: float
    l2_norm: float
    frobenius_norm: float
    max_magnitude: float
    sparsity: float

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise WeightStatisticsError("weight statistics require at least one element")
        if any(dimension < 0 for dimension in self.shape):
            raise WeightStatisticsError("weight shape dimensions cannot be negative")
        if math.prod(self.shape) != self.count:
            raise WeightStatisticsError("weight shape does not match element count")
        if not self.storage_dtype or not self.source_device:
            raise WeightStatisticsError("weight dtype and source device are required")
        numeric = (
            self.minimum,
            self.maximum,
            self.mean,
            self.variance,
            self.standard_deviation,
            self.l1_norm,
            self.l2_norm,
            self.frobenius_norm,
            self.max_magnitude,
            self.sparsity,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise WeightStatisticsError("weight statistics must be finite")
        if self.variance < 0 or self.standard_deviation < 0:
            raise WeightStatisticsError("weight dispersion cannot be negative")
        if not 0.0 <= self.sparsity <= 1.0:
            raise WeightStatisticsError("weight sparsity must be within [0, 1]")

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "count": self.count,
            "shape": list(self.shape),
            "storage_dtype": self.storage_dtype,
            "source_device": self.source_device,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "variance": self.variance,
            "standard_deviation": self.standard_deviation,
            "l1_norm": self.l1_norm,
            "l2_norm": self.l2_norm,
            "frobenius_norm": self.frobenius_norm,
            "max_magnitude": self.max_magnitude,
            "sparsity": self.sparsity,
        }

    def feature_records(self) -> tuple[FeatureRecord, ...]:
        precision = PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION,
            self.storage_dtype,
            "float64",
        )
        metadata = (
            ("element_count", self.count),
            ("shape", "x".join(str(value) for value in self.shape)),
            ("source_device", self.source_device),
        )
        values = (
            ("weight_count", float(self.count)),
            ("weight_minimum", self.minimum),
            ("weight_maximum", self.maximum),
            ("weight_mean", self.mean),
            ("weight_variance", self.variance),
            ("weight_standard_deviation", self.standard_deviation),
            ("weight_l1_norm", self.l1_norm),
            ("weight_l2_norm", self.l2_norm),
            ("weight_frobenius_norm", self.frobenius_norm),
            ("weight_max_magnitude", self.max_magnitude),
            ("weight_sparsity", self.sparsity),
        )
        records = [
            FeatureRecord(
                self.component_id,
                name,
                FeatureKind.SCALAR,
                value,
                "float64",
                "weight_statistics",
                WEIGHT_STATISTICS_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            )
            for name, value in values
        ]
        if self.shape:
            records.append(
                FeatureRecord(
                    self.component_id,
                    "weight_shape",
                    FeatureKind.VECTOR,
                    tuple(float(value) for value in self.shape),
                    "int64",
                    "weight_statistics",
                    WEIGHT_STATISTICS_EXTRACTOR_VERSION,
                    precision,
                    metadata=metadata,
                )
            )
        return tuple(records)


def _normalized_dtype(value: object) -> str:
    text = str(value)
    return text.removeprefix("torch.")


def _shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, Iterable):
        raise WeightStatisticsError("weight tensor shape is not iterable")
    try:
        shape = tuple(int(dimension) for dimension in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise WeightStatisticsError("weight tensor shape is invalid") from error
    if any(dimension < 0 for dimension in shape):
        raise WeightStatisticsError("weight tensor shape contains a negative dimension")
    return shape


def _host_values(tensor: WeightTensor) -> tuple[tuple[float, ...], tuple[int, ...], str, str]:
    try:
        detached = tensor.detach()
        count = int(detached.numel())
        shape = _shape(detached.shape)
        storage_dtype = _normalized_dtype(detached.dtype)
        source_device = str(detached.device)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise WeightStatisticsError("object does not expose a valid tensor surface") from error
    if count <= 0:
        raise WeightStatisticsError("empty weight tensors cannot produce finite statistics")
    if math.prod(shape) != count:
        raise WeightStatisticsError("weight tensor shape does not match numel")
    try:
        host = detached.cpu().double().reshape(-1)
        raw = host.tolist()
    except (AttributeError, TypeError, ValueError, RuntimeError) as error:
        raise WeightStatisticsError("weight tensor could not be detached and copied to CPU") from error
    if not isinstance(raw, list):
        raise WeightStatisticsError("flattened weight tensor did not produce a value list")
    try:
        values = tuple(float(item) for item in cast("list[object]", raw))
    except (TypeError, ValueError, OverflowError) as error:
        raise WeightStatisticsError("weight tensor contains non-numeric values") from error
    if len(values) != count:
        raise WeightStatisticsError("copied weight values do not match numel")
    if any(not math.isfinite(value) for value in values):
        raise WeightStatisticsError("weight tensors with non-finite values are unsupported")
    return values, shape, storage_dtype, source_device


def extract_weight_statistics(component_id: ComponentId, tensor: WeightTensor) -> WeightStatistics:
    """Detach one weight, take a bounded host snapshot, and compute stable statistics."""

    values, shape, storage_dtype, source_device = _host_values(tensor)
    count = len(values)
    mean = math.fsum(values) / count
    centered_squares = math.fsum((value - mean) ** 2 for value in values)
    variance = max(0.0, centered_squares / count)
    sum_squares = math.fsum(value * value for value in values)
    l2_norm = math.sqrt(sum_squares)
    return WeightStatistics(
        component_id=component_id,
        count=count,
        shape=shape,
        storage_dtype=storage_dtype,
        source_device=source_device,
        minimum=min(values),
        maximum=max(values),
        mean=mean,
        variance=variance,
        standard_deviation=math.sqrt(variance),
        l1_norm=math.fsum(abs(value) for value in values),
        l2_norm=l2_norm,
        frobenius_norm=l2_norm,
        max_magnitude=max(abs(value) for value in values),
        sparsity=math.fsum(1.0 for value in values if value == 0.0) / count,
    )

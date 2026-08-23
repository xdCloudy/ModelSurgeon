"""Percentiles, histograms, skewness, and kurtosis for bounded weight snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass

from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.features.weight_statistics import WeightTensor, _host_values
from modelsurgeon.graph import ComponentId

WEIGHT_DISTRIBUTION_EXTRACTOR_VERSION = "1"


class WeightDistributionError(ValueError):
    """Raised when distribution features cannot be computed deterministically."""


@dataclass(frozen=True, slots=True)
class WeightDistributionConfig:
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    histogram_bins: int = 32

    def __post_init__(self) -> None:
        if not self.quantiles:
            raise WeightDistributionError("at least one percentile is required")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.quantiles):
            raise WeightDistributionError("percentiles must be finite values within [0, 1]")
        if tuple(sorted(self.quantiles)) != self.quantiles or len(set(self.quantiles)) != len(
            self.quantiles
        ):
            raise WeightDistributionError("percentiles must be strictly increasing")
        if self.histogram_bins < 2 or self.histogram_bins > 4096:
            raise WeightDistributionError("histogram bin count must be between 2 and 4096")


@dataclass(frozen=True, slots=True)
class WeightDistribution:
    component_id: ComponentId
    count: int
    shape: tuple[int, ...]
    storage_dtype: str
    source_device: str
    percentiles: tuple[tuple[float, float], ...]
    histogram_edges: tuple[float, ...]
    histogram_counts: tuple[int, ...]
    skewness: float
    excess_kurtosis: float
    percentile_interpolation: str = "linear"
    histogram_method: str = "equal_width"

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise WeightDistributionError("weight distribution requires observations")
        if not self.storage_dtype or not self.source_device:
            raise WeightDistributionError("weight dtype and source device are required")
        if len(self.histogram_edges) != len(self.histogram_counts) + 1:
            raise WeightDistributionError("histogram edges and counts do not align")
        if sum(self.histogram_counts) != self.count:
            raise WeightDistributionError("histogram counts do not reconcile to tensor size")
        if any(count < 0 for count in self.histogram_counts):
            raise WeightDistributionError("histogram counts cannot be negative")
        numeric = (
            *(value for _, value in self.percentiles),
            *self.histogram_edges,
            self.skewness,
            self.excess_kurtosis,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise WeightDistributionError("distribution values must be finite")

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "count": self.count,
            "shape": list(self.shape),
            "storage_dtype": self.storage_dtype,
            "source_device": self.source_device,
            "percentiles": [
                {"quantile": quantile, "value": value}
                for quantile, value in self.percentiles
            ],
            "percentile_interpolation": self.percentile_interpolation,
            "histogram_edges": list(self.histogram_edges),
            "histogram_counts": list(self.histogram_counts),
            "histogram_method": self.histogram_method,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
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
            ("percentile_interpolation", self.percentile_interpolation),
            ("histogram_method", self.histogram_method),
            ("histogram_bins", len(self.histogram_counts)),
        )
        quantile_text = ",".join(format(quantile, ".17g") for quantile, _ in self.percentiles)
        records = [
            FeatureRecord(
                self.component_id,
                "weight_percentiles",
                FeatureKind.VECTOR,
                tuple(value for _, value in self.percentiles),
                "float64",
                "weight_distribution",
                WEIGHT_DISTRIBUTION_EXTRACTOR_VERSION,
                precision,
                metadata=(*metadata, ("quantiles", quantile_text)),
            ),
            FeatureRecord(
                self.component_id,
                "weight_histogram_edges",
                FeatureKind.VECTOR,
                self.histogram_edges,
                "float64",
                "weight_distribution",
                WEIGHT_DISTRIBUTION_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
            FeatureRecord(
                self.component_id,
                "weight_histogram_counts",
                FeatureKind.VECTOR,
                tuple(float(value) for value in self.histogram_counts),
                "int64",
                "weight_distribution",
                WEIGHT_DISTRIBUTION_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
            FeatureRecord(
                self.component_id,
                "weight_skewness",
                FeatureKind.SCALAR,
                self.skewness,
                "float64",
                "weight_distribution",
                WEIGHT_DISTRIBUTION_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
            FeatureRecord(
                self.component_id,
                "weight_excess_kurtosis",
                FeatureKind.SCALAR,
                self.excess_kurtosis,
                "float64",
                "weight_distribution",
                WEIGHT_DISTRIBUTION_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
        ]
        return tuple(records)


def _percentile(sorted_values: tuple[float, ...], quantile: float) -> float:
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _histogram(values: tuple[float, ...], bins: int) -> tuple[tuple[float, ...], tuple[int, ...]]:
    low = min(values)
    high = max(values)
    if low == high:
        half_width = max(0.5, abs(low) * 0.5)
        low -= half_width
        high += half_width
    width = (high - low) / bins
    edges = (*(low + width * index for index in range(bins)), high)
    counts = [0] * bins
    for value in values:
        if value == high:
            index = bins - 1
        else:
            index = int((value - low) / width)
            index = min(max(index, 0), bins - 1)
        counts[index] += 1
    return edges, tuple(counts)


def extract_weight_distribution(
    component_id: ComponentId,
    tensor: WeightTensor,
    config: WeightDistributionConfig | None = None,
) -> WeightDistribution:
    """Compute fixed-schema distribution features from one detached host snapshot."""

    resolved = config or WeightDistributionConfig()
    values, shape, storage_dtype, source_device = _host_values(tensor)
    ordered = tuple(sorted(values))
    mean = math.fsum(values) / len(values)
    centered = tuple(value - mean for value in values)
    variance = math.fsum(value * value for value in centered) / len(centered)
    if variance == 0.0:
        skewness = 0.0
        excess_kurtosis = 0.0
    else:
        standard_deviation = math.sqrt(variance)
        third = math.fsum(value**3 for value in centered) / len(centered)
        fourth = math.fsum(value**4 for value in centered) / len(centered)
        skewness = third / standard_deviation**3
        excess_kurtosis = fourth / variance**2 - 3.0
    edges, counts = _histogram(values, resolved.histogram_bins)
    return WeightDistribution(
        component_id=component_id,
        count=len(values),
        shape=shape,
        storage_dtype=storage_dtype,
        source_device=source_device,
        percentiles=tuple(
            (quantile, _percentile(ordered, quantile)) for quantile in resolved.quantiles
        ),
        histogram_edges=edges,
        histogram_counts=counts,
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )

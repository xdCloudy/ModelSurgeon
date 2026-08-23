"""Masked per-component activation summary feature collection."""

from __future__ import annotations

from dataclasses import dataclass

from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    FeatureSampleContext,
    PrecisionProvenance,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.instrumentation.statistics import StatisticsConfig, StreamingStatistics


@dataclass(frozen=True, slots=True)
class ActivationBatch:
    sample_ids: tuple[str, ...]
    values: tuple[tuple[tuple[float, ...], ...], ...]
    token_mask: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        if not self.sample_ids or len(self.sample_ids) != len(self.values):
            raise ValueError("activation samples and batch rows must align")
        if len(self.token_mask) != len(self.values):
            raise ValueError("activation token mask and batch rows must align")
        widths: set[int] = set()
        for row, mask in zip(self.values, self.token_mask, strict=True):
            if len(row) != len(mask):
                raise ValueError("activation token mask length must match sequence length")
            widths.update(len(token) for token in row)
        if not widths or len(widths) != 1 or 0 in widths:
            raise ValueError("activation feature width must be positive and consistent")


class ActivationSummaryCollector:
    def __init__(
        self,
        component_id: ComponentId,
        precision: PrecisionProvenance,
        *,
        statistics: StatisticsConfig | None = None,
    ) -> None:
        self.component_id = component_id
        self.precision = precision
        self.statistics = StreamingStatistics(statistics)
        self.sample_ids: list[str] = []
        self.feature_width: int | None = None

    def update(self, batch: ActivationBatch) -> None:
        if len(set(batch.sample_ids)) != len(batch.sample_ids):
            raise ValueError("activation batch sample IDs must be unique")
        duplicates = set(self.sample_ids).intersection(batch.sample_ids)
        if duplicates:
            raise ValueError("activation sample IDs cannot be observed twice")
        for sample_id, row, mask in zip(
            batch.sample_ids, batch.values, batch.token_mask, strict=True
        ):
            self.sample_ids.append(sample_id)
            for token, include in zip(row, mask, strict=True):
                self.feature_width = len(token)
                if include:
                    self.statistics.update(token)

    def records(self, context: FeatureSampleContext) -> tuple[FeatureRecord, ...]:
        if tuple(self.sample_ids) != context.sample_ids:
            raise ValueError("activation sample context does not match observed sample IDs")
        snapshot = self.statistics.snapshot()
        metadata = (
            ("aggregation_axes", "batch,token,feature"),
            ("feature_width", self.feature_width),
            ("observation_count", snapshot.count),
        )
        values = {
            "mean": snapshot.mean,
            "variance": snapshot.variance,
            "rms": snapshot.rms,
            "minimum": snapshot.minimum,
            "maximum": snapshot.maximum,
            "zero_frequency": snapshot.zero_frequency,
            "activation_frequency": snapshot.activation_frequency,
        }
        records = [
            FeatureRecord(
                self.component_id,
                f"activation_{name}",
                FeatureKind.SCALAR,
                value,
                "float64",
                "activation_summary",
                "1",
                self.precision,
                context,
                metadata,
            )
            for name, value in values.items()
        ]
        records.append(
            FeatureRecord(
                self.component_id,
                "activation_percentiles",
                FeatureKind.VECTOR,
                tuple(value for _, value in snapshot.percentiles),
                "float64",
                "activation_summary",
                "1",
                self.precision,
                context,
                (*metadata, ("quantiles", "0.5,0.9,0.95,0.99")),
            )
        )
        return tuple(records)

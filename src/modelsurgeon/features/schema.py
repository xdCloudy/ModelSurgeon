"""Versioned, framework-neutral feature and precision provenance records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from modelsurgeon.graph import ComponentId

FEATURE_SCHEMA_VERSION: Literal[1] = 1
type FeaturePrimitive = str | int | float | bool | None


class FeatureKind(StrEnum):
    SCALAR = "scalar"
    VECTOR = "vector"


class PrecisionSource(StrEnum):
    DIRECT_QUANTIZED = "direct_quantized"
    LOCALLY_DEQUANTIZED = "locally_dequantized"
    HIGH_PRECISION = "high_precision"


@dataclass(frozen=True, slots=True)
class ErrorProvenance:
    absolute_error: float
    relative_error: float
    reference_dtype: str
    method: str

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.absolute_error)
            or not math.isfinite(self.relative_error)
            or self.absolute_error < 0
            or self.relative_error < 0
        ):
            raise ValueError("feature precision errors must be finite and non-negative")
        if not self.reference_dtype or not self.method:
            raise ValueError("error reference dtype and method are required")

    def to_record(self) -> dict[str, str | float]:
        return {
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
            "reference_dtype": self.reference_dtype,
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class PrecisionProvenance:
    source: PrecisionSource
    storage_dtype: str
    compute_dtype: str
    quantization: str | None = None
    codec_version: str | None = None
    error: ErrorProvenance | None = None

    def __post_init__(self) -> None:
        if not self.storage_dtype or not self.compute_dtype:
            raise ValueError("storage and compute dtypes are required")
        if self.source is not PrecisionSource.HIGH_PRECISION and not self.quantization:
            raise ValueError(f"{self.source.value} precision requires exact quantization")
        if self.source is PrecisionSource.HIGH_PRECISION and self.quantization is not None:
            raise ValueError("high-precision features cannot claim quantized storage")
        if self.source is PrecisionSource.LOCALLY_DEQUANTIZED and self.error is None:
            raise ValueError("locally dequantized features require measured error provenance")
        if self.codec_version is not None and not self.codec_version:
            raise ValueError("codec version cannot be blank")

    def to_record(self) -> dict[str, object]:
        return {
            "source": self.source.value,
            "storage_dtype": self.storage_dtype,
            "compute_dtype": self.compute_dtype,
            "quantization": self.quantization,
            "codec_version": self.codec_version,
            "error": None if self.error is None else self.error.to_record(),
        }


@dataclass(frozen=True, slots=True)
class FeatureSampleContext:
    dataset: str
    revision: str
    split: str
    sample_ids: tuple[str, ...]
    preprocessing_version: str
    tokenizer: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        required = (
            self.dataset,
            self.revision,
            self.split,
            self.preprocessing_version,
            self.tokenizer,
            self.tokenizer_revision,
        )
        if any(not value for value in required):
            raise ValueError("feature sample context identity fields are required")
        if not self.sample_ids or any(not sample_id for sample_id in self.sample_ids):
            raise ValueError("feature sample context requires non-empty sample IDs")
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError("feature sample IDs must be unique")

    def to_record(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "split": self.split,
            "sample_ids": list(self.sample_ids),
            "preprocessing_version": self.preprocessing_version,
            "tokenizer": self.tokenizer,
            "tokenizer_revision": self.tokenizer_revision,
        }


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    component_id: ComponentId
    name: str
    kind: FeatureKind
    value: float | tuple[float, ...]
    dtype: str
    extractor: str
    extractor_version: str
    precision: PrecisionProvenance
    sample_context: FeatureSampleContext | None = None
    metadata: tuple[tuple[str, FeaturePrimitive], ...] = ()
    schema_version: Literal[1] = FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported feature schema version {self.schema_version}")
        if not self.name or not self.dtype or not self.extractor or not self.extractor_version:
            raise ValueError("feature name, dtype, extractor, and extractor version are required")
        values: tuple[float, ...]
        if self.kind is FeatureKind.SCALAR:
            if isinstance(self.value, tuple):
                raise ValueError("scalar features require one float value")
            values = (self.value,)
        else:
            if not isinstance(self.value, tuple) or not self.value:
                raise ValueError("vector features require a non-empty float tuple")
            values = self.value
        if any(not isinstance(value, float) or not math.isfinite(value) for value in values):
            raise ValueError("feature values must be finite floats")
        keys = [key for key, _ in self.metadata]
        if any(not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("feature metadata keys must be non-empty and unique")

    def to_record(self) -> dict[str, object]:
        value: float | list[float]
        value = list(self.value) if isinstance(self.value, tuple) else self.value
        return {
            "schema_version": self.schema_version,
            "component_id": str(self.component_id),
            "name": self.name,
            "kind": self.kind.value,
            "value": value,
            "dtype": self.dtype,
            "extractor": self.extractor,
            "extractor_version": self.extractor_version,
            "precision": self.precision.to_record(),
            "sample_context": (
                None if self.sample_context is None else self.sample_context.to_record()
            ),
            "metadata": dict(self.metadata),
        }

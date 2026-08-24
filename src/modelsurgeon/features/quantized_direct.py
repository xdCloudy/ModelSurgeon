"""Safe features derived directly from encoded GGUF quantization blocks."""

from __future__ import annotations

import math
import statistics
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

from modelsurgeon.adapters.gguf.quantization import (
    QUANT_LAYOUTS,
    ByteOrder,
    CodecLayout,
    GGMLQuantizationType,
)
from modelsurgeon.features.schema import (
    ErrorProvenance,
    PrecisionProvenance,
    PrecisionSource,
)

DIRECT_QUANTIZED_FEATURE_VERSION = "1"


class DirectQuantizedFeatureError(ValueError):
    """Raised when encoded-block feature extraction violates a bounded contract."""


class DirectQuantizedFeatureName(StrEnum):
    BLOCK_COUNT = "block_count"
    ENCODED_BYTES_PER_ELEMENT = "encoded_bytes_per_element"
    PRIMARY_SCALE_ABS_MEAN = "primary_scale_abs_mean"
    WEIGHT_MEAN = "weight_mean"
    WEIGHT_VARIANCE = "weight_variance"
    SPARSITY = "sparsity"


class EncodedBlockSource(Protocol):
    quant_type: GGMLQuantizationType
    element_count: int
    byte_order: ByteOrder

    def read_block(self, index: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class BytesEncodedBlockSource:
    quant_type: GGMLQuantizationType
    element_count: int
    payload: bytes
    byte_order: ByteOrder = ByteOrder.LITTLE

    def __post_init__(self) -> None:
        layout = QUANT_LAYOUTS[self.quant_type]
        expected = layout.encoded_size(self.element_count)
        if len(self.payload) != expected:
            raise DirectQuantizedFeatureError(
                f"encoded payload has {len(self.payload)} bytes, expected {expected}"
            )

    def read_block(self, index: int) -> bytes:
        layout = QUANT_LAYOUTS[self.quant_type]
        total = self.element_count // layout.block_size
        if index < 0 or index >= total:
            raise DirectQuantizedFeatureError("encoded block index is out of range")
        start = index * layout.type_size
        return self.payload[start : start + layout.type_size]


@dataclass(frozen=True, slots=True)
class DirectQuantizedFeature:
    name: DirectQuantizedFeatureName
    value: float
    codec: GGMLQuantizationType
    total_blocks: int
    covered_blocks: int
    coverage_fraction: float
    estimated_error: float
    error_method: str
    source: str = "direct_encoded_blocks"

    def __post_init__(self) -> None:
        if self.total_blocks <= 0 or not 0 < self.covered_blocks <= self.total_blocks:
            raise DirectQuantizedFeatureError("quantized feature block coverage is invalid")
        numeric = (self.value, self.coverage_fraction, self.estimated_error)
        if any(not math.isfinite(item) for item in numeric):
            raise DirectQuantizedFeatureError("quantized feature values must be finite")
        expected_coverage = self.covered_blocks / self.total_blocks
        if not math.isclose(self.coverage_fraction, expected_coverage, rel_tol=0, abs_tol=1e-15):
            raise DirectQuantizedFeatureError("quantized feature coverage fraction is inconsistent")
        if self.estimated_error < 0 or not self.error_method:
            raise DirectQuantizedFeatureError("quantized feature error provenance is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "version": DIRECT_QUANTIZED_FEATURE_VERSION,
            "name": self.name.value,
            "value": self.value,
            "codec": self.codec.value,
            "total_blocks": self.total_blocks,
            "covered_blocks": self.covered_blocks,
            "coverage_fraction": self.coverage_fraction,
            "estimated_error": self.estimated_error,
            "error_method": self.error_method,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class LocalDecodeRequired:
    name: DirectQuantizedFeatureName
    codec: GGMLQuantizationType
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise DirectQuantizedFeatureError("decode-required outcomes need a reason")

    def to_record(self) -> dict[str, str]:
        return {
            "version": DIRECT_QUANTIZED_FEATURE_VERSION,
            "name": self.name.value,
            "codec": self.codec.value,
            "status": "local_decode_required",
            "reason": self.reason,
        }


type DirectQuantizedFeatureOutcome = DirectQuantizedFeature | LocalDecodeRequired


@dataclass(frozen=True, slots=True)
class EmpiricalQuantizedFeatureErrorModel:
    """Study-backed error parameters suitable for feature precision provenance."""

    codec: GGMLQuantizationType
    feature_name: str
    mean_absolute_error: float
    mean_relative_absolute_error: float
    maximum_absolute_error: float
    spearman_rank_correlation: float
    sample_count: int
    study_sha256: str

    @classmethod
    def from_study_record(
        cls,
        codec: GGMLQuantizationType,
        feature_name: str,
        record: Mapping[str, object],
        study_sha256: str,
    ) -> EmpiricalQuantizedFeatureErrorModel:
        keys = (
            "mean_absolute_error",
            "mean_relative_absolute_error",
            "maximum_absolute_error",
            "spearman_rank_correlation",
        )
        raw_values = tuple(record.get(key) for key in keys)
        raw_count = record.get("sample_count")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_values
            )
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
        ):
            raise DirectQuantizedFeatureError(
                "empirical quantized feature error model is malformed"
            )
        return cls(
            codec,
            feature_name,
            float(cast(int | float, raw_values[0])),
            float(cast(int | float, raw_values[1])),
            float(cast(int | float, raw_values[2])),
            float(cast(int | float, raw_values[3])),
            raw_count,
            study_sha256,
        )

    def __post_init__(self) -> None:
        numeric = (
            self.mean_absolute_error,
            self.mean_relative_absolute_error,
            self.maximum_absolute_error,
            self.spearman_rank_correlation,
        )
        if (
            not self.feature_name
            or any(not math.isfinite(value) for value in numeric)
            or min(numeric[:3]) < 0.0
            or not -1.0 <= self.spearman_rank_correlation <= 1.0
            or self.sample_count <= 0
            or len(self.study_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.study_sha256)
        ):
            raise DirectQuantizedFeatureError("empirical quantized feature error model is invalid")

    def precision_provenance(self, *, storage_dtype: str) -> PrecisionProvenance:
        return PrecisionProvenance(
            PrecisionSource.LOCALLY_DEQUANTIZED,
            storage_dtype,
            "float64",
            quantization=self.codec.value,
            codec_version=DIRECT_QUANTIZED_FEATURE_VERSION,
            error=ErrorProvenance(
                self.mean_absolute_error,
                self.mean_relative_absolute_error,
                "float64",
                f"matched_tensor_empirical_v1:{self.study_sha256}",
            ),
        )


_PRIMARY_SCALE_CODECS = frozenset(
    {
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
    }
)


def _block_count(source: EncodedBlockSource, layout: CodecLayout) -> int:
    if source.element_count <= 0 or source.element_count % layout.block_size != 0:
        raise DirectQuantizedFeatureError(
            f"{source.quant_type.value} element count must be a positive multiple of "
            f"{layout.block_size}"
        )
    return source.element_count // layout.block_size


def _sample_indices(total_blocks: int, max_blocks: int) -> tuple[int, ...]:
    if max_blocks <= 0:
        raise DirectQuantizedFeatureError("quantized feature sample budget must be positive")
    if max_blocks >= total_blocks:
        return tuple(range(total_blocks))
    return tuple((index * total_blocks) // max_blocks for index in range(max_blocks))


def _primary_scale_offset(layout: CodecLayout) -> int | None:
    for field in layout.fields:
        if field.name in {"delta_f16", "delta_min_f16x2"}:
            return field.offset
    return None


def _read_primary_scale(
    source: EncodedBlockSource,
    layout: CodecLayout,
    block_index: int,
) -> float:
    offset = _primary_scale_offset(layout)
    if offset is None:
        raise DirectQuantizedFeatureError("codec layout does not expose a direct primary scale")
    block = source.read_block(block_index)
    if len(block) != layout.type_size:
        raise DirectQuantizedFeatureError(
            f"encoded block {block_index} has {len(block)} bytes, expected {layout.type_size}"
        )
    prefix = "<" if source.byte_order is ByteOrder.LITTLE else ">"
    value = float(struct.unpack(f"{prefix}e", block[offset : offset + 2])[0])
    if not math.isfinite(value):
        raise DirectQuantizedFeatureError("encoded primary scale is non-finite")
    return value


def _sample_error(values: tuple[float, ...], full_coverage: bool) -> tuple[float, str]:
    if full_coverage:
        return 0.0, "exact_full_block_coverage"
    if len(values) >= 2:
        standard_error = statistics.stdev(values) / math.sqrt(len(values))
        return float(standard_error), "sample_standard_error"
    return abs(values[0]), "single_sample_absolute_proxy"


def extract_direct_quantized_feature(
    source: EncodedBlockSource,
    name: DirectQuantizedFeatureName,
    *,
    max_sample_blocks: int = 256,
) -> DirectQuantizedFeatureOutcome:
    """Extract one safe encoded-block feature or require bounded local decode explicitly."""

    layout = QUANT_LAYOUTS[source.quant_type]
    total_blocks = _block_count(source, layout)

    if name is DirectQuantizedFeatureName.BLOCK_COUNT:
        return DirectQuantizedFeature(
            name,
            float(total_blocks),
            source.quant_type,
            total_blocks,
            total_blocks,
            1.0,
            0.0,
            "exact_layout_geometry",
        )
    if name is DirectQuantizedFeatureName.ENCODED_BYTES_PER_ELEMENT:
        return DirectQuantizedFeature(
            name,
            layout.type_size / layout.block_size,
            source.quant_type,
            total_blocks,
            total_blocks,
            1.0,
            0.0,
            "exact_layout_geometry",
        )

    if name is DirectQuantizedFeatureName.PRIMARY_SCALE_ABS_MEAN:
        if source.quant_type not in _PRIMARY_SCALE_CODECS:
            return LocalDecodeRequired(
                name,
                source.quant_type,
                "codec does not have a v1 validated direct primary-scale statistic",
            )
        indices = _sample_indices(total_blocks, max_sample_blocks)
        values = tuple(abs(_read_primary_scale(source, layout, index)) for index in indices)
        error, method = _sample_error(values, len(indices) == total_blocks)
        return DirectQuantizedFeature(
            name,
            math.fsum(values) / len(values),
            source.quant_type,
            total_blocks,
            len(indices),
            len(indices) / total_blocks,
            error,
            method,
        )

    return LocalDecodeRequired(
        name,
        source.quant_type,
        (
            "feature depends on decoded weight values and cannot be inferred safely "
            "from encoded blocks"
        ),
    )

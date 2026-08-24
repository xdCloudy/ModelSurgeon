"""Exact GGUF quantization block-layout and codec contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from itertools import chain
from operator import mul
from typing import Protocol, runtime_checkable


class GGMLQuantizationType(StrEnum):
    F32 = "F32"
    F16 = "F16"
    BF16 = "BF16"
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q8_0 = "Q8_0"
    Q2_K = "Q2_K"
    Q3_K = "Q3_K"
    Q4_K = "Q4_K"
    Q5_K = "Q5_K"
    Q6_K = "Q6_K"
    Q8_K = "Q8_K"
    IQ2_XXS = "IQ2_XXS"
    IQ2_XS = "IQ2_XS"
    IQ3_XXS = "IQ3_XXS"
    IQ1_S = "IQ1_S"
    IQ4_NL = "IQ4_NL"
    IQ3_S = "IQ3_S"
    IQ2_S = "IQ2_S"
    IQ4_XS = "IQ4_XS"
    IQ1_M = "IQ1_M"


class QuantizationFamily(StrEnum):
    DENSE = "dense"
    LEGACY = "legacy"
    Q8 = "q8"
    K_QUANT = "k_quant"
    IQ = "iq"


class ByteOrder(StrEnum):
    LITTLE = "little"
    BIG = "big"


class AxisEditMode(StrEnum):
    BLOCK_VALUES = "block_values"
    WHOLE_SLICES = "whole_slices"


class CodecContractError(ValueError):
    """Base error for invalid shapes, blocks, or codec registration."""


class UnsupportedCodecError(CodecContractError):
    """Raised when no exact codec implementation is registered."""


@dataclass(frozen=True, slots=True)
class BlockField:
    """One exact byte range inside an encoded block."""

    name: str
    offset: int
    size: int

    def __post_init__(self) -> None:
        if not self.name or self.offset < 0 or self.size <= 0:
            raise ValueError("block fields require a name, non-negative offset, and positive size")


@dataclass(frozen=True, slots=True)
class CodecLayout:
    """Pinned logical and physical block layout for one concrete ggml_type."""

    quant_type: GGMLQuantizationType
    family: QuantizationFamily
    block_size: int
    type_size: int
    fields: tuple[BlockField, ...]
    layout_version: int = 1

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.type_size <= 0 or self.layout_version <= 0:
            raise ValueError("codec block, type, and layout versions must be positive")
        expected_offset = 0
        for field in self.fields:
            if field.offset != expected_offset:
                raise ValueError(
                    f"{self.quant_type.value} field {field.name!r} is not contiguous at "
                    f"offset {expected_offset}"
                )
            expected_offset += field.size
        if expected_offset != self.type_size:
            raise ValueError(
                f"{self.quant_type.value} fields occupy {expected_offset} bytes, "
                f"expected {self.type_size}"
            )

    def encoded_size(self, element_count: int) -> int:
        if element_count < 0 or element_count % self.block_size != 0:
            raise CodecContractError(
                f"{self.quant_type.value} element count must be a non-negative multiple "
                f"of {self.block_size}"
            )
        return element_count // self.block_size * self.type_size


def _fields(*parts: tuple[str, int]) -> tuple[BlockField, ...]:
    offset = 0
    result: list[BlockField] = []
    for name, size in parts:
        result.append(BlockField(name, offset, size))
        offset += size
    return tuple(result)


QUANT_LAYOUTS: dict[GGMLQuantizationType, CodecLayout] = {
    GGMLQuantizationType.F32: CodecLayout(
        GGMLQuantizationType.F32,
        QuantizationFamily.DENSE,
        1,
        4,
        _fields(("value", 4)),
    ),
    GGMLQuantizationType.F16: CodecLayout(
        GGMLQuantizationType.F16,
        QuantizationFamily.DENSE,
        1,
        2,
        _fields(("value", 2)),
    ),
    GGMLQuantizationType.BF16: CodecLayout(
        GGMLQuantizationType.BF16,
        QuantizationFamily.DENSE,
        1,
        2,
        _fields(("value", 2)),
    ),
    GGMLQuantizationType.Q8_0: CodecLayout(
        GGMLQuantizationType.Q8_0,
        QuantizationFamily.Q8,
        32,
        34,
        _fields(("delta_f16", 2), ("quants_i8", 32)),
    ),
    GGMLQuantizationType.Q2_K: CodecLayout(
        GGMLQuantizationType.Q2_K,
        QuantizationFamily.K_QUANT,
        256,
        84,
        _fields(("scales_mins_u4", 16), ("quants_u2", 64), ("delta_min_f16x2", 4)),
    ),
    GGMLQuantizationType.Q3_K: CodecLayout(
        GGMLQuantizationType.Q3_K,
        QuantizationFamily.K_QUANT,
        256,
        110,
        _fields(("high_mask", 32), ("quants_u2", 64), ("scales_i6", 12), ("delta_f16", 2)),
    ),
    GGMLQuantizationType.Q4_K: CodecLayout(
        GGMLQuantizationType.Q4_K,
        QuantizationFamily.K_QUANT,
        256,
        144,
        _fields(("delta_min_f16x2", 4), ("scales_mins_u6", 12), ("quants_u4", 128)),
    ),
    GGMLQuantizationType.Q5_K: CodecLayout(
        GGMLQuantizationType.Q5_K,
        QuantizationFamily.K_QUANT,
        256,
        176,
        _fields(
            ("delta_min_f16x2", 4),
            ("scales_mins_u6", 12),
            ("high_bits", 32),
            ("quants_u4", 128),
        ),
    ),
    GGMLQuantizationType.Q6_K: CodecLayout(
        GGMLQuantizationType.Q6_K,
        QuantizationFamily.K_QUANT,
        256,
        210,
        _fields(("low_bits", 128), ("high_bits", 64), ("scales_i8", 16), ("delta_f16", 2)),
    ),
    GGMLQuantizationType.Q8_K: CodecLayout(
        GGMLQuantizationType.Q8_K,
        QuantizationFamily.K_QUANT,
        256,
        292,
        _fields(("delta_f32", 4), ("quants_i8", 256), ("block_sums_i16", 32)),
    ),
    GGMLQuantizationType.IQ2_XXS: CodecLayout(
        GGMLQuantizationType.IQ2_XXS,
        QuantizationFamily.IQ,
        256,
        66,
        _fields(("delta_f16", 2), ("grid_sign_indexes", 64)),
    ),
    GGMLQuantizationType.IQ2_XS: CodecLayout(
        GGMLQuantizationType.IQ2_XS,
        QuantizationFamily.IQ,
        256,
        74,
        _fields(("delta_f16", 2), ("grid_sign_indexes", 64), ("scales_u4", 8)),
    ),
    GGMLQuantizationType.IQ3_XXS: CodecLayout(
        GGMLQuantizationType.IQ3_XXS,
        QuantizationFamily.IQ,
        256,
        98,
        _fields(("delta_f16", 2), ("grid_sign_indexes", 96)),
    ),
    GGMLQuantizationType.IQ1_S: CodecLayout(
        GGMLQuantizationType.IQ1_S,
        QuantizationFamily.IQ,
        256,
        50,
        _fields(("delta_f16", 2), ("grid_indexes", 32), ("high_bits_scales", 16)),
    ),
    GGMLQuantizationType.IQ4_NL: CodecLayout(
        GGMLQuantizationType.IQ4_NL,
        QuantizationFamily.IQ,
        32,
        18,
        _fields(("delta_f16", 2), ("nonlinear_indexes", 16)),
    ),
    GGMLQuantizationType.IQ3_S: CodecLayout(
        GGMLQuantizationType.IQ3_S,
        QuantizationFamily.IQ,
        256,
        110,
        _fields(
            ("delta_f16", 2),
            ("grid_indexes", 64),
            ("high_bits", 8),
            ("signs", 32),
            ("scales_u4", 4),
        ),
    ),
    GGMLQuantizationType.IQ2_S: CodecLayout(
        GGMLQuantizationType.IQ2_S,
        QuantizationFamily.IQ,
        256,
        82,
        _fields(("delta_f16", 2), ("grid_indexes", 64), ("high_bits", 8), ("scales", 8)),
    ),
    GGMLQuantizationType.IQ4_XS: CodecLayout(
        GGMLQuantizationType.IQ4_XS,
        QuantizationFamily.IQ,
        256,
        136,
        _fields(
            ("delta_f16", 2),
            ("scales_high", 2),
            ("scales_low", 4),
            ("nonlinear_indexes", 128),
        ),
    ),
    GGMLQuantizationType.IQ1_M: CodecLayout(
        GGMLQuantizationType.IQ1_M,
        QuantizationFamily.IQ,
        256,
        56,
        _fields(("grid_indexes", 32), ("high_bits", 16), ("scales", 8)),
    ),
}

# These pinned layouts are sufficient for GGUF indexing and byte-preserving copy.
# They are deliberately excluded from QUANT_LAYOUTS because ModelSurgeon does not
# claim native decode/encode conformance for the legacy formats.
LEGACY_STORAGE_LAYOUTS: dict[GGMLQuantizationType, CodecLayout] = {
    GGMLQuantizationType.Q4_0: CodecLayout(
        GGMLQuantizationType.Q4_0,
        QuantizationFamily.LEGACY,
        32,
        18,
        _fields(("delta_f16", 2), ("quants_u4", 16)),
    ),
    GGMLQuantizationType.Q4_1: CodecLayout(
        GGMLQuantizationType.Q4_1,
        QuantizationFamily.LEGACY,
        32,
        20,
        _fields(("delta_min_f16x2", 4), ("quants_u4", 16)),
    ),
    GGMLQuantizationType.Q5_0: CodecLayout(
        GGMLQuantizationType.Q5_0,
        QuantizationFamily.LEGACY,
        32,
        22,
        _fields(("delta_f16", 2), ("high_bits", 4), ("quants_u4", 16)),
    ),
    GGMLQuantizationType.Q5_1: CodecLayout(
        GGMLQuantizationType.Q5_1,
        QuantizationFamily.LEGACY,
        32,
        24,
        _fields(("delta_min_f16x2", 4), ("high_bits", 4), ("quants_u4", 16)),
    ),
}

GGUF_STORAGE_LAYOUTS = {**QUANT_LAYOUTS, **LEGACY_STORAGE_LAYOUTS}


@dataclass(frozen=True, slots=True)
class AxisEditConstraint:
    axis: int
    mode: AxisEditMode
    index_granularity: int
    shape: tuple[int, ...]
    row_bytes: int
    tensor_bytes: int

    def __post_init__(self) -> None:
        if (
            self.axis < 0
            or self.index_granularity <= 0
            or self.row_bytes <= 0
            or self.tensor_bytes <= 0
        ):
            raise ValueError("axis, edit granularity, and encoded sizes must be valid")


@dataclass(frozen=True, slots=True)
class AlignmentConstraint:
    """Validated GGUF tensor offset under one container alignment."""

    container_alignment: int
    tensor_offset: int


def validate_tensor_alignment(
    tensor_offset: int,
    container_alignment: int,
) -> AlignmentConstraint:
    """Require the bounded power-of-two alignment selected by the container."""
    if (
        container_alignment < 8
        or container_alignment > 1 << 20
        or container_alignment & (container_alignment - 1)
    ):
        raise CodecContractError(
            "container alignment must be a power of two from 8 through 1048576"
        )
    if tensor_offset < 0 or tensor_offset % container_alignment != 0:
        raise CodecContractError(
            f"tensor offset {tensor_offset} is not aligned to {container_alignment} bytes"
        )
    return AlignmentConstraint(container_alignment, tensor_offset)


def _plan_axis_edit(
    quant_type: GGMLQuantizationType,
    shape: tuple[int, ...],
    axis: int,
    layouts: Mapping[GGMLQuantizationType, CodecLayout],
) -> AxisEditConstraint:
    layout = layouts[quant_type]
    if not shape or any(dimension <= 0 for dimension in shape):
        raise CodecContractError("GGUF tensor dimensions must be positive")
    if axis < 0 or axis >= len(shape):
        raise CodecContractError(f"edit axis {axis} is outside rank {len(shape)}")
    if shape[0] % layout.block_size != 0:
        raise CodecContractError(
            f"{quant_type.value} contiguous dimension {shape[0]} must be divisible by "
            f"block size {layout.block_size}"
        )
    row_bytes = layout.encoded_size(shape[0])
    row_count = reduce(mul, shape[1:], 1)
    tensor_bytes = row_bytes * row_count
    if tensor_bytes > (1 << 63) - 1:
        raise CodecContractError("encoded tensor size exceeds the supported signed 64-bit range")
    return AxisEditConstraint(
        axis=axis,
        mode=AxisEditMode.BLOCK_VALUES if axis == 0 else AxisEditMode.WHOLE_SLICES,
        index_granularity=layout.block_size if axis == 0 else 1,
        shape=shape,
        row_bytes=row_bytes,
        tensor_bytes=tensor_bytes,
    )


def plan_axis_edit(
    quant_type: GGMLQuantizationType,
    shape: tuple[int, ...],
    axis: int,
) -> AxisEditConstraint:
    """Validate a native-codec shape and report exact edit granularity."""
    return _plan_axis_edit(quant_type, shape, axis, QUANT_LAYOUTS)


def plan_storage_axis_edit(
    quant_type: GGMLQuantizationType,
    shape: tuple[int, ...],
    axis: int,
) -> AxisEditConstraint:
    """Plan byte-copy geometry, including layouts without a native float codec."""
    return _plan_axis_edit(quant_type, shape, axis, GGUF_STORAGE_LAYOUTS)


def plan_supported_axes(
    quant_type: GGMLQuantizationType,
    shape: tuple[int, ...],
) -> tuple[AxisEditConstraint, ...]:
    """Report every supported storage axis and its exact mutation granularity."""
    if not shape:
        raise CodecContractError("GGUF tensor dimensions must be positive")
    return tuple(plan_axis_edit(quant_type, shape, axis) for axis in range(len(shape)))


@dataclass(frozen=True, slots=True)
class CodecIdentity:
    quant_type: GGMLQuantizationType
    implementation: str
    version: str
    upstream_revision: str

    def __post_init__(self) -> None:
        if not self.implementation or not self.version or not self.upstream_revision:
            raise ValueError("codec implementation, version, and upstream revision are required")


@dataclass(frozen=True, slots=True)
class BlockOperation:
    block_count: int
    element_count: int
    byte_count: int

    def __post_init__(self) -> None:
        if self.block_count < 0 or self.element_count < 0 or self.byte_count < 0:
            raise ValueError("block operation counts must be non-negative")


@dataclass(frozen=True, slots=True)
class BlockValidation:
    valid: bool
    block_count: int
    reason: str

    def __post_init__(self) -> None:
        if self.block_count < 0 or not self.reason:
            raise ValueError("block validation requires a non-negative count and reason")

    def require_valid(self) -> None:
        if not self.valid:
            raise CodecContractError(self.reason)


@dataclass(frozen=True, slots=True)
class QuantizationError:
    sample_count: int
    mean_absolute_error: float
    mean_squared_error: float
    max_absolute_error: float
    reference_l2: float

    @classmethod
    def measure(
        cls,
        reference: Sequence[float],
        candidate: Sequence[float],
    ) -> QuantizationError:
        if len(reference) != len(candidate) or not reference:
            raise CodecContractError("error estimation requires equal non-empty samples")
        values = chain(reference, candidate)
        if not all(math.isfinite(value) for value in values):
            raise CodecContractError("error estimation requires finite values")
        count = len(reference)
        indices = range(count)
        return cls(
            sample_count=count,
            mean_absolute_error=math.fsum(
                abs(float(reference[index]) - float(candidate[index])) for index in indices
            )
            / count,
            mean_squared_error=math.fsum(
                (float(reference[index]) - float(candidate[index])) ** 2
                for index in indices
            )
            / count,
            max_absolute_error=max(
                abs(float(reference[index]) - float(candidate[index])) for index in indices
            ),
            reference_l2=math.sqrt(math.fsum(float(value) ** 2 for value in reference)),
        )


@runtime_checkable
class QuantizationCodec(Protocol):
    """Bounded block codec; implementations cannot accept a family fallback."""

    @property
    def identity(self) -> CodecIdentity: ...

    @property
    def layout(self) -> CodecLayout: ...

    def validate_blocks(self, source: memoryview, *, byte_order: ByteOrder) -> BlockValidation: ...

    def decode_blocks(
        self,
        source: memoryview,
        destination: MutableSequence[float],
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation: ...

    def encode_blocks(
        self,
        source: Sequence[float],
        destination: memoryview,
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation: ...

    def estimate_error(
        self,
        reference: Sequence[float],
        candidate: Sequence[float],
    ) -> QuantizationError: ...


class CodecRegistry:
    """Exact-type registry that deliberately has no family-level fallback."""

    def __init__(self) -> None:
        self._codecs: dict[GGMLQuantizationType, QuantizationCodec] = {}

    def register(self, codec: QuantizationCodec) -> None:
        quant_type = codec.identity.quant_type
        if codec.layout.quant_type is not quant_type:
            raise CodecContractError("codec identity and layout quantization types disagree")
        if codec.layout != QUANT_LAYOUTS[quant_type]:
            raise CodecContractError(
                f"codec layout does not match pinned {quant_type.value} layout"
            )
        if quant_type in self._codecs:
            raise CodecContractError(f"codec already registered for {quant_type.value}")
        self._codecs[quant_type] = codec

    def resolve(self, quant_type: GGMLQuantizationType) -> QuantizationCodec:
        try:
            return self._codecs[quant_type]
        except KeyError as exc:
            raise UnsupportedCodecError(
                f"no exact codec registered for {quant_type.value}; "
                "family substitution is forbidden"
            ) from exc

    def supported_types(self) -> tuple[GGMLQuantizationType, ...]:
        return tuple(sorted(self._codecs, key=lambda item: item.value))

"""Distinct Q5_K tensor codec and whole-file recipe metadata handling."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.adapters.gguf.conformance import GGML_UPSTREAM_REVISION
from modelsurgeon.adapters.gguf.quantization import (
    QUANT_LAYOUTS,
    BlockOperation,
    BlockValidation,
    ByteOrder,
    CodecContractError,
    CodecIdentity,
    CodecLayout,
    GGMLQuantizationType,
    QuantizationError,
)


class Q5KRecipe(StrEnum):
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"


@dataclass(frozen=True, slots=True)
class Q5KRecipeMetadata:
    recipe: Q5KRecipe
    general_file_type: int
    tensor_type: GGMLQuantizationType = GGMLQuantizationType.Q5_K


_FILE_TYPES = {16: Q5KRecipe.Q5_K_S, 17: Q5KRecipe.Q5_K_M}


def resolve_q5_k_recipe(metadata: Mapping[str, object]) -> Q5KRecipeMetadata:
    """Resolve a whole-file recipe without treating it as a tensor codec variant."""

    raw = metadata.get("general.file_type")
    if not isinstance(raw, int) or isinstance(raw, bool) or raw not in _FILE_TYPES:
        raise CodecContractError("general.file_type is not a supported Q5_K_S/Q5_K_M recipe")
    return Q5KRecipeMetadata(_FILE_TYPES[raw], raw)


def _nearest(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _scale_min(index: int, packed: memoryview) -> tuple[int, int]:
    if index < 4:
        return packed[index] & 63, packed[index + 4] & 63
    scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4)
    minimum = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4)
    return scale, minimum


def _pack_scales(scales: Sequence[int], minima: Sequence[int]) -> bytes:
    packed = bytearray(12)
    for index, (scale, minimum) in enumerate(zip(scales, minima, strict=True)):
        if index < 4:
            packed[index] = scale
            packed[index + 4] = minimum
        else:
            packed[index + 4] = (scale & 15) | ((minimum & 15) << 4)
            packed[index - 4] |= (scale >> 4) << 6
            packed[index] |= (minimum >> 4) << 6
    return bytes(packed)


class Q5_KCodec:
    identity = CodecIdentity(
        GGMLQuantizationType.Q5_K,
        "modelsurgeon.struct",
        "1",
        GGML_UPSTREAM_REVISION,
    )
    layout: CodecLayout = QUANT_LAYOUTS[GGMLQuantizationType.Q5_K]

    def validate_blocks(self, source: memoryview, *, byte_order: ByteOrder) -> BlockValidation:
        del byte_order
        valid = source.ndim == 1 and source.contiguous and source.nbytes % 176 == 0
        return BlockValidation(
            valid,
            source.nbytes // 176 if valid else 0,
            "valid Q5_K super-blocks" if valid else "partial or non-contiguous Q5_K block",
        )

    def encoded_block_range(
        self, source: memoryview, block_offset: int, block_count: int
    ) -> memoryview:
        validation = self.validate_blocks(source, byte_order=ByteOrder.LITTLE)
        validation.require_valid()
        if (
            block_offset < 0
            or block_count < 0
            or block_offset > validation.block_count - block_count
        ):
            raise CodecContractError("Q5_K super-block range escapes encoded source")
        return source.cast("B")[block_offset * 176 : (block_offset + block_count) * 176]

    def _decode_block(self, block: memoryview, byte_order: ByteOrder) -> tuple[float, ...]:
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        delta, minimum_delta = struct.unpack(prefix + "ee", block[:4])
        if not math.isfinite(delta) or not math.isfinite(minimum_delta):
            raise CodecContractError("Q5_K block scales must be finite")
        packed, high, low = block[4:16], block[16:48], block[48:176]
        output: list[float] = []
        high_one, high_two, scale_index = 1, 2, 0
        for segment in range(4):
            scale1, min1 = _scale_min(scale_index, packed)
            scale2, min2 = _scale_min(scale_index + 1, packed)
            base = segment * 32
            output.extend(
                delta * scale1 * ((low[base + lane] & 15) + (16 if high[lane] & high_one else 0))
                - minimum_delta * min1
                for lane in range(32)
            )
            output.extend(
                delta * scale2 * ((low[base + lane] >> 4) + (16 if high[lane] & high_two else 0))
                - minimum_delta * min2
                for lane in range(32)
            )
            high_one <<= 2
            high_two <<= 2
            scale_index += 2
        return tuple(output)

    def decode_blocks(
        self,
        source: memoryview,
        destination: MutableSequence[float],
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation:
        validation = self.validate_blocks(source, byte_order=byte_order)
        validation.require_valid()
        view = source.cast("B")
        for offset in range(0, len(view), 176):
            destination.extend(self._decode_block(view[offset : offset + 176], byte_order))
        return BlockOperation(validation.block_count, validation.block_count * 256, source.nbytes)

    def _encode_block(self, values: Sequence[float], byte_order: ByteOrder) -> bytes:
        local_scales: list[float] = []
        local_minima: list[float] = []
        for start in range(0, 256, 32):
            group = values[start : start + 32]
            minimum = max(0.0, -min(group))
            local_minima.append(minimum)
            local_scales.append((max(group) + minimum) / 31.0)
        max_scale, max_minimum = max(local_scales), max(local_minima)
        delta, min_delta = max_scale / 63.0, max_minimum / 63.0
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        encoded_deltas = struct.pack(prefix + "ee", delta, min_delta)
        delta, min_delta = struct.unpack(prefix + "ee", encoded_deltas)
        scales = [min(63, _nearest(value / delta)) if delta else 0 for value in local_scales]
        minima = [
            min(63, _nearest(value / min_delta)) if min_delta else 0
            for value in local_minima
        ]
        levels: list[int] = []
        for group_index in range(8):
            step = delta * scales[group_index]
            offset = min_delta * minima[group_index]
            levels.extend(
                max(0, min(31, _nearest((value + offset) / step))) if step else 0
                for value in values[group_index * 32 : (group_index + 1) * 32]
            )
        high, low = bytearray(32), bytearray(128)
        high_one, high_two = 1, 2
        for segment in range(4):
            first = levels[segment * 64 : segment * 64 + 32]
            second = levels[segment * 64 + 32 : segment * 64 + 64]
            for lane, (one, two) in enumerate(zip(first, second, strict=True)):
                low[segment * 32 + lane] = (one & 15) | ((two & 15) << 4)
                if one > 15:
                    high[lane] |= high_one
                if two > 15:
                    high[lane] |= high_two
            high_one <<= 2
            high_two <<= 2
        return encoded_deltas + _pack_scales(scales, minima) + bytes(high + low)

    def encode_blocks(
        self,
        source: Sequence[float],
        destination: memoryview,
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation:
        expected = self.layout.encoded_size(len(source))
        if destination.readonly or not destination.contiguous or destination.nbytes != expected:
            raise CodecContractError("Q5_K destination must be writable and exact-sized")
        if not all(math.isfinite(value) for value in source):
            raise CodecContractError("Q5_K input values must be finite")
        view = destination.cast("B")
        for index, start in enumerate(range(0, len(source), 256)):
            view[index * 176 : (index + 1) * 176] = self._encode_block(
                source[start : start + 256], byte_order
            )
        return BlockOperation(len(source) // 256, len(source), expected)

    def estimate_error(
        self, reference: Sequence[float], candidate: Sequence[float]
    ) -> QuantizationError:
        return QuantizationError.measure(reference, candidate)


Q5_K_CODEC = Q5_KCodec()

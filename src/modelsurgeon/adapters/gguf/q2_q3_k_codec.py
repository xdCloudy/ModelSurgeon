"""Separate Q2_K and Q3_K super-block codecs."""

from __future__ import annotations

import math
import struct
from collections.abc import MutableSequence, Sequence

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


def _nearest(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


class _KCodecBase:
    identity: CodecIdentity
    layout: CodecLayout

    def validate_blocks(self, source: memoryview, *, byte_order: ByteOrder) -> BlockValidation:
        del byte_order
        valid = (
            source.ndim == 1
            and source.contiguous
            and source.nbytes % self.layout.type_size == 0
        )
        return BlockValidation(
            valid,
            source.nbytes // self.layout.type_size if valid else 0,
            f"valid {self.identity.quant_type.value} super-blocks"
            if valid
            else f"partial or non-contiguous {self.identity.quant_type.value} block",
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
            raise CodecContractError(
                f"{self.identity.quant_type.value} super-block range escapes encoded source"
            )
        size = self.layout.type_size
        return source.cast("B")[block_offset * size : (block_offset + block_count) * size]

    def estimate_error(
        self, reference: Sequence[float], candidate: Sequence[float]
    ) -> QuantizationError:
        return QuantizationError.measure(reference, candidate)

    def _destination(self, source: Sequence[float], destination: memoryview) -> memoryview:
        expected = self.layout.encoded_size(len(source))
        if destination.readonly or not destination.contiguous or destination.nbytes != expected:
            raise CodecContractError(
                f"{self.identity.quant_type.value} destination must be writable and exact-sized"
            )
        if not all(math.isfinite(value) for value in source):
            raise CodecContractError(
                f"{self.identity.quant_type.value} input values must be finite"
            )
        return destination.cast("B")


class Q2_KCodec(_KCodecBase):
    identity = CodecIdentity(
        GGMLQuantizationType.Q2_K,
        "modelsurgeon.struct",
        "1",
        GGML_UPSTREAM_REVISION,
    )
    layout = QUANT_LAYOUTS[GGMLQuantizationType.Q2_K]

    def _decode_block(self, block: memoryview, byte_order: ByteOrder) -> tuple[float, ...]:
        scales, quants = block[:16], block[16:80]
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        delta, min_delta = struct.unpack(prefix + "ee", block[80:84])
        if not math.isfinite(delta) or not math.isfinite(min_delta):
            raise CodecContractError("Q2_K block scales must be finite")
        output: list[float] = []
        scale_index = 0
        for half in range(2):
            base = half * 32
            for shift in (0, 2, 4, 6):
                for lane_half in (0, 16):
                    packed = scales[scale_index]
                    scale_index += 1
                    step, offset = delta * (packed & 15), min_delta * (packed >> 4)
                    output.extend(
                        step * ((quants[base + lane_half + lane] >> shift) & 3) - offset
                        for lane in range(16)
                    )
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
        for offset in range(0, len(view), 84):
            destination.extend(self._decode_block(view[offset : offset + 84], byte_order))
        return BlockOperation(validation.block_count, validation.block_count * 256, source.nbytes)

    def _encode_block(self, values: Sequence[float], byte_order: ByteOrder) -> bytes:
        local_scales, local_minima = [], []
        for start in range(0, 256, 16):
            group_values = values[start : start + 16]
            minimum = max(0.0, -min(group_values))
            local_minima.append(minimum)
            local_scales.append((max(group_values) + minimum) / 3.0)
        delta = max(local_scales) / 15.0
        min_delta = max(local_minima) / 15.0
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        encoded_delta = struct.pack(prefix + "ee", delta, min_delta)
        delta, min_delta = struct.unpack(prefix + "ee", encoded_delta)
        scale_codes = [min(15, _nearest(item / delta)) if delta else 0 for item in local_scales]
        min_codes = [
            min(15, _nearest(item / min_delta)) if min_delta else 0 for item in local_minima
        ]
        levels: list[int] = []
        for group_index in range(16):
            step = delta * scale_codes[group_index]
            offset = min_delta * min_codes[group_index]
            levels.extend(
                max(0, min(3, _nearest((value + offset) / step))) if step else 0
                for value in values[group_index * 16 : (group_index + 1) * 16]
            )
        quants = bytearray(64)
        for half in range(2):
            for shift_index in range(4):
                for lane_half in range(2):
                    group_index = half * 8 + shift_index * 2 + lane_half
                    for lane in range(16):
                        quants[half * 32 + lane_half * 16 + lane] |= (
                            levels[group_index * 16 + lane] << (2 * shift_index)
                        )
        packed_scales = bytes(
            scale | (minimum << 4)
            for scale, minimum in zip(scale_codes, min_codes, strict=True)
        )
        return packed_scales + bytes(quants) + encoded_delta

    def encode_blocks(
        self,
        source: Sequence[float],
        destination: memoryview,
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation:
        view = self._destination(source, destination)
        for index, start in enumerate(range(0, len(source), 256)):
            view[index * 84 : (index + 1) * 84] = self._encode_block(
                source[start : start + 256], byte_order
            )
        return BlockOperation(len(source) // 256, len(source), len(view))


def _unpack_q3_scales(packed: memoryview) -> tuple[int, ...]:
    low = tuple((packed[index % 8] >> (4 if index >= 8 else 0)) & 15 for index in range(16))
    high = tuple(
        (packed[8 + (index % 4)] >> (2 * (index // 4))) & 3 for index in range(16)
    )
    return tuple((low[index] | (high[index] << 4)) - 32 for index in range(16))


def _pack_q3_scales(scales: Sequence[int]) -> bytes:
    codes = [value + 32 for value in scales]
    packed = bytearray(12)
    for index in range(8):
        packed[index] = (codes[index] & 15) | ((codes[index + 8] & 15) << 4)
    for index, code in enumerate(codes):
        packed[8 + index % 4] |= ((code >> 4) & 3) << (2 * (index // 4))
    return bytes(packed)


class Q3_KCodec(_KCodecBase):
    identity = CodecIdentity(
        GGMLQuantizationType.Q3_K,
        "modelsurgeon.struct",
        "1",
        GGML_UPSTREAM_REVISION,
    )
    layout = QUANT_LAYOUTS[GGMLQuantizationType.Q3_K]

    def _decode_block(self, block: memoryview, byte_order: ByteOrder) -> tuple[float, ...]:
        high_mask, quants, scales = block[:32], block[32:96], _unpack_q3_scales(block[96:108])
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        delta = struct.unpack(prefix + "e", block[108:110])[0]
        if not math.isfinite(delta):
            raise CodecContractError("Q3_K block scale must be finite")
        output: list[float] = []
        group = 0
        mask = 1
        for half in range(2):
            base = half * 32
            for shift in (0, 2, 4, 6):
                for lane_half in (0, 16):
                    output.extend(
                        delta
                        * scales[group]
                        * (
                            ((quants[base + lane_half + lane] >> shift) & 3)
                            - (0 if high_mask[lane_half + lane] & mask else 4)
                        )
                        for lane in range(16)
                    )
                    group += 1
                mask <<= 1
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
        for offset in range(0, len(view), 110):
            destination.extend(self._decode_block(view[offset : offset + 110], byte_order))
        return BlockOperation(validation.block_count, validation.block_count * 256, source.nbytes)

    def _encode_block(self, values: Sequence[float], byte_order: ByteOrder) -> bytes:
        if max(abs(value) for value in values) < 1e-15:
            return bytes(self.layout.type_size)
        local_scales: list[float] = []
        for start in range(0, 256, 16):
            group_values = values[start : start + 16]
            maximum = max(group_values, key=lambda value: abs(value))
            local_scales.append(maximum / (3.0 if maximum >= 0 else -4.0))
        signed_max = max(local_scales, key=lambda value: abs(value))
        inverse = -32.0 / signed_max if signed_max else 0.0
        delta = 1.0 / inverse if inverse else 0.0
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        encoded_delta = struct.pack(prefix + "e", delta)
        delta = struct.unpack(prefix + "e", encoded_delta)[0]
        scales = [max(-32, min(31, _nearest(inverse * item))) for item in local_scales]
        levels: list[int] = []
        for group_index in range(16):
            step = delta * scales[group_index]
            levels.extend(
                max(-4, min(3, _nearest(value / step))) if step else 0
                for value in values[group_index * 16 : (group_index + 1) * 16]
            )
        high_mask, quants = bytearray(32), bytearray(64)
        mask = 1
        for half in range(2):
            for shift_index in range(4):
                for lane_half in range(2):
                    group_index = half * 8 + shift_index * 2 + lane_half
                    for lane in range(16):
                        level = levels[group_index * 16 + lane]
                        quants[half * 32 + lane_half * 16 + lane] |= (
                            (level & 3) << (2 * shift_index)
                        )
                        if level >= 0:
                            high_mask[lane_half * 16 + lane] |= mask
                mask <<= 1
        return bytes(high_mask + quants) + _pack_q3_scales(scales) + encoded_delta

    def encode_blocks(
        self,
        source: Sequence[float],
        destination: memoryview,
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation:
        view = self._destination(source, destination)
        for index, start in enumerate(range(0, len(source), 256)):
            view[index * 110 : (index + 1) * 110] = self._encode_block(
                source[start : start + 256], byte_order
            )
        return BlockOperation(len(source) // 256, len(source), len(view))


Q2_K_CODEC = Q2_KCodec()
Q3_K_CODEC = Q3_KCodec()

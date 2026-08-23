"""Exact-layout, super-block-bounded Q6_K GGUF codec."""

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

_BLOCK_VALUES = 256


def _nearest(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _make_qx(values: Sequence[float]) -> float:
    maximum = max(values, key=lambda value: abs(value))
    if abs(maximum) < 1e-15:
        return 0.0

    def candidate(scale_shift: float) -> tuple[float, float, tuple[int, ...]]:
        inverse = -(32.0 + scale_shift) / maximum
        levels = tuple(max(-32, min(31, _nearest(inverse * value))) for value in values)
        sum_lx = math.fsum(value**3 * level for value, level in zip(values, levels, strict=True))
        sum_l2 = math.fsum(value**2 * level**2 for value, level in zip(values, levels, strict=True))
        scale = sum_lx / sum_l2 if sum_l2 else 0.0
        return scale, scale * sum_lx, levels

    best_scale, best_score, _ = candidate(0.0)
    for step in range(-9, 10):
        if step:
            scale, score, _ = candidate(0.1 * step)
            if score > best_score:
                best_scale, best_score = scale, score
    return best_scale


class Q6_KCodec:
    """Encode and decode the distinct 256-value, 210-byte Q6_K layout."""

    identity = CodecIdentity(
        GGMLQuantizationType.Q6_K,
        "modelsurgeon.struct",
        "1",
        GGML_UPSTREAM_REVISION,
    )
    layout: CodecLayout = QUANT_LAYOUTS[GGMLQuantizationType.Q6_K]

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
            "valid Q6_K super-blocks" if valid else "partial or non-contiguous Q6_K block",
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
            raise CodecContractError("Q6_K super-block range escapes encoded source")
        start = block_offset * self.layout.type_size
        return source.cast("B")[start : start + block_count * self.layout.type_size]

    def _decode_block(self, block: memoryview, byte_order: ByteOrder) -> tuple[float, ...]:
        ql = block[:128]
        qh = block[128:192]
        scales = struct.unpack("16b", block[192:208])
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        delta = float(struct.unpack(prefix + "e", block[208:210])[0])
        if not math.isfinite(delta):
            raise CodecContractError("Q6_K super-block scale must be finite")
        output = [0.0] * _BLOCK_VALUES
        for half in range(2):
            ql_base, qh_base, value_base, scale_base = half * 64, half * 32, half * 128, half * 8
            for lane in range(32):
                low_a, low_b = ql[ql_base + lane], ql[ql_base + 32 + lane]
                high = qh[qh_base + lane]
                values = (
                    (low_a & 15) | ((high & 3) << 4),
                    (low_b & 15) | (((high >> 2) & 3) << 4),
                    (low_a >> 4) | (((high >> 4) & 3) << 4),
                    (low_b >> 4) | (((high >> 6) & 3) << 4),
                )
                pair = lane // 16
                for quarter, quant in enumerate(values):
                    scale = scales[scale_base + pair + 2 * quarter]
                    output[value_base + lane + 32 * quarter] = delta * scale * (quant - 32)
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
        for offset in range(0, len(view), self.layout.type_size):
            destination.extend(
                self._decode_block(view[offset : offset + self.layout.type_size], byte_order)
            )
        return BlockOperation(
            validation.block_count,
            validation.block_count * _BLOCK_VALUES,
            source.nbytes,
        )

    def _encode_block(self, values: Sequence[float], byte_order: ByteOrder) -> bytes:
        subgroup_scales = [_make_qx(values[index : index + 16]) for index in range(0, 256, 16)]
        maximum = max(subgroup_scales, key=lambda value: abs(value))
        if abs(maximum) < 1e-15:
            return bytes(self.layout.type_size)
        inverse = -128.0 / maximum
        delta = 1.0 / inverse
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        try:
            encoded_delta = struct.pack(prefix + "e", delta)
        except (OverflowError, struct.error) as error:
            raise CodecContractError("Q6_K scale is outside float16 representation") from error
        rounded_delta = float(struct.unpack(prefix + "e", encoded_delta)[0])
        scales = [min(127, _nearest(inverse * scale)) for scale in subgroup_scales]
        levels = [32] * 256
        for group in range(16):
            local_delta = rounded_delta * scales[group]
            if local_delta:
                for index in range(16):
                    quant = max(-32, min(31, _nearest(values[group * 16 + index] / local_delta)))
                    levels[group * 16 + index] = quant + 32
        ql, qh = bytearray(128), bytearray(64)
        for half in range(2):
            base = half * 128
            for lane in range(32):
                a, b, c, d = (levels[base + lane + 32 * part] for part in range(4))
                ql[half * 64 + lane] = (a & 15) | ((c & 15) << 4)
                ql[half * 64 + 32 + lane] = (b & 15) | ((d & 15) << 4)
                qh[half * 32 + lane] = (
                    (a >> 4) | ((b >> 4) << 2) | ((c >> 4) << 4) | ((d >> 4) << 6)
                )
        return bytes(ql + qh) + struct.pack("16b", *scales) + encoded_delta

    def encode_blocks(
        self,
        source: Sequence[float],
        destination: memoryview,
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation:
        expected = self.layout.encoded_size(len(source))
        if (
            destination.readonly
            or destination.ndim != 1
            or not destination.contiguous
            or destination.nbytes != expected
        ):
            raise CodecContractError("Q6_K destination must be writable and exact-sized")
        if not all(math.isfinite(value) for value in source):
            raise CodecContractError("Q6_K input values must be finite")
        view = destination.cast("B")
        for block_index, start in enumerate(range(0, len(source), _BLOCK_VALUES)):
            encoded = self._encode_block(source[start : start + _BLOCK_VALUES], byte_order)
            offset = block_index * self.layout.type_size
            view[offset : offset + self.layout.type_size] = encoded
        return BlockOperation(len(source) // _BLOCK_VALUES, len(source), expected)

    def estimate_error(
        self, reference: Sequence[float], candidate: Sequence[float]
    ) -> QuantizationError:
        return QuantizationError.measure(reference, candidate)


Q6_K_CODEC = Q6_KCodec()

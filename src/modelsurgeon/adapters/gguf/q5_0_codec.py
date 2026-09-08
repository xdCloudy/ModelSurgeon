"""Bit-exact, block-bounded Q5_0 GGUF codec."""

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


class Q5_0Codec:
    """Encode or decode exact 32-value symmetric Q5_0 blocks."""

    identity = CodecIdentity(
        GGMLQuantizationType.Q5_0,
        "modelsurgeon.struct",
        "1",
        GGML_UPSTREAM_REVISION,
    )
    layout: CodecLayout = QUANT_LAYOUTS[GGMLQuantizationType.Q5_0]

    def validate_blocks(self, source: memoryview, *, byte_order: ByteOrder) -> BlockValidation:
        del byte_order
        valid = source.ndim == 1 and source.contiguous and source.nbytes % 22 == 0
        return BlockValidation(
            valid,
            source.nbytes // 22 if valid else 0,
            "valid Q5_0 blocks" if valid else "partial or non-contiguous Q5_0 block",
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
            raise CodecContractError("Q5_0 block range escapes encoded source")
        start = block_offset * 22
        return source.cast("B")[start : start + block_count * 22]

    def decode_blocks(
        self,
        source: memoryview,
        destination: MutableSequence[float],
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation:
        validation = self.validate_blocks(source, byte_order=byte_order)
        validation.require_valid()
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        view = source.cast("B")
        for offset in range(0, len(view), 22):
            delta = float(struct.unpack(prefix + "e", view[offset : offset + 2])[0])
            if not math.isfinite(delta):
                raise CodecContractError("Q5_0 block scale must be finite")
            high_bits = int.from_bytes(view[offset + 2 : offset + 6], byte_order.value)
            quants = view[offset + 6 : offset + 22]
            for lane in range(16):
                low = quants[lane] & 0x0F
                high = (high_bits >> lane) & 1
                destination.append(delta * ((low | (high << 4)) - 16))
            for lane in range(16):
                low = quants[lane] >> 4
                high = (high_bits >> (lane + 16)) & 1
                destination.append(delta * ((low | (high << 4)) - 16))
        return BlockOperation(validation.block_count, validation.block_count * 32, source.nbytes)

    @staticmethod
    def _roundf(value: float) -> int:
        return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)

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
            raise CodecContractError("Q5_0 destination must be writable and exact-sized")
        if not all(math.isfinite(value) for value in source):
            raise CodecContractError("Q5_0 input values must be finite")
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        view = destination.cast("B")
        for index, start in enumerate(range(0, len(source), 32)):
            block = [float(value) for value in source[start : start + 32]]
            delta = max(abs(value) for value in block) / 16.0
            inverse = 0.0 if delta == 0.0 else 1.0 / delta
            quantized = tuple(max(-16, min(15, self._roundf(value * inverse))) for value in block)
            low_bytes = bytes(
                (quantized[lane] + 16) & 0x0F
                | (((quantized[lane + 16] + 16) & 0x0F) << 4)
                for lane in range(16)
            )
            high_bits = 0
            for lane, value in enumerate(quantized):
                if value >= 0:
                    high_bits |= 1 << lane
            encoded = struct.pack(prefix + "eI", delta, high_bits) + low_bytes
            view[index * 22 : (index + 1) * 22] = encoded
        return BlockOperation(len(source) // 32, len(source), expected)

    def estimate_error(
        self, reference: Sequence[float], candidate: Sequence[float]
    ) -> QuantizationError:
        return QuantizationError.measure(reference, candidate)


Q5_0_CODEC = Q5_0Codec()

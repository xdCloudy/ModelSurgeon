"""Bit-exact, block-bounded Q8_0 GGUF codec."""

from __future__ import annotations

import math
import struct
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class Q8QuantizationReport:
    operation: BlockOperation
    error: QuantizationError


class Q8_0Codec:
    """Encode or decode exact 32-value Q8_0 blocks without family fallback."""

    identity = CodecIdentity(
        GGMLQuantizationType.Q8_0,
        "modelsurgeon.struct",
        "1",
        GGML_UPSTREAM_REVISION,
    )
    layout: CodecLayout = QUANT_LAYOUTS[GGMLQuantizationType.Q8_0]

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
            "valid Q8_0 blocks" if valid else "partial or non-contiguous Q8_0 block",
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
            raise CodecContractError("Q8_0 block range escapes encoded source")
        start = block_offset * self.layout.type_size
        return source.cast("B")[start : start + block_count * self.layout.type_size]

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
        for offset in range(0, len(view), self.layout.type_size):
            delta = float(struct.unpack(prefix + "e", view[offset : offset + 2])[0])
            if not math.isfinite(delta):
                raise CodecContractError("Q8_0 block scale must be finite")
            quants = struct.unpack("32b", view[offset + 2 : offset + 34])
            destination.extend(delta * value for value in quants)
        return BlockOperation(
            validation.block_count,
            validation.block_count * self.layout.block_size,
            source.nbytes,
        )

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
            raise CodecContractError("Q8_0 destination must be writable and exact-sized")
        if not all(math.isfinite(value) for value in source):
            raise CodecContractError("Q8_0 input values must be finite")
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        view = destination.cast("B")
        output_offset = 0
        for start in range(0, len(source), self.layout.block_size):
            block = source[start : start + self.layout.block_size]
            delta = max(abs(float(value)) for value in block) / 127.0
            inverse = 0.0 if delta == 0 else 1.0 / delta
            quants = tuple(self._roundf(float(value) * inverse) for value in block)
            try:
                encoded_delta = struct.pack(prefix + "e", delta)
                encoded_quants = struct.pack("32b", *quants)
            except (OverflowError, struct.error) as error:
                raise CodecContractError("Q8_0 scale or quant is outside representation") from error
            view[output_offset : output_offset + 2] = encoded_delta
            view[output_offset + 2 : output_offset + 34] = encoded_quants
            output_offset += self.layout.type_size
        return BlockOperation(len(source) // 32, len(source), expected)

    def estimate_error(
        self, reference: Sequence[float], candidate: Sequence[float]
    ) -> QuantizationError:
        return QuantizationError.measure(reference, candidate)

    def encode_with_error(
        self,
        source: Sequence[float],
        destination: memoryview,
        *,
        byte_order: ByteOrder,
    ) -> Q8QuantizationReport:
        operation = self.encode_blocks(source, destination, byte_order=byte_order)
        decoded: list[float] = []
        self.decode_blocks(destination, decoded, byte_order=byte_order)
        return Q8QuantizationReport(operation, self.estimate_error(source, decoded))


Q8_0_CODEC = Q8_0Codec()

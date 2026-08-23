"""Exact bounded codecs for F32, F16, and BF16 GGUF tensors."""

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
    QuantizationFamily,
)


class DenseGGUFCodec:
    """Encode and decode one exact unquantized GGUF scalar type."""

    def __init__(self, quant_type: GGMLQuantizationType) -> None:
        layout = QUANT_LAYOUTS[quant_type]
        if layout.family is not QuantizationFamily.DENSE:
            raise CodecContractError(f"{quant_type.value} is not an unquantized scalar type")
        self._layout = layout
        self._identity = CodecIdentity(
            quant_type, "modelsurgeon.struct", "1", GGML_UPSTREAM_REVISION
        )

    @property
    def identity(self) -> CodecIdentity:
        return self._identity

    @property
    def layout(self) -> CodecLayout:
        return self._layout

    def validate_blocks(self, source: memoryview, *, byte_order: ByteOrder) -> BlockValidation:
        del byte_order
        valid = source.ndim == 1 and source.nbytes % self.layout.type_size == 0
        return BlockValidation(
            valid,
            source.nbytes // self.layout.type_size if valid else 0,
            "valid dense scalar blocks" if valid else "partial or non-contiguous dense block",
        )

    def _decode_one(self, data: bytes, byte_order: ByteOrder) -> float:
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        if self.identity.quant_type is GGMLQuantizationType.F32:
            return float(struct.unpack(prefix + "f", data)[0])
        if self.identity.quant_type is GGMLQuantizationType.F16:
            return float(struct.unpack(prefix + "e", data)[0])
        bits = int.from_bytes(data, byte_order.value) << 16
        return float(struct.unpack(">f", bits.to_bytes(4, "big"))[0])

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
        size = self.layout.type_size
        for offset in range(0, len(view), size):
            destination.append(self._decode_one(bytes(view[offset : offset + size]), byte_order))
        return BlockOperation(validation.block_count, validation.block_count, source.nbytes)

    def _encode_one(self, value: float, byte_order: ByteOrder) -> bytes:
        if not math.isfinite(value):
            raise CodecContractError("dense codec input values must be finite")
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        try:
            if self.identity.quant_type is GGMLQuantizationType.F32:
                return struct.pack(prefix + "f", value)
            if self.identity.quant_type is GGMLQuantizationType.F16:
                return struct.pack(prefix + "e", value)
            bits = int.from_bytes(struct.pack(">f", value), "big")
        except (OverflowError, struct.error) as error:
            raise CodecContractError("value is outside dense scalar representation") from error
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        return (rounded & 0xFFFF).to_bytes(2, byte_order.value)

    def encode_blocks(
        self,
        source: Sequence[float],
        destination: memoryview,
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation:
        expected = self.layout.encoded_size(len(source))
        if destination.readonly or destination.ndim != 1 or destination.nbytes != expected:
            raise CodecContractError("dense codec destination must be writable and exact-sized")
        view = destination.cast("B")
        offset = 0
        for raw in source:
            encoded = self._encode_one(float(raw), byte_order)
            view[offset : offset + len(encoded)] = encoded
            offset += len(encoded)
        return BlockOperation(len(source), len(source), expected)

    def estimate_error(
        self, reference: Sequence[float], candidate: Sequence[float]
    ) -> QuantizationError:
        return QuantizationError.measure(reference, candidate)


F32_CODEC = DenseGGUFCodec(GGMLQuantizationType.F32)
F16_CODEC = DenseGGUFCodec(GGMLQuantizationType.F16)
BF16_CODEC = DenseGGUFCodec(GGMLQuantizationType.BF16)
DENSE_GGUF_CODECS = (F32_CODEC, F16_CODEC, BF16_CODEC)

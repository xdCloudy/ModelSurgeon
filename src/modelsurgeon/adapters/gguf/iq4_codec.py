"""Prioritized IQ4_NL and IQ4_XS native GGUF codecs."""

from __future__ import annotations

import math
import struct
from collections.abc import MutableSequence, Sequence

from modelsurgeon.adapters.gguf.conformance import GGML_UPSTREAM_REVISION
from modelsurgeon.adapters.gguf.iq_support import require_iq_native_write_target
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

IQ4_NL_VALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)


def _nearest(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _indices(values: Sequence[float], delta: float) -> tuple[int, ...]:
    if delta == 0:
        return (0,) * len(values)
    return tuple(
        min(range(16), key=lambda index: abs(value - delta * IQ4_NL_VALUES[index]))
        for value in values
    )


def _fit_delta(values: Sequence[float]) -> tuple[float, tuple[int, ...]]:
    maximum = max(values, key=lambda value: abs(value))
    if abs(maximum) < 1e-15:
        return 0.0, (0,) * len(values)
    delta = maximum / (113 if maximum > 0 else -127)
    indexes = _indices(values, delta)
    for _ in range(2):
        quants = [IQ4_NL_VALUES[index] for index in indexes]
        denominator = math.fsum(quant * quant for quant in quants)
        delta = math.fsum(
            value * quant for value, quant in zip(values, quants, strict=True)
        ) / denominator
        indexes = _indices(values, delta)
    return delta, indexes


class _IQ4Base:
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
            f"valid {self.identity.quant_type.value} blocks"
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
                f"{self.identity.quant_type.value} block range escapes encoded source"
            )
        size = self.layout.type_size
        return source.cast("B")[block_offset * size : (block_offset + block_count) * size]

    def estimate_error(
        self, reference: Sequence[float], candidate: Sequence[float]
    ) -> QuantizationError:
        return QuantizationError.measure(reference, candidate)

    def _destination(self, source: Sequence[float], destination: memoryview) -> memoryview:
        expected = self.layout.encoded_size(len(source))
        if (
            destination.ndim != 1
            or destination.readonly
            or not destination.contiguous
            or destination.nbytes != expected
        ):
            raise CodecContractError(
                f"{self.identity.quant_type.value} destination must be writable and exact-sized"
            )
        if not all(math.isfinite(value) for value in source):
            raise CodecContractError(
                f"{self.identity.quant_type.value} input values must be finite"
            )
        return destination.cast("B")


class IQ4_NLCodec(_IQ4Base):
    identity = CodecIdentity(
        GGMLQuantizationType.IQ4_NL,
        "modelsurgeon.struct",
        "1",
        GGML_UPSTREAM_REVISION,
    )
    layout = QUANT_LAYOUTS[GGMLQuantizationType.IQ4_NL]

    @staticmethod
    def _decode_block(block: memoryview, byte_order: ByteOrder) -> tuple[float, ...]:
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        delta = struct.unpack(prefix + "e", block[:2])[0]
        if not math.isfinite(delta):
            raise CodecContractError("IQ4_NL block scale must be finite")
        return tuple(
            delta
            * IQ4_NL_VALUES[
                (block[2 + index % 16] >> (4 if index >= 16 else 0)) & 15
            ]
            for index in range(32)
        )

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
        for offset in range(0, len(view), 18):
            destination.extend(self._decode_block(view[offset : offset + 18], byte_order))
        return BlockOperation(validation.block_count, validation.block_count * 32, source.nbytes)

    @staticmethod
    def _encode_block(values: Sequence[float], byte_order: ByteOrder) -> bytes:
        if max(abs(value) for value in values) < 1e-15:
            return bytes(18)
        delta, indexes = _fit_delta(values)
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        try:
            encoded_delta = struct.pack(prefix + "e", delta)
        except (OverflowError, struct.error) as error:
            raise CodecContractError("IQ4_NL block scale exceeds binary16 range") from error
        stored_delta = struct.unpack(prefix + "e", encoded_delta)[0]
        indexes = _indices(values, stored_delta)
        packed = bytes(indexes[index] | (indexes[index + 16] << 4) for index in range(16))
        return encoded_delta + packed

    def encode_blocks(
        self,
        source: Sequence[float],
        destination: memoryview,
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation:
        view = self._destination(source, destination)
        for index, start in enumerate(range(0, len(source), 32)):
            view[index * 18 : (index + 1) * 18] = self._encode_block(
                source[start : start + 32], byte_order
            )
        return BlockOperation(len(source) // 32, len(source), len(view))


class IQ4_XSCodec(_IQ4Base):
    identity = CodecIdentity(
        GGMLQuantizationType.IQ4_XS,
        "modelsurgeon.struct",
        "1",
        GGML_UPSTREAM_REVISION,
    )
    layout = QUANT_LAYOUTS[GGMLQuantizationType.IQ4_XS]

    @staticmethod
    def _scales(block: memoryview, byte_order: ByteOrder) -> tuple[int, ...]:
        high = int.from_bytes(block[2:4], byte_order.value)
        low = block[4:8]
        return tuple(
            ((low[index // 2] >> (4 if index % 2 else 0)) & 15)
            | (((high >> (2 * index)) & 3) << 4)
            for index in range(8)
        )

    def _decode_block(self, block: memoryview, byte_order: ByteOrder) -> tuple[float, ...]:
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        delta = struct.unpack(prefix + "e", block[:2])[0]
        if not math.isfinite(delta):
            raise CodecContractError("IQ4_XS super-block scale must be finite")
        scales = tuple(code - 32 for code in self._scales(block, byte_order))
        quants = block[8:136]
        output: list[float] = []
        for group in range(8):
            local_delta = delta * scales[group]
            base = group * 16
            output.extend(
                local_delta * IQ4_NL_VALUES[quants[base + index] & 15] for index in range(16)
            )
            output.extend(
                local_delta * IQ4_NL_VALUES[quants[base + index] >> 4] for index in range(16)
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
        for offset in range(0, len(view), 136):
            destination.extend(self._decode_block(view[offset : offset + 136], byte_order))
        return BlockOperation(validation.block_count, validation.block_count * 256, source.nbytes)

    @staticmethod
    def _encode_block(values: Sequence[float], byte_order: ByteOrder) -> bytes:
        if max(abs(value) for value in values) < 1e-15:
            return bytes(136)
        local = [_fit_delta(values[start : start + 32])[0] for start in range(0, 256, 32)]
        maximum = max(local, key=lambda value: abs(value))
        inverse = -32.0 / maximum
        global_delta = 1.0 / inverse
        prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
        try:
            encoded_delta = struct.pack(prefix + "e", global_delta)
        except (OverflowError, struct.error) as error:
            raise CodecContractError("IQ4_XS super-block scale exceeds binary16 range") from error
        global_delta = struct.unpack(prefix + "e", encoded_delta)[0]
        scales = [max(-32, min(31, _nearest(inverse * item))) for item in local]
        codes = [scale + 32 for scale in scales]
        high = sum(((code >> 4) & 3) << (2 * index) for index, code in enumerate(codes))
        low = bytes(codes[index] & 15 | ((codes[index + 1] & 15) << 4) for index in range(0, 8, 2))
        quants = bytearray(128)
        for group in range(8):
            indexes = _indices(
                values[group * 32 : (group + 1) * 32], global_delta * scales[group]
            )
            for index in range(16):
                quants[group * 16 + index] = indexes[index] | (indexes[index + 16] << 4)
        return encoded_delta + high.to_bytes(2, byte_order.value) + low + bytes(quants)

    def encode_blocks(
        self,
        source: Sequence[float],
        destination: memoryview,
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation:
        view = self._destination(source, destination)
        for index, start in enumerate(range(0, len(source), 256)):
            view[index * 136 : (index + 1) * 136] = self._encode_block(
                source[start : start + 256], byte_order
            )
        return BlockOperation(len(source) // 256, len(source), len(view))


IQ4_NL_CODEC = IQ4_NLCodec()
IQ4_XS_CODEC = IQ4_XSCodec()
IQ4_CODECS = {
    GGMLQuantizationType.IQ4_NL: IQ4_NL_CODEC,
    GGMLQuantizationType.IQ4_XS: IQ4_XS_CODEC,
}


def resolve_iq_native_write_codec(quant_type: GGMLQuantizationType) -> _IQ4Base:
    require_iq_native_write_target(quant_type)
    try:
        return IQ4_CODECS[quant_type]
    except KeyError as error:
        raise CodecContractError(f"no native IQ write codec for {quant_type.value}") from error

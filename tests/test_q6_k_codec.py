"""Tests for the distinct Q6_K super-block codec."""

from __future__ import annotations

import pytest

from modelsurgeon.adapters.gguf import (
    GGUF_CODEC_CONFORMANCE_VECTORS,
    Q6_K_CODEC,
    CodecContractError,
    validate_codec_vector,
)
from modelsurgeon.adapters.gguf.quantization import ByteOrder, GGMLQuantizationType


def _values() -> list[float]:
    return [((index % 37) - 18) * (1.0 + index // 64) / 9 for index in range(256)]


def test_zero_conformance_vector_decodes_and_reencodes_exactly() -> None:
    vector = next(
        item
        for item in GGUF_CODEC_CONFORMANCE_VECTORS
        if item.quant_type is GGMLQuantizationType.Q6_K
    )
    validate_codec_vector(vector)
    decoded: list[float] = []
    Q6_K_CODEC.decode_blocks(memoryview(vector.encoded), decoded, byte_order=ByteOrder.LITTLE)
    encoded = bytearray(210)
    Q6_K_CODEC.encode_blocks(decoded, memoryview(encoded), byte_order=ByteOrder.LITTLE)
    assert decoded == [0.0] * 256
    assert bytes(encoded) == vector.encoded


@pytest.mark.parametrize("byte_order", [ByteOrder.LITTLE, ByteOrder.BIG])
def test_nonzero_super_block_round_trip_and_endian_scale(byte_order: ByteOrder) -> None:
    values = _values()
    encoded = bytearray(210)
    Q6_K_CODEC.encode_blocks(values, memoryview(encoded), byte_order=byte_order)
    decoded: list[float] = []
    Q6_K_CODEC.decode_blocks(memoryview(encoded), decoded, byte_order=byte_order)
    error = Q6_K_CODEC.estimate_error(values, decoded)
    assert error.max_absolute_error < 0.5
    assert len(set(encoded[192:208])) > 1


def test_layout_range_and_validation_are_q6_k_specific() -> None:
    assert Q6_K_CODEC.layout.quant_type is GGMLQuantizationType.Q6_K
    assert Q6_K_CODEC.layout.type_size == 210
    assert tuple(field.name for field in Q6_K_CODEC.layout.fields) == (
        "low_bits", "high_bits", "scales_i8", "delta_f16"
    )
    encoded = memoryview(bytes(420))
    assert len(Q6_K_CODEC.encoded_block_range(encoded, 1, 1)) == 210
    with pytest.raises(CodecContractError, match="escapes"):
        Q6_K_CODEC.encoded_block_range(encoded, 2, 1)
    with pytest.raises(CodecContractError, match="partial"):
        validation = Q6_K_CODEC.validate_blocks(
            memoryview(bytes(209)), byte_order=ByteOrder.LITTLE
        )
        validation.require_valid()

"""Tests for F32, F16, and BF16 GGUF codecs."""

from __future__ import annotations

import struct

import pytest

from modelsurgeon.adapters.gguf import (
    BF16_CODEC,
    F16_CODEC,
    F32_CODEC,
    CodecContractError,
)
from modelsurgeon.adapters.gguf.quantization import ByteOrder


@pytest.mark.parametrize("codec", [F32_CODEC, F16_CODEC, BF16_CODEC])
@pytest.mark.parametrize("byte_order", [ByteOrder.LITTLE, ByteOrder.BIG])
def test_odd_value_counts_round_trip_in_both_byte_orders(
    codec: object, byte_order: ByteOrder
) -> None:
    values = [-3.25, -0.0, 1.0, 2.5, 17.0]
    size = codec.layout.encoded_size(len(values))  # type: ignore[attr-defined]
    encoded = bytearray(size)

    operation = codec.encode_blocks(values, memoryview(encoded), byte_order=byte_order)  # type: ignore[attr-defined]
    decoded: list[float] = []
    codec.decode_blocks(memoryview(encoded), decoded, byte_order=byte_order)  # type: ignore[attr-defined]

    assert operation.element_count == 5
    tolerance = 0 if codec is F32_CODEC else 0.02
    assert decoded == pytest.approx(values, abs=tolerance)


def test_exact_scalar_bytes_match_endian_and_conformance_values() -> None:
    cases = (
        (F32_CODEC, "0000803f", "3f800000"),
        (F16_CODEC, "003c", "3c00"),
        (BF16_CODEC, "803f", "3f80"),
    )
    for codec, little, big in cases:
        outputs = []
        for order in (ByteOrder.LITTLE, ByteOrder.BIG):
            encoded = bytearray(codec.layout.type_size)
            codec.encode_blocks([1.0], memoryview(encoded), byte_order=order)
            outputs.append(encoded.hex())
        assert outputs == [little, big]


def test_bf16_rounds_to_nearest_even() -> None:
    value = struct.unpack(">f", bytes.fromhex("3f808000"))[0]
    encoded = bytearray(2)
    BF16_CODEC.encode_blocks([value], memoryview(encoded), byte_order=ByteOrder.BIG)
    assert encoded.hex() == "3f80"


def test_partial_read_wrong_destination_and_nonfinite_input_fail() -> None:
    with pytest.raises(CodecContractError, match="partial"):
        F32_CODEC.validate_blocks(memoryview(bytes(3)), byte_order=ByteOrder.LITTLE).require_valid()
    with pytest.raises(CodecContractError, match="exact-sized"):
        F16_CODEC.encode_blocks([1.0], memoryview(bytearray(3)), byte_order=ByteOrder.LITTLE)
    with pytest.raises(CodecContractError, match="finite"):
        F32_CODEC.encode_blocks(
            [float("nan")], memoryview(bytearray(4)), byte_order=ByteOrder.LITTLE
        )

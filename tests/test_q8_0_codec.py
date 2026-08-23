"""Tests for exact block-aware Q8_0 encoding and decoding."""

from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.adapters.gguf import (
    GGUF_CODEC_CONFORMANCE_VECTORS,
    Q8_0_CODEC,
    CodecContractError,
    validate_codec_vector,
)
from modelsurgeon.adapters.gguf.quantization import ByteOrder


def _source() -> list[float]:
    return [-1.0 + 2.0 * index / 31 for index in range(32)]


def test_encode_is_bit_exact_with_pinned_upstream_vector() -> None:
    vector = next(
        item for item in GGUF_CODEC_CONFORMANCE_VECTORS if item.quant_type.value == "Q8_0"
    )
    encoded = bytearray(34)

    report = Q8_0_CODEC.encode_with_error(
        _source(), memoryview(encoded), byte_order=ByteOrder.LITTLE
    )

    assert bytes(encoded) == vector.encoded
    assert report.operation.block_count == 1
    assert report.error.max_absolute_error < 0.004
    validate_codec_vector(replace(vector, encoded=bytes(encoded)))


@pytest.mark.parametrize("byte_order", [ByteOrder.LITTLE, ByteOrder.BIG])
def test_multiple_blocks_round_trip_and_report_error(byte_order: ByteOrder) -> None:
    values = _source() + [value * 17 for value in reversed(_source())]
    encoded = bytearray(68)

    report = Q8_0_CODEC.encode_with_error(
        values, memoryview(encoded), byte_order=byte_order
    )
    decoded: list[float] = []
    Q8_0_CODEC.decode_blocks(memoryview(encoded), decoded, byte_order=byte_order)

    assert report.operation.element_count == 64
    assert decoded == pytest.approx(values, abs=0.07)
    assert report.error.mean_absolute_error > 0


def test_block_range_returns_only_requested_complete_blocks() -> None:
    encoded = bytearray(68)
    Q8_0_CODEC.encode_blocks(
        _source() + [0.0] * 32, memoryview(encoded), byte_order=ByteOrder.LITTLE
    )

    second = Q8_0_CODEC.encoded_block_range(memoryview(encoded), 1, 1)
    decoded: list[float] = []
    Q8_0_CODEC.decode_blocks(second, decoded, byte_order=ByteOrder.LITTLE)

    assert len(second) == 34
    assert decoded == [0.0] * 32


def test_partial_ranges_shapes_and_nonfinite_scales_fail_before_decode() -> None:
    with pytest.raises(CodecContractError, match="partial"):
        validation = Q8_0_CODEC.validate_blocks(
            memoryview(bytes(33)), byte_order=ByteOrder.LITTLE
        )
        validation.require_valid()
    with pytest.raises(CodecContractError, match="escapes"):
        Q8_0_CODEC.encoded_block_range(memoryview(bytes(34)), 1, 1)
    with pytest.raises(CodecContractError, match="multiple of 32"):
        Q8_0_CODEC.encode_blocks(
            [0.0] * 31, memoryview(bytearray(34)), byte_order=ByteOrder.LITTLE
        )
    corrupt = memoryview(bytearray.fromhex("007c" + "00" * 32))
    with pytest.raises(CodecContractError, match="finite"):
        Q8_0_CODEC.decode_blocks(corrupt, [], byte_order=ByteOrder.LITTLE)

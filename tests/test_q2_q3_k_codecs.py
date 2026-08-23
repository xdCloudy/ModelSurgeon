"""Tests for separate Q2_K and Q3_K super-block codecs."""

from __future__ import annotations

import pytest

from modelsurgeon.adapters.gguf import (
    GGUF_CODEC_CONFORMANCE_VECTORS,
    Q2_K_CODEC,
    Q3_K_CODEC,
    CodecContractError,
)
from modelsurgeon.adapters.gguf.quantization import ByteOrder, GGMLQuantizationType


def _values() -> list[float]:
    return [((index % 29) - 14) * (1 + index // 80) / 8 for index in range(256)]


@pytest.mark.parametrize(
    ("codec", "quant_type", "size"),
    [
        (Q2_K_CODEC, GGMLQuantizationType.Q2_K, 84),
        (Q3_K_CODEC, GGMLQuantizationType.Q3_K, 110),
    ],
)
def test_each_zero_conformance_vector_round_trips_in_its_own_layout(
    codec: object, quant_type: GGMLQuantizationType, size: int
) -> None:
    vector = next(item for item in GGUF_CODEC_CONFORMANCE_VECTORS if item.quant_type is quant_type)
    decoded: list[float] = []
    codec.decode_blocks(memoryview(vector.encoded), decoded, byte_order=ByteOrder.LITTLE)  # type: ignore[attr-defined]
    encoded = bytearray(size)
    codec.encode_blocks(decoded, memoryview(encoded), byte_order=ByteOrder.LITTLE)  # type: ignore[attr-defined]
    assert bytes(encoded) == vector.encoded


@pytest.mark.parametrize("codec", [Q2_K_CODEC, Q3_K_CODEC])
@pytest.mark.parametrize("byte_order", [ByteOrder.LITTLE, ByteOrder.BIG])
def test_nonzero_blocks_round_trip_with_type_appropriate_error(
    codec: object, byte_order: ByteOrder
) -> None:
    values = _values()
    encoded = bytearray(codec.layout.type_size)  # type: ignore[attr-defined]
    codec.encode_blocks(values, memoryview(encoded), byte_order=byte_order)  # type: ignore[attr-defined]
    decoded: list[float] = []
    codec.decode_blocks(memoryview(encoded), decoded, byte_order=byte_order)  # type: ignore[attr-defined]
    error = codec.estimate_error(values, decoded)  # type: ignore[attr-defined]
    assert error.max_absolute_error < 2.0


def test_codecs_cannot_accept_each_others_super_blocks() -> None:
    assert Q2_K_CODEC.layout != Q3_K_CODEC.layout
    with pytest.raises(CodecContractError, match="partial"):
        q2_validation = Q2_K_CODEC.validate_blocks(
            memoryview(bytes(110)), byte_order=ByteOrder.LITTLE
        )
        q2_validation.require_valid()
    with pytest.raises(CodecContractError, match="partial"):
        q3_validation = Q3_K_CODEC.validate_blocks(
            memoryview(bytes(84)), byte_order=ByteOrder.LITTLE
        )
        q3_validation.require_valid()


def test_range_access_is_exact_type_scoped() -> None:
    assert len(Q2_K_CODEC.encoded_block_range(memoryview(bytes(168)), 1, 1)) == 84
    assert len(Q3_K_CODEC.encoded_block_range(memoryview(bytes(220)), 1, 1)) == 110

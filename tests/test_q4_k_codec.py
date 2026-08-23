"""Tests for Q4_K physical blocks and Q4_K_S/Q4_K_M recipe metadata."""

from __future__ import annotations

import pytest

from modelsurgeon.adapters.gguf import (
    GGUF_CODEC_CONFORMANCE_VECTORS,
    Q4_K_CODEC,
    CodecContractError,
    Q4KRecipe,
    resolve_q4_k_recipe,
    validate_codec_vector,
)
from modelsurgeon.adapters.gguf.quantization import ByteOrder, GGMLQuantizationType


def _values() -> list[float]:
    return [((index % 43) - 21) * (1 + index // 96) / 7 for index in range(256)]


def test_zero_conformance_vector_round_trips_exactly() -> None:
    vector = next(
        item
        for item in GGUF_CODEC_CONFORMANCE_VECTORS
        if item.quant_type is GGMLQuantizationType.Q4_K
    )
    validate_codec_vector(vector)
    decoded: list[float] = []
    Q4_K_CODEC.decode_blocks(memoryview(vector.encoded), decoded, byte_order=ByteOrder.LITTLE)
    encoded = bytearray(144)
    Q4_K_CODEC.encode_blocks(decoded, memoryview(encoded), byte_order=ByteOrder.LITTLE)
    assert bytes(encoded) == vector.encoded


@pytest.mark.parametrize("byte_order", [ByteOrder.LITTLE, ByteOrder.BIG])
def test_nonzero_q4_k_round_trip_and_packed_scales(byte_order: ByteOrder) -> None:
    values = _values()
    encoded = bytearray(144)
    Q4_K_CODEC.encode_blocks(values, memoryview(encoded), byte_order=byte_order)
    decoded: list[float] = []
    Q4_K_CODEC.decode_blocks(memoryview(encoded), decoded, byte_order=byte_order)
    assert Q4_K_CODEC.estimate_error(values, decoded).max_absolute_error < 1.0
    assert any(encoded[4:16])


def test_recipe_metadata_never_changes_physical_tensor_type() -> None:
    small = resolve_q4_k_recipe({"general.file_type": 14})
    medium = resolve_q4_k_recipe({"general.file_type": 15})
    assert small.recipe is Q4KRecipe.Q4_K_S
    assert medium.recipe is Q4KRecipe.Q4_K_M
    assert small.tensor_type is medium.tensor_type is GGMLQuantizationType.Q4_K
    assert Q4_K_CODEC.layout.type_size == 144
    with pytest.raises(CodecContractError, match="not a supported"):
        resolve_q4_k_recipe({"general.file_type": 17})


def test_range_and_partial_validation_are_super_block_scoped() -> None:
    assert len(Q4_K_CODEC.encoded_block_range(memoryview(bytes(288)), 1, 1)) == 144
    with pytest.raises(CodecContractError, match="escapes"):
        Q4_K_CODEC.encoded_block_range(memoryview(bytes(144)), 1, 1)
    with pytest.raises(CodecContractError, match="partial"):
        validation = Q4_K_CODEC.validate_blocks(
            memoryview(bytes(143)), byte_order=ByteOrder.LITTLE
        )
        validation.require_valid()

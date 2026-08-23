"""Tests for Q5_K physical blocks and Q5_K_S/Q5_K_M recipe metadata."""

from __future__ import annotations

import pytest

from modelsurgeon.adapters.gguf import (
    GGUF_CODEC_CONFORMANCE_VECTORS,
    Q5_K_CODEC,
    CodecContractError,
    Q5KRecipe,
    resolve_q5_k_recipe,
    validate_codec_vector,
)
from modelsurgeon.adapters.gguf.quantization import ByteOrder, GGMLQuantizationType


def _values() -> list[float]:
    return [((index % 43) - 21) * (1 + index // 96) / 7 for index in range(256)]


def test_zero_conformance_vector_round_trips_exactly() -> None:
    vector = next(
        item
        for item in GGUF_CODEC_CONFORMANCE_VECTORS
        if item.quant_type is GGMLQuantizationType.Q5_K
    )
    validate_codec_vector(vector)
    decoded: list[float] = []
    Q5_K_CODEC.decode_blocks(memoryview(vector.encoded), decoded, byte_order=ByteOrder.LITTLE)
    encoded = bytearray(176)
    Q5_K_CODEC.encode_blocks(decoded, memoryview(encoded), byte_order=ByteOrder.LITTLE)
    assert bytes(encoded) == vector.encoded


@pytest.mark.parametrize("byte_order", [ByteOrder.LITTLE, ByteOrder.BIG])
def test_nonzero_q5_k_round_trip_and_packed_scales(byte_order: ByteOrder) -> None:
    values = _values()
    encoded = bytearray(176)
    Q5_K_CODEC.encode_blocks(values, memoryview(encoded), byte_order=byte_order)
    decoded: list[float] = []
    Q5_K_CODEC.decode_blocks(memoryview(encoded), decoded, byte_order=byte_order)
    assert Q5_K_CODEC.estimate_error(values, decoded).max_absolute_error < 0.7
    assert any(encoded[4:16])
    assert any(encoded[16:48])


def test_recipe_metadata_never_changes_physical_tensor_type() -> None:
    small = resolve_q5_k_recipe({"general.file_type": 16})
    medium = resolve_q5_k_recipe({"general.file_type": 17})
    assert small.recipe is Q5KRecipe.Q5_K_S
    assert medium.recipe is Q5KRecipe.Q5_K_M
    assert small.tensor_type is medium.tensor_type is GGMLQuantizationType.Q5_K
    assert Q5_K_CODEC.layout.type_size == 176
    with pytest.raises(CodecContractError, match="not a supported"):
        resolve_q5_k_recipe({"general.file_type": 12})


def test_q5_k_range_and_partial_validation_are_super_block_scoped() -> None:
    assert len(Q5_K_CODEC.encoded_block_range(memoryview(bytes(352)), 1, 1)) == 176
    with pytest.raises(CodecContractError, match="escapes"):
        Q5_K_CODEC.encoded_block_range(memoryview(bytes(176)), 1, 1)
    with pytest.raises(CodecContractError, match="partial"):
        validation = Q5_K_CODEC.validate_blocks(
            memoryview(bytes(175)), byte_order=ByteOrder.LITTLE
        )
        validation.require_valid()

"""Tests for the prioritized native IQ4_NL and IQ4_XS codecs."""

from __future__ import annotations

import pytest

from modelsurgeon.adapters.gguf import (
    GGUF_CODEC_CONFORMANCE_VECTORS,
    IQ4_NL_CODEC,
    IQ4_XS_CODEC,
    CodecContractError,
    resolve_iq_native_write_codec,
    validate_codec_vector,
)
from modelsurgeon.adapters.gguf.quantization import ByteOrder, GGMLQuantizationType


@pytest.mark.parametrize(
    ("quant_type", "codec", "encoded_size"),
    [
        (GGMLQuantizationType.IQ4_NL, IQ4_NL_CODEC, 18),
        (GGMLQuantizationType.IQ4_XS, IQ4_XS_CODEC, 136),
    ],
)
def test_zero_conformance_vectors_round_trip_exactly(
    quant_type: GGMLQuantizationType, codec: object, encoded_size: int
) -> None:
    vector = next(
        item for item in GGUF_CODEC_CONFORMANCE_VECTORS if item.quant_type is quant_type
    )
    validate_codec_vector(vector)
    decoded: list[float] = []
    codec.decode_blocks(memoryview(vector.encoded), decoded, byte_order=ByteOrder.LITTLE)
    encoded = bytearray(encoded_size)
    codec.encode_blocks(decoded, memoryview(encoded), byte_order=ByteOrder.LITTLE)
    assert not any(decoded)
    assert bytes(encoded) == vector.encoded


@pytest.mark.parametrize("byte_order", [ByteOrder.LITTLE, ByteOrder.BIG])
def test_nonzero_iq4_nl_round_trip(byte_order: ByteOrder) -> None:
    values = [((index % 17) - 8) * (1 + index // 16) / 3 for index in range(32)]
    encoded = bytearray(18)
    IQ4_NL_CODEC.encode_blocks(values, memoryview(encoded), byte_order=byte_order)
    decoded: list[float] = []
    IQ4_NL_CODEC.decode_blocks(memoryview(encoded), decoded, byte_order=byte_order)
    assert IQ4_NL_CODEC.estimate_error(values, decoded).max_absolute_error < 1.0
    assert any(encoded)


@pytest.mark.parametrize("byte_order", [ByteOrder.LITTLE, ByteOrder.BIG])
def test_nonzero_iq4_xs_round_trip_and_packed_scales(byte_order: ByteOrder) -> None:
    values = [((index % 37) - 18) * (1 + index // 64) / 5 for index in range(256)]
    encoded = bytearray(136)
    IQ4_XS_CODEC.encode_blocks(values, memoryview(encoded), byte_order=byte_order)
    decoded: list[float] = []
    IQ4_XS_CODEC.decode_blocks(memoryview(encoded), decoded, byte_order=byte_order)
    assert IQ4_XS_CODEC.estimate_error(values, decoded).max_absolute_error < 1.5
    assert any(encoded[2:8])


def test_exact_type_dispatch_never_substitutes_iq_families() -> None:
    assert resolve_iq_native_write_codec(GGMLQuantizationType.IQ4_NL) is IQ4_NL_CODEC
    assert resolve_iq_native_write_codec(GGMLQuantizationType.IQ4_XS) is IQ4_XS_CODEC
    with pytest.raises(CodecContractError, match="read_only"):
        resolve_iq_native_write_codec(GGMLQuantizationType.IQ2_XS)
    with pytest.raises(CodecContractError, match="deferred"):
        resolve_iq_native_write_codec(GGMLQuantizationType.IQ1_M)


def test_layout_validation_and_ranges_are_exact_type_scoped() -> None:
    assert IQ4_NL_CODEC.layout.type_size == 18
    assert IQ4_XS_CODEC.layout.type_size == 136
    assert len(IQ4_NL_CODEC.encoded_block_range(memoryview(bytes(36)), 1, 1)) == 18
    with pytest.raises(CodecContractError, match="partial"):
        IQ4_NL_CODEC.validate_blocks(
            memoryview(bytes(136)), byte_order=ByteOrder.LITTLE
        ).require_valid()
    with pytest.raises(CodecContractError, match="partial"):
        IQ4_XS_CODEC.validate_blocks(
            memoryview(bytes(18)), byte_order=ByteOrder.LITTLE
        ).require_valid()
    with pytest.raises(CodecContractError, match="escapes"):
        IQ4_XS_CODEC.encoded_block_range(memoryview(bytes(136)), 1, 1)


def test_encode_rejects_partial_and_non_finite_input_before_mutation() -> None:
    destination = bytearray(b"x" * 18)
    with pytest.raises(CodecContractError, match="multiple of 32"):
        IQ4_NL_CODEC.encode_blocks(
            [0.0] * 31, memoryview(destination), byte_order=ByteOrder.LITTLE
        )
    assert destination == bytearray(b"x" * 18)
    with pytest.raises(CodecContractError, match="finite"):
        IQ4_NL_CODEC.encode_blocks(
            [0.0] * 31 + [float("nan")],
            memoryview(destination),
            byte_order=ByteOrder.LITTLE,
        )
    assert destination == bytearray(b"x" * 18)

"""Tests for pinned GGUF codec conformance vector coverage and packing."""

from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.adapters.gguf import (
    GGML_UPSTREAM_REVISION,
    GGUF_CODEC_CONFORMANCE_VECTORS,
    CodecVectorError,
    validate_codec_vector,
)
from modelsurgeon.adapters.gguf.quantization import QUANT_LAYOUTS, ByteOrder


def test_every_supported_exact_type_has_a_pinned_one_block_vector() -> None:
    by_type = {vector.quant_type: vector for vector in GGUF_CODEC_CONFORMANCE_VECTORS}
    assert set(by_type) == set(QUANT_LAYOUTS)
    for quant_type, layout in QUANT_LAYOUTS.items():
        vector = by_type[quant_type]
        validate_codec_vector(vector)
        assert vector.shape == (layout.block_size,)
        assert len(vector.encoded) == layout.type_size
        assert len(vector.decoded) == layout.block_size
        assert vector.upstream_revision == GGML_UPSTREAM_REVISION
        assert vector.source_license == "MIT"
        assert len(vector.encoded_sha256) == 64


def test_q8_vector_is_nonzero_and_exercises_scale_signs_and_extrema() -> None:
    vector = next(
        item for item in GGUF_CODEC_CONFORMANCE_VECTORS if item.quant_type.value == "Q8_0"
    )
    assert any(vector.encoded)
    assert min(vector.decoded) < 0 < max(vector.decoded)
    assert vector.decoded[0] == pytest.approx(-0.999938965)
    assert vector.decoded[-1] == pytest.approx(0.999938965)


def test_validator_detects_byte_order_and_field_packing_errors() -> None:
    dense = next(
        item for item in GGUF_CODEC_CONFORMANCE_VECTORS if item.quant_type.value == "F32"
    )
    with pytest.raises(CodecVectorError, match="byte order or packing"):
        validate_codec_vector(replace(dense, encoded=dense.encoded[::-1]))
    with pytest.raises(CodecVectorError, match="little-endian"):
        validate_codec_vector(replace(dense, byte_order=ByteOrder.BIG))

    q4 = next(
        item for item in GGUF_CODEC_CONFORMANCE_VECTORS if item.quant_type.value == "Q4_K"
    )
    corrupted = ((q4.field_spans[0][0], 1, q4.field_spans[0][2]), *q4.field_spans[1:])
    with pytest.raises(CodecVectorError, match="field packing"):
        validate_codec_vector(replace(q4, field_spans=corrupted))

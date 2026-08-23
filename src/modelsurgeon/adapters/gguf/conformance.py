"""Pinned license-compatible GGUF codec conformance vectors."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

from modelsurgeon.adapters.gguf.quantization import (
    QUANT_LAYOUTS,
    ByteOrder,
    GGMLQuantizationType,
)

GGML_UPSTREAM_REVISION = "95b8e33e16bb9a60de780a70930ebf729db6a90a"
GGUF_PY_QUANTS_BLOB = "80966b6ef1518a45b86745d94eb70d05c3c5490f"
GGUF_VECTOR_SOURCE = (
    "https://github.com/ggml-org/llama.cpp/blob/"
    f"{GGML_UPSTREAM_REVISION}/gguf-py/gguf/quants.py"
)


class CodecVectorError(ValueError):
    """Raised when a vector disagrees with its pinned codec layout or values."""


@dataclass(frozen=True, slots=True)
class CodecConformanceVector:
    quant_type: GGMLQuantizationType
    shape: tuple[int, ...]
    encoded: bytes
    decoded: tuple[float, ...]
    field_spans: tuple[tuple[str, int, int], ...]
    upstream_revision: str = GGML_UPSTREAM_REVISION
    source_url: str = GGUF_VECTOR_SOURCE
    source_license: str = "MIT"
    byte_order: ByteOrder = ByteOrder.LITTLE

    @property
    def encoded_sha256(self) -> str:
        return hashlib.sha256(self.encoded).hexdigest()


_Q8_ENCODED = bytes.fromhex(
    "08208189919aa2aab2bac3cbd3dbe3ecf4fc040c141d252d353d464e565e666f777f"
)
_Q8_DECODED = (
    -0.999938965, -0.936950684, -0.873962402, -0.803100586,
    -0.740112305, -0.677124023, -0.614135742, -0.551147461,
    -0.480285645, -0.417297363, -0.354309082, -0.291320801,
    -0.22833252, -0.157470703, -0.0944824219, -0.0314941406,
    0.0314941406, 0.0944824219, 0.157470703, 0.22833252,
    0.291320801, 0.354309082, 0.417297363, 0.480285645,
    0.551147461, 0.614135742, 0.677124023, 0.740112305,
    0.803100586, 0.873962402, 0.936950684, 0.999938965,
)


def _spans(quant_type: GGMLQuantizationType) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (field.name, field.offset, field.size) for field in QUANT_LAYOUTS[quant_type].fields
    )


def _vector(quant_type: GGMLQuantizationType) -> CodecConformanceVector:
    layout = QUANT_LAYOUTS[quant_type]
    encoded = bytes(layout.type_size)
    decoded = (0.0,) * layout.block_size
    if quant_type is GGMLQuantizationType.F32:
        encoded, decoded = bytes.fromhex("0000803f"), (1.0,)
    elif quant_type is GGMLQuantizationType.F16:
        encoded, decoded = bytes.fromhex("003c"), (1.0,)
    elif quant_type is GGMLQuantizationType.BF16:
        encoded, decoded = bytes.fromhex("803f"), (1.0,)
    elif quant_type is GGMLQuantizationType.Q8_0:
        encoded, decoded = _Q8_ENCODED, _Q8_DECODED
    return CodecConformanceVector(
        quant_type, (layout.block_size,), encoded, decoded, _spans(quant_type)
    )


GGUF_CODEC_CONFORMANCE_VECTORS = tuple(_vector(item) for item in QUANT_LAYOUTS)


def _decoded_dense(vector: CodecConformanceVector) -> tuple[float, ...] | None:
    if vector.quant_type is GGMLQuantizationType.F32:
        return (struct.unpack("<f", vector.encoded)[0],)
    if vector.quant_type is GGMLQuantizationType.F16:
        return (struct.unpack("<e", vector.encoded)[0],)
    if vector.quant_type is GGMLQuantizationType.BF16:
        bits = int.from_bytes(vector.encoded, "little") << 16
        return (struct.unpack("<f", bits.to_bytes(4, "little"))[0],)
    return None


def _decoded_q8(vector: CodecConformanceVector) -> tuple[float, ...]:
    delta = struct.unpack("<e", vector.encoded[:2])[0]
    quants = struct.unpack("<32b", vector.encoded[2:])
    return tuple(delta * item for item in quants)


def validate_codec_vector(vector: CodecConformanceVector) -> None:
    """Detect type size, block shape, byte order, and field-packing disagreement."""

    layout = QUANT_LAYOUTS[vector.quant_type]
    if vector.byte_order is not ByteOrder.LITTLE:
        raise CodecVectorError("pinned codec vectors must use little-endian block fields")
    if vector.shape != (layout.block_size,) or len(vector.decoded) != layout.block_size:
        raise CodecVectorError("decoded vector shape does not match one complete codec block")
    if len(vector.encoded) != layout.type_size:
        raise CodecVectorError("encoded vector size does not match codec type size")
    if vector.field_spans != _spans(vector.quant_type):
        raise CodecVectorError("encoded vector field packing does not match pinned layout")
    expected = _decoded_dense(vector)
    if vector.quant_type is GGMLQuantizationType.Q8_0:
        expected = _decoded_q8(vector)
    if expected is None:
        if any(vector.encoded) or any(vector.decoded):
            raise CodecVectorError("zero-block vector contains non-zero encoded or decoded data")
        return
    if any(
        not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-8)
        for actual, wanted in zip(expected, vector.decoded, strict=True)
    ):
        raise CodecVectorError("decoded values disagree with encoded byte order or packing")

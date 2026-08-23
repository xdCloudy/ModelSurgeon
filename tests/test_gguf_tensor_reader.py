"""Tests for lazy, tensor-scoped GGUF payload reads."""

from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from modelsurgeon.adapters.gguf import (
    GGUFTensorBoundsError,
    GGUFTensorReader,
    GGUFTensorReadError,
    GGUFTensorReadLimitError,
    GGUFTensorReadLimits,
    StaleGGUFTensorHandleError,
    UnknownGGUFTensorError,
    open_gguf,
)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _fixture(path: Path, *, payload_seed: int = 0) -> bytes:
    metadata = bytearray()
    metadata.extend(_string("general.alignment"))
    metadata.extend(struct.pack("<II", 4, 32))
    metadata.extend(_string("general.quantization_version"))
    metadata.extend(struct.pack("<II", 4, 2))

    descriptors = bytearray()
    descriptors.extend(_string("quant.weight"))
    descriptors.extend(struct.pack("<IQIQ", 1, 64, 8, 0))
    descriptors.extend(_string("dense.weight"))
    descriptors.extend(struct.pack("<IQIQ", 1, 8, 0, 96))

    data = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, 2, 2))
    data.extend(metadata)
    data.extend(descriptors)
    data.extend(bytes((-len(data)) % 32))
    payload = bytes((payload_seed + index) % 256 for index in range(128))
    data.extend(payload)
    path.write_bytes(data)
    return payload


def test_index_is_stable_and_does_not_read_payload_until_requested(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    payload = _fixture(path)

    with (
        open_gguf(path) as source,
        patch.object(source, "raw_bytes", wraps=source.raw_bytes) as raw_bytes,
    ):
        reader = GGUFTensorReader(source)
        quant = reader.index.tensor("quant.weight")

        assert raw_bytes.call_count == 0
        assert [handle.name for handle in reader.index.tensors] == [
            "quant.weight",
            "dense.weight",
        ]
        assert quant.ordinal == 0
        assert quant.byte_size == 68
        assert quant.encoded_block_bytes == 34
        assert quant.logical_block_values == 32
        assert quant.block_count == 2

        first = reader.read_blocks(quant, 0, 1)
        assert raw_bytes.call_count == 1
        assert first.data == payload[:34]
        assert first.tensor_byte_offset == 0
        assert first.element_offset == 0


def test_complete_block_chunks_are_bounded_and_cover_tensor_once(tmp_path: Path) -> None:
    path = tmp_path / "chunks.gguf"
    payload = _fixture(path, payload_seed=11)

    with open_gguf(path) as source:
        reader = GGUFTensorReader(
            source,
            limits=GGUFTensorReadLimits(max_chunk_bytes=68),
        )
        handle = reader.index.tensor("quant.weight")
        chunks = tuple(reader.iter_chunks(handle, max_chunk_bytes=34))

    assert [chunk.block_offset for chunk in chunks] == [0, 1]
    assert [chunk.block_count for chunk in chunks] == [1, 1]
    assert [chunk.element_offset for chunk in chunks] == [0, 32]
    assert b"".join(chunk.data for chunk in chunks) == payload[:68]


def test_byte_and_block_reads_cannot_escape_tensor_or_chunk_limit(tmp_path: Path) -> None:
    path = tmp_path / "bounds.gguf"
    _fixture(path)

    with open_gguf(path) as source:
        reader = GGUFTensorReader(
            source,
            limits=GGUFTensorReadLimits(max_chunk_bytes=34),
        )
        handle = reader.index.tensor("quant.weight")

        with pytest.raises(GGUFTensorBoundsError, match="escapes"):
            reader.read_bytes(handle, 67, 2)
        with pytest.raises(GGUFTensorBoundsError, match="escapes"):
            reader.read_blocks(handle, 2, 1)
        with pytest.raises(GGUFTensorBoundsError, match="non-negative"):
            reader.read_blocks(handle, -1, 1)
        with pytest.raises(GGUFTensorReadLimitError, match="chunk limit"):
            tuple(reader.iter_chunks(handle, max_chunk_bytes=33))
        with pytest.raises(GGUFTensorReadLimitError, match="requested 68"):
            reader.read_blocks(handle, 0, 2)


def test_foreign_or_modified_handles_fail_before_source_read(tmp_path: Path) -> None:
    first_path = tmp_path / "first.gguf"
    second_path = tmp_path / "second.gguf"
    _fixture(first_path)
    _fixture(second_path)

    with open_gguf(first_path) as first, open_gguf(second_path) as second:
        first_reader = GGUFTensorReader(first)
        second_reader = GGUFTensorReader(second)
        first_handle = first_reader.index.tensor("quant.weight")
        foreign_handle = second_reader.index.tensor("quant.weight")

        with patch.object(first, "raw_bytes", wraps=first.raw_bytes) as raw_bytes:
            with pytest.raises(StaleGGUFTensorHandleError, match="different GGUF source"):
                first_reader.read_blocks(foreign_handle, 0, 1)
            with pytest.raises(StaleGGUFTensorHandleError, match="does not match"):
                first_reader.read_blocks(replace(first_handle, byte_size=34), 0, 1)
            assert raw_bytes.call_count == 0


def test_unknown_names_and_closed_sources_have_exact_errors(tmp_path: Path) -> None:
    path = tmp_path / "closed.gguf"
    _fixture(path)
    source = open_gguf(path)
    reader = GGUFTensorReader(source)
    handle = reader.index.tensor("dense.weight")

    with pytest.raises(UnknownGGUFTensorError, match="missing"):
        reader.index.tensor("missing")
    source.close()
    with pytest.raises(GGUFTensorReadError, match="source is closed"):
        reader.read_bytes(handle, 0, 1)

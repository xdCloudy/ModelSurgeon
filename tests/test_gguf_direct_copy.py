"""Tests for bounded byte-identical GGUF tensor copying."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from modelsurgeon.adapters.gguf import (
    GGUFDiskEstimate,
    GGUFTensorReader,
    GGUFTensorReadLimitError,
    GGUFTensorReadLimits,
    GGUFValueType,
    GGUFWriteMetadata,
    GGUFWriteTensor,
    copy_unchanged_gguf_tensor,
    open_gguf,
    plan_gguf_output,
    preflight_gguf_disk,
    write_gguf_transactionally,
)

_METADATA = (
    GGUFWriteMetadata("general.architecture", GGUFValueType.STRING, "test"),
    GGUFWriteMetadata("general.quantization_version", GGUFValueType.UINT32, 2),
)


def _disk(path: Path, tensors: tuple[GGUFWriteTensor, ...]):
    layout = plan_gguf_output(_METADATA, tensors)
    return preflight_gguf_disk(
        path,
        path.parent,
        GGUFDiskEstimate(layout.total_bytes, 0),
    )


def test_unchanged_quantized_payload_is_identical_and_chunk_bounded(tmp_path: Path) -> None:
    source_path = tmp_path / "source.gguf"
    payload = bytes(range(68))
    source_tensor = GGUFWriteTensor("quant.weight", (64,), 8, (payload,))
    write_gguf_transactionally(
        source_path,
        _METADATA,
        (source_tensor,),
        _disk(source_path, (source_tensor,)),
    )

    destination = tmp_path / "copied.gguf"
    with open_gguf(source_path) as source:
        reader = GGUFTensorReader(
            source, limits=GGUFTensorReadLimits(max_chunk_bytes=34)
        )
        unchanged = copy_unchanged_gguf_tensor(
            reader, reader.index.tensor("quant.weight"), max_chunk_bytes=34
        )
        output_tensor = unchanged.as_write_tensor()
        with patch.object(source, "raw_bytes", wraps=source.raw_bytes) as reads:
            result = write_gguf_transactionally(
                destination,
                _METADATA,
                (output_tensor,),
                _disk(destination, (output_tensor,)),
            )

    assert unchanged.peak_payload_buffer_bytes == 34
    assert [call.args[1] for call in reads.call_args_list] == [34, 34]
    assert dict(result.tensor_sha256)["quant.weight"] == hashlib.sha256(payload).hexdigest()
    with open_gguf(destination) as copied:
        tensor = copied.container.tensor("quant.weight")
        assert tensor is not None
        assert copied.raw_bytes(tensor.data_offset, tensor.byte_size) == payload


def test_copy_rejects_chunk_smaller_than_one_block(tmp_path: Path) -> None:
    source_path = tmp_path / "source.gguf"
    payload = bytes(68)
    tensor = GGUFWriteTensor("quant.weight", (64,), 8, (payload,))
    write_gguf_transactionally(source_path, _METADATA, (tensor,), _disk(source_path, (tensor,)))

    with open_gguf(source_path) as source:
        reader = GGUFTensorReader(source, limits=GGUFTensorReadLimits(68))
        with pytest.raises(GGUFTensorReadLimitError, match="fit one encoded block"):
            copy_unchanged_gguf_tensor(
                reader, reader.index.tensor("quant.weight"), max_chunk_bytes=33
            )

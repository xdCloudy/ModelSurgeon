from __future__ import annotations

import hashlib
import math
from array import array
from pathlib import Path

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    Q4_K_CODEC,
    ByteOrder,
    CodecRegistry,
    GGMLQuantizationType,
    GGUFDiskEstimate,
    GGUFValueType,
    GGUFWriteMetadata,
    GGUFWriteTensor,
    open_gguf,
    plan_axis_edit,
    plan_gguf_output,
    preflight_gguf_disk,
    write_gguf_transactionally,
)
from modelsurgeon.surgery import (
    NativeGGUFLowRankError,
    NativeGGUFLowRankLimits,
    execute_native_gguf_low_rank_replacement,
)

_SHAPES = (
    ("token_embd.weight", (256, 256)),
    ("output_norm.weight", (256,)),
    ("blk.0.attn_norm.weight", (256,)),
    ("blk.0.ffn_norm.weight", (256,)),
    ("blk.0.attn_q.weight", (256, 256)),
    ("blk.0.attn_k.weight", (256, 128)),
    ("blk.0.attn_v.weight", (256, 128)),
    ("blk.0.attn_output.weight", (256, 256)),
    ("blk.0.ffn_gate.weight", (256, 512)),
    ("blk.0.ffn_up.weight", (256, 512)),
    ("blk.0.ffn_down.weight", (512, 256)),
)


def _metadata() -> tuple[GGUFWriteMetadata, ...]:
    return (
        GGUFWriteMetadata("general.architecture", GGUFValueType.STRING, "llama"),
        GGUFWriteMetadata("general.quantization_version", GGUFValueType.UINT32, 2),
        GGUFWriteMetadata("llama.block_count", GGUFValueType.UINT32, 1),
        GGUFWriteMetadata("llama.embedding_length", GGUFValueType.UINT32, 256),
        GGUFWriteMetadata("llama.feed_forward_length", GGUFValueType.UINT32, 512),
        GGUFWriteMetadata("llama.attention.head_count", GGUFValueType.UINT32, 8),
        GGUFWriteMetadata("llama.attention.head_count_kv", GGUFValueType.UINT32, 4),
    )


def _q4_payload(shape: tuple[int, ...]) -> bytes:
    values = array("f", (((index * 17) % 101 - 50) / 11 for index in range(256)))
    block = bytearray(Q4_K_CODEC.layout.type_size)
    Q4_K_CODEC.encode_blocks(values, memoryview(block), byte_order=ByteOrder.LITTLE)
    size = plan_axis_edit(GGMLQuantizationType.Q4_K, shape, 0).tensor_bytes
    return bytes(block) * (size // len(block))


def _write_fixture(path: Path) -> None:
    tensors: list[GGUFWriteTensor] = []
    for ordinal, (name, shape) in enumerate(_SHAPES):
        if len(shape) == 1:
            size = plan_axis_edit(GGMLQuantizationType.F32, shape, 0).tensor_bytes
            tensors.append(GGUFWriteTensor(name, shape, 0, (bytes((ordinal + 3,)) * size,)))
        else:
            tensors.append(GGUFWriteTensor(name, shape, 12, (_q4_payload(shape),)))
    tensor_tuple = tuple(tensors)
    layout = plan_gguf_output(_metadata(), tensor_tuple)
    disk = preflight_gguf_disk(
        path,
        path.parent,
        GGUFDiskEstimate(layout.total_bytes, 0, layout.alignment),
    )
    write_gguf_transactionally(path, _metadata(), tensor_tuple, disk)


def _registry() -> CodecRegistry:
    registry = CodecRegistry()
    registry.register(Q4_K_CODEC)
    return registry


def test_selected_q4_tensor_records_low_rank_and_requantization_error(tmp_path) -> None:
    source_path = tmp_path / "source.gguf"
    destination = tmp_path / "output.gguf"
    _write_fixture(source_path)
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    disk = preflight_gguf_disk(
        destination,
        destination.parent,
        GGUFDiskEstimate(source_path.stat().st_size * 2, 0),
    )

    with open_gguf(source_path) as source:
        result = execute_native_gguf_low_rank_replacement(
            source,
            ModelFamily.LLAMA,
            "blk.0.attn_q.weight",
            4,
            destination,
            disk,
            _registry(),
        )

    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_digest
    assert result.requested_rank == result.effective_rank == 4
    assert math.isfinite(result.relative_frobenius_error)
    assert result.relative_frobenius_error >= 0
    assert result.requantization_report.complete
    assert result.requantization_report.error_summaries
    assert result.requantization_mean_squared_error >= 0
    assert result.requantization_max_absolute_error >= 0
    output_hashes = dict(result.write_result.tensor_sha256)
    assert all(output_hashes[name] == digest for name, digest in result.unchanged_tensor_sha256)
    assert result.output_discovery.shape.layers == 1


def test_workspace_limit_rejects_before_publishing(tmp_path) -> None:
    source_path = tmp_path / "source.gguf"
    destination = tmp_path / "output.gguf"
    _write_fixture(source_path)
    disk = preflight_gguf_disk(
        destination,
        destination.parent,
        GGUFDiskEstimate(source_path.stat().st_size * 2, 0),
    )

    with open_gguf(source_path) as source, pytest.raises(NativeGGUFLowRankError, match="workspace"):
        execute_native_gguf_low_rank_replacement(
            source,
            ModelFamily.LLAMA,
            "blk.0.attn_q.weight",
            4,
            destination,
            disk,
            _registry(),
            limits=NativeGGUFLowRankLimits(max_workspace_bytes=1024),
        )

    assert not destination.exists()

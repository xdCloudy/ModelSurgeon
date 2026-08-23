"""End-to-end tests for bounded native quantized GGUF MLP execution."""

from __future__ import annotations

import hashlib
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
    discard_resumable_gguf,
    discover_gguf_components,
    open_gguf,
    plan_axis_edit,
    plan_gguf_output,
    preflight_gguf_disk,
    write_gguf_transactionally,
)
from modelsurgeon.surgery import (
    NativeGGUFMLPExecutionError,
    NativeGGUFMLPExecutionLimits,
    execute_native_gguf_mlp_channel_removal,
    plan_native_gguf_mlp_channel_removal,
)

_SHAPES = (
    ("token_embd.weight", (256, 256), 0),
    ("output_norm.weight", (256,), 0),
    ("blk.0.attn_norm.weight", (256,), 0),
    ("blk.0.ffn_norm.weight", (256,), 0),
    ("blk.0.attn_q.weight", (256, 256), 0),
    ("blk.0.attn_k.weight", (256, 128), 0),
    ("blk.0.attn_v.weight", (256, 128), 0),
    ("blk.0.attn_output.weight", (256, 256), 0),
    ("blk.0.ffn_gate.weight", (256, 512), 12),
    ("blk.0.ffn_up.weight", (256, 512), 12),
    ("blk.0.ffn_down.weight", (512, 256), 12),
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


def _q4_payload(shape: tuple[int, ...], byte_order: ByteOrder) -> bytes:
    values = array("f", ((index % 31 - 15) / 8 for index in range(256)))
    encoded = bytearray(Q4_K_CODEC.layout.type_size)
    Q4_K_CODEC.encode_blocks(values, memoryview(encoded), byte_order=byte_order)
    byte_size = plan_axis_edit(Q4_K_CODEC.identity.quant_type, shape, 0).tensor_bytes
    return bytes(encoded) * (byte_size // len(encoded))


def _write_fixture(path: Path, *, byte_order: ByteOrder = ByteOrder.LITTLE) -> None:
    tensors: list[GGUFWriteTensor] = []
    for ordinal, (name, shape, ggml_type_id) in enumerate(_SHAPES):
        if ggml_type_id == 12:
            payload = _q4_payload(shape, byte_order)
        else:
            byte_size = plan_axis_edit(GGMLQuantizationType.F32, shape, 0).tensor_bytes
            payload = bytes(((ordinal + 17) % 251,)) * byte_size
        tensors.append(GGUFWriteTensor(name, shape, ggml_type_id, (payload,)))
    tensor_tuple = tuple(tensors)
    layout = plan_gguf_output(_metadata(), tensor_tuple, byte_order=byte_order)
    disk = preflight_gguf_disk(
        path,
        path.parent,
        GGUFDiskEstimate(layout.total_bytes, 0, layout.alignment),
    )
    write_gguf_transactionally(
        path,
        _metadata(),
        tensor_tuple,
        disk,
        byte_order=byte_order,
    )

def _execute(
    source_path: Path,
    destination: Path,
    removed: tuple[int, ...],
    *,
    limits: NativeGGUFMLPExecutionLimits | None = None,
):
    registry = CodecRegistry()
    registry.register(Q4_K_CODEC)
    disk = preflight_gguf_disk(
        destination,
        destination.parent,
        GGUFDiskEstimate(source_path.stat().st_size * 2, 0),
    )
    with open_gguf(source_path) as source:
        discovery = discover_gguf_components(source.container, family=ModelFamily.LLAMA)
        plan = plan_native_gguf_mlp_channel_removal(
            discovery,
            layer_index=0,
            removed_channels=removed,
        )
        return execute_native_gguf_mlp_channel_removal(
            source,
            plan,
            destination,
            disk,
            registry,
            limits=limits,
        )


def test_scattered_removal_repacks_only_down_rows_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "input.gguf"
    destination = tmp_path / "output.gguf"
    _write_fixture(source, byte_order=ByteOrder.BIG)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    result = _execute(source, destination, tuple(range(0, 512, 2)))

    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
    assert result.output_discovery.shape.feed_forward_length == 256
    assert result.requantization_errors
    assert result.peak_row_working_bytes < 64 * 1024
    assert dict(result.write_result.tensor_sha256)["blk.0.attn_q.weight"] == dict(
        result.unchanged_tensor_sha256
    )["blk.0.attn_q.weight"]
    shapes = {
        tensor.descriptor.name: tensor.descriptor.dimensions
        for tensor in result.output_discovery.tensors
    }
    assert shapes["blk.0.ffn_gate.weight"] == (256, 256)
    assert shapes["blk.0.ffn_up.weight"] == (256, 256)
    assert shapes["blk.0.ffn_down.weight"] == (256, 256)
    with open_gguf(destination) as output:
        assert output.container.byte_order is ByteOrder.BIG
        assert output.container.version == 3
        assert output.container.alignment == 32


def test_aligned_removal_uses_only_encoded_copy_paths(tmp_path: Path) -> None:
    source = tmp_path / "input.gguf"
    destination = tmp_path / "output.gguf"
    _write_fixture(source)

    result = _execute(source, destination, tuple(range(256)))

    assert result.output_discovery.shape.feed_forward_length == 256
    assert result.requantization_errors == ()
    assert result.peak_row_working_bytes == 0


def test_row_memory_limit_fails_without_publishing_output(tmp_path: Path) -> None:
    source = tmp_path / "input.gguf"
    destination = tmp_path / "output.gguf"
    _write_fixture(source)

    with pytest.raises(NativeGGUFMLPExecutionError, match="one MLP row requires"):
        _execute(
            source,
            destination,
            tuple(range(0, 512, 2)),
            limits=NativeGGUFMLPExecutionLimits(max_row_working_bytes=1024),
        )

    assert not destination.exists()
    discard_resumable_gguf(destination)

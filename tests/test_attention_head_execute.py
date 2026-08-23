"""End-to-end tests for native quantized GGUF attention-head execution."""

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
    GGUFRequantizationLimits,
    NativeGGUFAttentionHeadExecutionError,
    NativeGGUFAttentionHeadExecutionLimits,
    execute_native_gguf_attention_head_removal,
    resolve_native_gguf_attention_head_removal_rules,
)


def _metadata(query_heads: int, kv_heads: int) -> tuple[GGUFWriteMetadata, ...]:
    return (
        GGUFWriteMetadata("general.architecture", GGUFValueType.STRING, "llama"),
        GGUFWriteMetadata("general.quantization_version", GGUFValueType.UINT32, 2),
        GGUFWriteMetadata("llama.block_count", GGUFValueType.UINT32, 1),
        GGUFWriteMetadata("llama.embedding_length", GGUFValueType.UINT32, 1024),
        GGUFWriteMetadata("llama.feed_forward_length", GGUFValueType.UINT32, 512),
        GGUFWriteMetadata("llama.attention.head_count", GGUFValueType.UINT32, query_heads),
        GGUFWriteMetadata("llama.attention.head_count_kv", GGUFValueType.UINT32, kv_heads),
    )


def _payload(
    shape: tuple[int, ...],
    byte_order: ByteOrder,
    *,
    varied: bool = False,
) -> bytes:
    values = array("f", ((index % 37 - 18) / 7 for index in range(256)))
    encoded = bytearray(Q4_K_CODEC.layout.type_size)
    Q4_K_CODEC.encode_blocks(values, memoryview(encoded), byte_order=byte_order)
    size = plan_axis_edit(Q4_K_CODEC.identity.quant_type, shape, 0).tensor_bytes
    count = size // len(encoded)
    if not varied:
        return bytes(encoded) * count
    alternate_values = array(
        "f", ((index % 29 - 7) / 3 for index in range(256))
    )
    alternate = bytearray(Q4_K_CODEC.layout.type_size)
    Q4_K_CODEC.encode_blocks(
        alternate_values,
        memoryview(alternate),
        byte_order=byte_order,
    )
    blocks = (bytes(encoded), bytes(alternate))
    return b"".join(blocks[index % 2] for index in range(count))


def _write_fixture(path: Path, query_heads: int, kv_heads: int) -> None:
    head_width = 1024 // query_heads
    shapes = (
        ("token_embd.weight", (1024, 256)),
        ("output_norm.weight", (1024,)),
        ("blk.0.attn_norm.weight", (1024,)),
        ("blk.0.ffn_norm.weight", (1024,)),
        ("blk.0.attn_q.weight", (1024, query_heads * head_width)),
        ("blk.0.attn_k.weight", (1024, kv_heads * head_width)),
        ("blk.0.attn_v.weight", (1024, kv_heads * head_width)),
        ("blk.0.attn_output.weight", (query_heads * head_width, 1024)),
        ("blk.0.ffn_gate.weight", (1024, 512)),
        ("blk.0.ffn_up.weight", (1024, 512)),
        ("blk.0.ffn_down.weight", (512, 1024)),
    )
    metadata = _metadata(query_heads, kv_heads)
    tensors = tuple(
        GGUFWriteTensor(
            name,
            shape,
            12,
            (
                _payload(
                    shape,
                    ByteOrder.LITTLE,
                    varied=name == "blk.0.attn_output.weight",
                ),
            ),
        )
        for name, shape in shapes
    )
    layout = plan_gguf_output(metadata, tensors)
    disk = preflight_gguf_disk(
        path,
        path.parent,
        GGUFDiskEstimate(layout.total_bytes, 0, layout.alignment),
    )
    write_gguf_transactionally(path, metadata, tensors, disk)


def _execute(
    source_path: Path,
    destination: Path,
    removed: tuple[int, ...],
    *,
    limits: NativeGGUFAttentionHeadExecutionLimits | None = None,
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
        rules = resolve_native_gguf_attention_head_removal_rules(
            discovery,
            removed_query_heads=removed,
        )
        return execute_native_gguf_attention_head_removal(
            source,
            rules,
            destination,
            disk,
            registry,
            limits=limits,
        )


def test_mha_removal_copies_aligned_qkvo_and_preserves_other_tensors(tmp_path: Path) -> None:
    source = tmp_path / "mha.gguf"
    destination = tmp_path / "mha-output.gguf"
    _write_fixture(source, 4, 4)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()

    result = _execute(source, destination, (1, 3))

    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
    assert result.output_discovery.shape.attention_heads == 2
    assert result.output_discovery.shape.kv_heads == 2
    assert result.output_discovery.shape.key_length == 256
    assert result.requantization_errors == ()
    assert result.peak_row_working_bytes == 0
    shapes = {
        item.descriptor.name: item.descriptor.dimensions
        for item in result.output_discovery.tensors
    }
    assert shapes["blk.0.attn_q.weight"] == (1024, 512)
    assert shapes["blk.0.attn_k.weight"] == (1024, 512)
    assert shapes["blk.0.attn_v.weight"] == (1024, 512)
    assert shapes["blk.0.attn_output.weight"] == (512, 1024)
    assert dict(result.write_result.tensor_sha256)["blk.0.ffn_up.weight"] == dict(
        result.unchanged_tensor_sha256
    )["blk.0.ffn_up.weight"]


def test_gqa_removal_repacks_only_output_rows_and_keeps_kv_payloads(tmp_path: Path) -> None:
    source = tmp_path / "gqa.gguf"
    destination = tmp_path / "gqa-output.gguf"
    _write_fixture(source, 8, 2)

    result = _execute(source, destination, (1, 5))

    assert result.output_discovery.shape.attention_heads == 6
    assert result.output_discovery.shape.kv_heads == 2
    assert len(result.requantization_errors) == 1024
    assert result.peak_row_working_bytes < 32 * 1024
    unchanged = dict(result.unchanged_tensor_sha256)
    output = dict(result.write_result.tensor_sha256)
    assert output["blk.0.attn_k.weight"] == unchanged["blk.0.attn_k.weight"]
    assert output["blk.0.attn_v.weight"] == unchanged["blk.0.attn_v.weight"]
    shapes = {
        item.descriptor.name: item.descriptor.dimensions
        for item in result.output_discovery.tensors
    }
    assert shapes["blk.0.attn_q.weight"] == (1024, 768)
    assert shapes["blk.0.attn_output.weight"] == (768, 1024)


def test_repack_memory_ceiling_aborts_without_publication(tmp_path: Path) -> None:
    source = tmp_path / "gqa.gguf"
    destination = tmp_path / "output.gguf"
    _write_fixture(source, 8, 2)

    with pytest.raises(NativeGGUFAttentionHeadExecutionError, match="one attention row"):
        _execute(
            source,
            destination,
            (1, 5),
            limits=NativeGGUFAttentionHeadExecutionLimits(
                max_row_working_bytes=1024
            ),
        )

    assert not destination.exists()
    discard_resumable_gguf(destination)


def test_requantization_error_ceiling_aborts_without_publication(tmp_path: Path) -> None:
    source = tmp_path / "gqa.gguf"
    destination = tmp_path / "output.gguf"
    _write_fixture(source, 8, 2)

    with pytest.raises(NativeGGUFAttentionHeadExecutionError, match="max error"):
        _execute(
            source,
            destination,
            (1, 5),
            limits=NativeGGUFAttentionHeadExecutionLimits(
                requantization=GGUFRequantizationLimits(max_absolute_error=0.0)
            ),
        )

    assert not destination.exists()
    discard_resumable_gguf(destination)

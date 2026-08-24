from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    GGMLQuantizationType,
    GGUFDiskEstimate,
    GGUFValueType,
    GGUFWriteMetadata,
    GGUFWriteTensor,
    discover_gguf_components,
    open_gguf,
    plan_axis_edit,
    plan_gguf_output,
    preflight_gguf_disk,
    write_gguf_transactionally,
)
from modelsurgeon.surgery import (
    NativeGGUFLayerRemovalError,
    execute_native_gguf_transformer_layer_removal,
    plan_native_gguf_transformer_layer_removal,
)


def _metadata() -> tuple[GGUFWriteMetadata, ...]:
    return (
        GGUFWriteMetadata("general.architecture", GGUFValueType.STRING, "llama"),
        GGUFWriteMetadata("general.quantization_version", GGUFValueType.UINT32, 2),
        GGUFWriteMetadata("llama.block_count", GGUFValueType.UINT32, 3),
        GGUFWriteMetadata("llama.embedding_length", GGUFValueType.UINT32, 256),
        GGUFWriteMetadata("llama.feed_forward_length", GGUFValueType.UINT32, 512),
        GGUFWriteMetadata("llama.attention.head_count", GGUFValueType.UINT32, 8),
        GGUFWriteMetadata("llama.attention.head_count_kv", GGUFValueType.UINT32, 4),
    )


def _shapes() -> tuple[tuple[str, tuple[int, ...]], ...]:
    result = [("token_embd.weight", (256, 256)), ("output_norm.weight", (256,))]
    for layer in range(3):
        prefix = f"blk.{layer}"
        result.extend(
            (
                (f"{prefix}.attn_norm.weight", (256,)),
                (f"{prefix}.ffn_norm.weight", (256,)),
                (f"{prefix}.attn_q.weight", (256, 256)),
                (f"{prefix}.attn_k.weight", (256, 128)),
                (f"{prefix}.attn_v.weight", (256, 128)),
                (f"{prefix}.attn_output.weight", (256, 256)),
                (f"{prefix}.ffn_gate.weight", (256, 512)),
                (f"{prefix}.ffn_up.weight", (256, 512)),
                (f"{prefix}.ffn_down.weight", (512, 256)),
            )
        )
    return tuple(result)


def _write_fixture(path: Path) -> None:
    tensors: list[GGUFWriteTensor] = []
    for ordinal, (name, shape) in enumerate(_shapes()):
        quant_type = GGMLQuantizationType.F32 if len(shape) == 1 else GGMLQuantizationType.Q4_K
        ggml_type_id = 0 if quant_type is GGMLQuantizationType.F32 else 12
        byte_size = plan_axis_edit(quant_type, shape, 0).tensor_bytes
        payload = bytes(((ordinal + 11) % 251,)) * byte_size
        tensors.append(GGUFWriteTensor(name, shape, ggml_type_id, (payload,)))
    tensor_tuple = tuple(tensors)
    layout = plan_gguf_output(_metadata(), tensor_tuple)
    disk = preflight_gguf_disk(
        path,
        path.parent,
        GGUFDiskEstimate(layout.total_bytes, 0, layout.alignment),
    )
    write_gguf_transactionally(path, _metadata(), tensor_tuple, disk)


def test_middle_quantized_layer_is_omitted_and_following_payloads_are_renamed(tmp_path) -> None:
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
        discovery = discover_gguf_components(source.container, family=ModelFamily.LLAMA)
        plan = plan_native_gguf_transformer_layer_removal(discovery, (1,))
        result = execute_native_gguf_transformer_layer_removal(
            source, plan, destination, disk, copy_chunk_bytes=4096
        )

    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_digest
    assert result.plan.layer_mapping == ((0, 0), (1, None), (2, 1))
    assert result.output_discovery.shape.layers == 2
    assert result.output_discovery.parameter_count < discovery.parameter_count
    assert result.peak_copy_buffer_bytes <= 4096
    assert result.omitted_tensor_names
    assert all(name.startswith("blk.1.") for name in result.omitted_tensor_names)
    names = {item.descriptor.name for item in result.output_discovery.tensors}
    assert not any(name.startswith("blk.2.") for name in names)
    renamed = next(
        item for item in result.retained_tensors if item.source_name == "blk.2.attn_q.weight"
    )
    assert renamed.output_name == "blk.1.attn_q.weight"
    assert dict(result.write_result.tensor_sha256)[renamed.output_name] == renamed.payload_sha256


def test_plan_rejects_removing_every_layer(tmp_path) -> None:
    source_path = tmp_path / "source.gguf"
    _write_fixture(source_path)

    with open_gguf(source_path) as source:
        discovery = discover_gguf_components(source.container, family=ModelFamily.LLAMA)
        with pytest.raises(NativeGGUFLayerRemovalError, match="every layer"):
            plan_native_gguf_transformer_layer_removal(discovery, (0, 1, 2))

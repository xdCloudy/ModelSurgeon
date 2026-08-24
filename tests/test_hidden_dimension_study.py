from __future__ import annotations

import struct
from pathlib import Path

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    GGMLQuantizationType,
    discover_gguf_components,
    open_gguf,
    plan_axis_edit,
)
from modelsurgeon.surgery import (
    HiddenDimensionStudyError,
    evaluate_coordinated_hidden_dimension_surgery,
)

_TENSORS = (
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


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _metadata_uint32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<II", 4, value)


def _write_model(path: Path, architecture: str) -> None:
    metadata = (
        _metadata_string("general.architecture", architecture)
        + _metadata_uint32("general.alignment", 32)
        + _metadata_uint32("general.quantization_version", 2)
        + _metadata_uint32(f"{architecture}.block_count", 1)
        + _metadata_uint32(f"{architecture}.embedding_length", 256)
        + _metadata_uint32(f"{architecture}.feed_forward_length", 512)
        + _metadata_uint32(f"{architecture}.attention.head_count", 8)
        + _metadata_uint32(f"{architecture}.attention.head_count_kv", 4)
    )
    descriptors = bytearray()
    relative_offset = 0
    ranges: list[tuple[int, int]] = []
    for name, dimensions in _TENSORS:
        quant_type = GGMLQuantizationType.F32 if len(dimensions) == 1 else GGMLQuantizationType.Q4_K
        ggml_type_id = 0 if quant_type is GGMLQuantizationType.F32 else 12
        size = plan_axis_edit(quant_type, dimensions, 0).tensor_bytes
        descriptors.extend(_string(name))
        descriptors.extend(struct.pack("<I", len(dimensions)))
        descriptors.extend(struct.pack(f"<{len(dimensions)}Q", *dimensions))
        descriptors.extend(struct.pack("<IQ", ggml_type_id, relative_offset))
        ranges.append((relative_offset, size))
        relative_offset = (relative_offset + size + 31) // 32 * 32
    data = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(_TENSORS), 8))
    data.extend(metadata)
    data.extend(descriptors)
    data.extend(bytes((-len(data)) % 32))
    data.extend(bytes(max(offset + size for offset, size in ranges)))
    path.write_bytes(data)


def _discoveries(tmp_path: Path):
    llama_path = tmp_path / "llama.gguf"
    qwen_path = tmp_path / "qwen.gguf"
    _write_model(llama_path, "llama")
    _write_model(qwen_path, "qwen2")
    with open_gguf(llama_path) as source:
        llama = discover_gguf_components(source.container, family=ModelFamily.LLAMA)
    with open_gguf(qwen_path) as source:
        qwen = discover_gguf_components(source.container, family=ModelFamily.QWEN)
    return llama, qwen


def test_two_family_study_documents_rejection_without_physical_mutation(tmp_path) -> None:
    study = evaluate_coordinated_hidden_dimension_surgery(_discoveries(tmp_path))

    assert not study.physically_feasible
    assert not study.physical_mutation_implemented
    assert [item.family for item in study.assessments] == [ModelFamily.LLAMA, ModelFamily.QWEN]
    for item in study.assessments:
        assert item.hidden_dimension == 256
        assert item.key_head_dimension == item.value_head_dimension == 32
        assert item.quantized_axis0_granularity == 256
        assert "token_embd.weight" in item.globally_coupled_tensors
        assert len(item.normalization_tensors) == 3
        assert not item.feasible_operation_classes
        assert any("rotary" in reason for reason in item.rejection_reasons)
        assert any("tying" in reason for reason in item.rejection_reasons)


def test_study_enforces_family_and_tensor_bounds(tmp_path) -> None:
    llama, qwen = _discoveries(tmp_path)

    with pytest.raises(HiddenDimensionStudyError, match="one Llama and one Qwen"):
        evaluate_coordinated_hidden_dimension_surgery((llama, llama))
    with pytest.raises(HiddenDimensionStudyError, match="tensor count"):
        evaluate_coordinated_hidden_dimension_surgery((llama, qwen), max_tensors_per_family=1)

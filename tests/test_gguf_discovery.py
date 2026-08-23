"""Tests for GGUF component, shape, and coupling graph discovery."""

from __future__ import annotations

import struct
from functools import reduce
from operator import mul
from pathlib import Path

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    GGUFDiscoveryError,
    GGUFTensorShapeError,
    MissingGGUFTensorError,
    discover_gguf_components,
    open_gguf,
)
from modelsurgeon.graph import ConstraintKind, EdgeKind, validate_component_graph

Tensor = tuple[str, tuple[int, ...]]


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _metadata_uint32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<II", 4, value)


def _model_tensors() -> tuple[Tensor, ...]:
    return (
        ("token_embd.weight", (8, 32)),
        ("output_norm.weight", (8,)),
        ("blk.0.attn_norm.weight", (8,)),
        ("blk.0.ffn_norm.weight", (8,)),
        ("blk.0.attn_q.weight", (8, 8)),
        ("blk.0.attn_k.weight", (8, 4)),
        ("blk.0.attn_v.weight", (8, 4)),
        ("blk.0.attn_output.weight", (8, 8)),
        ("blk.0.ffn_gate.weight", (8, 16)),
        ("blk.0.ffn_up.weight", (8, 16)),
        ("blk.0.ffn_down.weight", (16, 8)),
    )


def _write_model(
    path: Path,
    *,
    architecture: str = "qwen2",
    tensors: tuple[Tensor, ...] | None = None,
) -> tuple[Tensor, ...]:
    tensor_records = tensors or _model_tensors()
    metadata = (
        _metadata_string("general.architecture", architecture)
        + _metadata_uint32("general.alignment", 32)
        + _metadata_uint32(f"{architecture}.block_count", 1)
        + _metadata_uint32(f"{architecture}.embedding_length", 8)
        + _metadata_uint32(f"{architecture}.feed_forward_length", 16)
        + _metadata_uint32(f"{architecture}.attention.head_count", 2)
        + _metadata_uint32(f"{architecture}.attention.head_count_kv", 1)
    )
    descriptors = bytearray()
    relative_offset = 0
    ranges: list[tuple[int, int]] = []
    for name, dimensions in tensor_records:
        byte_size = reduce(mul, dimensions, 1) * 4
        descriptors.extend(_string(name))
        descriptors.extend(struct.pack("<I", len(dimensions)))
        descriptors.extend(struct.pack(f"<{len(dimensions)}Q", *dimensions))
        descriptors.extend(struct.pack("<IQ", 0, relative_offset))
        ranges.append((relative_offset, byte_size))
        relative_offset = (relative_offset + byte_size + 31) // 32 * 32

    data = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensor_records), 7))
    data.extend(metadata)
    data.extend(descriptors)
    data.extend(bytes((-len(data)) % 32))
    payload_size = max((offset + size for offset, size in ranges), default=0)
    data.extend(bytes(payload_size))
    path.write_bytes(data)
    return tensor_records


def test_discovers_reconciled_physical_graph_and_exact_couplings(tmp_path: Path) -> None:
    path = tmp_path / "qwen2.gguf"
    tensors = _write_model(path)

    with open_gguf(path) as source:
        discovery = discover_gguf_components(source.container)

    expected_parameters = sum(reduce(mul, shape, 1) for _, shape in tensors)
    assert discovery.family is ModelFamily.QWEN
    assert discovery.architecture == "qwen2"
    assert discovery.shape.layers == 1
    assert discovery.shape.embedding_length == 8
    assert discovery.shape.attention_heads == 2
    assert discovery.shape.kv_heads == 1
    assert discovery.parameter_count == expected_parameters
    assert sum(tensor.element_count for tensor in discovery.tensors) == expected_parameters
    assert discovery.to_record()["tensor_count"] == len(tensors)
    assert validate_component_graph(discovery.graph).valid

    parameter_nodes = [node for node in discovery.graph.nodes if node.kind == "parameter"]
    assert len(parameter_nodes) == len(tensors)
    assert sum(int(dict(node.attributes)["element_count"]) for node in parameter_nodes) == (
        expected_parameters
    )
    key_node = next(
        node
        for node in parameter_nodes
        if dict(node.attributes)["physical_name"] == "blk.0.attn_k.weight"
    )
    assert dict(key_node.attributes) == {
        "physical_name": "blk.0.attn_k.weight",
        "role": "attention_k",
        "ggml_type": "F32",
        "rank": 2,
        "element_count": 32,
        "storage_bytes": 128,
        "dimension_0": 8,
        "axis_0": "input_feature",
        "dimension_1": 4,
        "axis_1": "kv_head",
    }
    assert [constraint.kind for constraint in discovery.graph.constraints] == [
        ConstraintKind.SAME_HEAD_SET,
        ConstraintKind.SAME_HEAD_SET,
        ConstraintKind.SAME_HIDDEN_SIZE,
    ]
    assert sum(
        edge.kind is EdgeKind.COUPLED for edge in discovery.graph.edges
    ) == 5


def test_missing_required_tensor_and_shape_mismatch_fail_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.gguf"
    _write_model(
        missing,
        tensors=tuple(item for item in _model_tensors() if item[0] != "blk.0.attn_v.weight"),
    )
    with (
        open_gguf(missing) as source,
        pytest.raises(MissingGGUFTensorError, match=r"attn_v\.weight"),
    ):
        discover_gguf_components(source.container)

    mismatch = tmp_path / "mismatch.gguf"
    _write_model(
        mismatch,
        tensors=tuple(
            (name, (8, 8)) if name == "blk.0.attn_k.weight" else (name, shape)
            for name, shape in _model_tensors()
        ),
    )
    with (
        open_gguf(mismatch) as source,
        pytest.raises(GGUFTensorShapeError, match=r"kv_head.*size 8, expected 4"),
    ):
        discover_gguf_components(source.container)


def test_unknown_architecture_and_unmapped_tensor_fail_closed(tmp_path: Path) -> None:
    unknown_family = tmp_path / "unknown-family.gguf"
    _write_model(unknown_family, architecture="future")
    with (
        open_gguf(unknown_family) as source,
        pytest.raises(GGUFDiscoveryError, match="unsupported GGUF"),
    ):
        discover_gguf_components(source.container)

    unknown_tensor = tmp_path / "unknown-tensor.gguf"
    _write_model(
        unknown_tensor,
        tensors=(*_model_tensors(), ("blk.0.future.weight", (8, 8))),
    )
    with (
        open_gguf(unknown_tensor) as source,
        pytest.raises(GGUFDiscoveryError, match="no mapping"),
    ):
        discover_gguf_components(source.container)


def test_ambiguous_llama_alias_requires_explicit_family(tmp_path: Path) -> None:
    path = tmp_path / "llama.gguf"
    _write_model(path, architecture="llama")

    with open_gguf(path) as source:
        with pytest.raises(GGUFDiscoveryError, match="multiple families"):
            discover_gguf_components(source.container)
        discovery = discover_gguf_components(source.container, family=ModelFamily.LLAMA)

    assert discovery.family is ModelFamily.LLAMA

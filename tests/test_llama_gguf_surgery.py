"""Tests for the versioned Llama GGUF physical-surgery adapter."""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Any

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    CouplingKind,
    LlamaGGUFSurgeryError,
    MetadataSemantic,
    MissingGGUFTensorError,
    TensorRole,
    build_llama_gguf_surgery_adapter,
    discover_gguf_components,
    load_llama_gguf_surgery_adapter,
    open_gguf,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "llama_gguf_surgery_v1.json"


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _metadata_string(key: str, value: str) -> bytes:
    return _string(key) + struct.pack("<I", 8) + _string(value)


def _metadata_uint32(key: str, value: int) -> bytes:
    return _string(key) + struct.pack("<II", 4, value)


def _fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _write_fixture(path: Path, fixture: dict[str, Any]) -> None:
    architecture = fixture["architecture"]
    values = fixture["metadata"]
    metadata = _metadata_string("general.architecture", architecture)
    metadata += _metadata_uint32("general.alignment", 32)
    for suffix, value in values.items():
        metadata += _metadata_uint32(f"{architecture}.{suffix}", value)

    descriptors = bytearray()
    offset = 0
    ranges: list[tuple[int, int]] = []
    for name, dimensions in fixture["tensors"]:
        byte_size = reduce(mul, dimensions, 1) * 4
        descriptors.extend(_string(name))
        descriptors.extend(struct.pack("<I", len(dimensions)))
        descriptors.extend(struct.pack(f"<{len(dimensions)}Q", *dimensions))
        descriptors.extend(struct.pack("<IQ", 0, offset))
        ranges.append((offset, byte_size))
        offset = (offset + byte_size + 31) // 32 * 32

    data = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(fixture["tensors"]), 7))
    data.extend(metadata)
    data.extend(descriptors)
    data.extend(bytes((-len(data)) % 32))
    data.extend(bytes(max(start + size for start, size in ranges)))
    path.write_bytes(data)


def test_pinned_llama_metadata_fixture_builds_complete_surgery_view(tmp_path: Path) -> None:
    path = tmp_path / "llama.gguf"
    _write_fixture(path, _fixture())

    with open_gguf(path) as source:
        adapter = load_llama_gguf_surgery_adapter(source.container)

    assert adapter.to_record() == {
        "family": "llama",
        "architecture": "llama",
        "contract_version": 1,
        "layer_count": 1,
        "head_count": 4,
        "kv_head_count": 2,
        "head_width": 2,
        "query_heads_per_kv": 2,
        "output_weight_present": True,
    }
    assert adapter.attention.query_width == 8
    assert adapter.attention.kv_width == 4
    assert adapter.layer(0).tensor(TensorRole.ATTENTION_K).tensor_name == (
        "blk.0.attn_k.weight"
    )
    assert [group.kind for group in adapter.layer(0).coupling_groups] == [
        CouplingKind.ATTENTION_HEAD,
        CouplingKind.KV_HEAD,
        CouplingKind.MLP_CHANNEL,
    ]


def test_metadata_and_layer_renames_reuse_explicit_llama_contract(tmp_path: Path) -> None:
    path = tmp_path / "llama.gguf"
    _write_fixture(path, _fixture())
    with open_gguf(path) as source:
        adapter = load_llama_gguf_surgery_adapter(source.container)

    assert adapter.metadata_updates(((MetadataSemantic.BLOCK_COUNT, 3),)) == (
        ("llama.block_count", 3),
    )
    assert adapter.rename_tensor_blocks("blk.0.ffn_up.weight", ((0, 2),)) == (
        "blk.2.ffn_up.weight"
    )
    with pytest.raises(LlamaGGUFSurgeryError, match="outside block_count"):
        adapter.layer(1)


def test_missing_required_llama_tensor_fails_during_load(tmp_path: Path) -> None:
    fixture = _fixture()
    fixture["tensors"] = [
        item for item in fixture["tensors"] if item[0] != "blk.0.attn_v.weight"
    ]
    path = tmp_path / "missing.gguf"
    _write_fixture(path, fixture)

    with (
        open_gguf(path) as source,
        pytest.raises(MissingGGUFTensorError, match=r"attn_v\.weight"),
    ):
        load_llama_gguf_surgery_adapter(source.container)


def test_unsupported_contract_and_invalid_gqa_geometry_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "llama.gguf"
    _write_fixture(path, _fixture())
    with open_gguf(path) as source:
        discovery = discover_gguf_components(source.container, family=ModelFamily.LLAMA)

    with pytest.raises(LlamaGGUFSurgeryError, match="contract version 2"):
        build_llama_gguf_surgery_adapter(replace(discovery, contract_version=2))
    invalid_shape = replace(discovery.shape, kv_heads=3)
    with pytest.raises(LlamaGGUFSurgeryError, match="divisible"):
        build_llama_gguf_surgery_adapter(replace(discovery, shape=invalid_shape))

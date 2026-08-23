"""Tests for explicit dense Qwen2/Qwen3 GGUF surgery support."""

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
    QWEN_GGUF_VARIANTS,
    CouplingKind,
    QwenGGUFSurgeryError,
    TensorRole,
    build_qwen_gguf_surgery_adapter,
    discover_gguf_components,
    load_qwen_gguf_surgery_adapter,
    open_gguf,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "qwen_gguf_surgery_v1.json"


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _entry(key: str, value: str | int) -> bytes:
    if isinstance(value, str):
        return _string(key) + struct.pack("<I", 8) + _string(value)
    return _string(key) + struct.pack("<II", 4, value)


def _fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _write(path: Path, architecture: str) -> None:
    fixture = _fixture()
    metadata = _entry("general.architecture", architecture) + _entry(
        "general.alignment", 32
    )
    for suffix, value in fixture["metadata"].items():
        metadata += _entry(f"{architecture}.{suffix}", value)
    descriptors = bytearray()
    ranges: list[tuple[int, int]] = []
    offset = 0
    for name, dimensions in fixture["tensors"]:
        size = reduce(mul, dimensions, 1) * 4
        descriptors.extend(_string(name))
        descriptors.extend(struct.pack("<I", len(dimensions)))
        descriptors.extend(struct.pack(f"<{len(dimensions)}Q", *dimensions))
        descriptors.extend(struct.pack("<IQ", 0, offset))
        ranges.append((offset, size))
        offset = (offset + size + 31) // 32 * 32
    data = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(fixture["tensors"]), 7))
    data.extend(metadata)
    data.extend(descriptors)
    data.extend(bytes((-len(data)) % 32))
    data.extend(bytes(max(start + size for start, size in ranges)))
    path.write_bytes(data)


@pytest.mark.parametrize(("architecture", "generation"), [("qwen2", 2), ("qwen3", 3)])
def test_dense_qwen_fixtures_build_valid_explicit_views(
    tmp_path: Path, architecture: str, generation: int
) -> None:
    path = tmp_path / f"{architecture}.gguf"
    _write(path, architecture)
    with open_gguf(path) as source:
        adapter = load_qwen_gguf_surgery_adapter(source.container)

    assert adapter.variant.generation == generation
    assert adapter.variant.native_surgery is True
    assert adapter.attention.head_width == 2
    assert adapter.attention.kv_width == 4
    assert adapter.attention.query_heads_per_kv == 2
    assert adapter.layer(0).tensor(TensorRole.MLP_DOWN).tensor_name == (
        "blk.0.ffn_down.weight"
    )
    assert [group.kind for group in adapter.layer(0).coupling_groups] == [
        CouplingKind.ATTENTION_HEAD,
        CouplingKind.KV_HEAD,
        CouplingKind.MLP_CHANNEL,
    ]
    assert adapter.to_record()["architecture"] == architecture


@pytest.mark.parametrize("architecture", ["qwen2moe", "qwen3moe"])
def test_moe_variants_are_explicit_and_fail_before_planning(
    tmp_path: Path, architecture: str
) -> None:
    path = tmp_path / f"{architecture}.gguf"
    _write(path, architecture)
    with (
        open_gguf(path) as source,
        pytest.raises(QwenGGUFSurgeryError, match="expert/router"),
    ):
        load_qwen_gguf_surgery_adapter(source.container)
    variant = next(item for item in QWEN_GGUF_VARIANTS if item.architecture == architecture)
    assert variant.mixture_of_experts is True
    assert variant.native_surgery is False


def test_qwen_contract_version_and_gqa_constraints_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "qwen2.gguf"
    _write(path, "qwen2")
    with open_gguf(path) as source:
        discovery = discover_gguf_components(source.container, family=ModelFamily.QWEN)

    with pytest.raises(QwenGGUFSurgeryError, match="contract version 2"):
        build_qwen_gguf_surgery_adapter(replace(discovery, contract_version=2))
    with pytest.raises(QwenGGUFSurgeryError, match="divisible"):
        build_qwen_gguf_surgery_adapter(
            replace(discovery, shape=replace(discovery.shape, kv_heads=3))
        )

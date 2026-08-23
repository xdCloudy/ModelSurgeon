"""Tests for the finite Gemma GGUF physical-surgery boundary."""

from __future__ import annotations

import struct
from dataclasses import replace
from functools import reduce
from operator import mul
from pathlib import Path

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    GEMMA_GGUF_VARIANTS,
    CouplingKind,
    GemmaGGUFSurgeryError,
    build_gemma_gguf_surgery_adapter,
    discover_gguf_components,
    load_gemma_gguf_surgery_adapter,
    open_gguf,
)

_TENSORS = (
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


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _entry(key: str, value: str | int) -> bytes:
    if isinstance(value, str):
        return _string(key) + struct.pack("<I", 8) + _string(value)
    return _string(key) + struct.pack("<II", 4, value)


def _write(path: Path, architecture: str) -> None:
    metadata = _entry("general.architecture", architecture) + _entry(
        "general.alignment", 32
    )
    for suffix, value in (
        ("block_count", 1),
        ("embedding_length", 8),
        ("feed_forward_length", 16),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
    ):
        metadata += _entry(f"{architecture}.{suffix}", value)
    descriptors = bytearray()
    ranges: list[tuple[int, int]] = []
    offset = 0
    for name, dimensions in _TENSORS:
        size = reduce(mul, dimensions, 1) * 4
        descriptors.extend(_string(name))
        descriptors.extend(struct.pack("<I", len(dimensions)))
        descriptors.extend(struct.pack(f"<{len(dimensions)}Q", *dimensions))
        descriptors.extend(struct.pack("<IQ", 0, offset))
        ranges.append((offset, size))
        offset = (offset + size + 31) // 32 * 32
    data = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(_TENSORS), 7))
    data.extend(metadata)
    data.extend(descriptors)
    data.extend(bytes((-len(data)) % 32))
    data.extend(bytes(max(start + size for start, size in ranges)))
    path.write_bytes(data)


def test_gemma_v1_fixture_builds_normalization_and_coupling_view(tmp_path: Path) -> None:
    path = tmp_path / "gemma.gguf"
    _write(path, "gemma")
    with open_gguf(path) as source:
        adapter = load_gemma_gguf_surgery_adapter(source.container)
    assert adapter.variant.generation == 1
    assert adapter.variant.constraint == "dense-rmsnorm"
    assert adapter.attention.head_width == 2
    assert adapter.output_weight_present is False
    assert [group.kind for group in adapter.layer(0).coupling_groups] == [
        CouplingKind.ATTENTION_HEAD,
        CouplingKind.KV_HEAD,
        CouplingKind.MLP_CHANNEL,
    ]


@pytest.mark.parametrize("architecture", ["gemma2", "gemma3"])
def test_later_gemma_variants_are_recognized_but_fail_closed(
    tmp_path: Path, architecture: str
) -> None:
    path = tmp_path / f"{architecture}.gguf"
    _write(path, architecture)
    with (
        open_gguf(path) as source,
        pytest.raises(GemmaGGUFSurgeryError, match="requires unsupported"),
    ):
        load_gemma_gguf_surgery_adapter(source.container)
    variant = next(item for item in GEMMA_GGUF_VARIANTS if item.architecture == architecture)
    assert variant.native_surgery is False


def test_gemma_future_contract_and_invalid_gqa_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "gemma.gguf"
    _write(path, "gemma")
    with open_gguf(path) as source:
        discovery = discover_gguf_components(source.container, family=ModelFamily.GEMMA)
    with pytest.raises(GemmaGGUFSurgeryError, match="contract version 2"):
        build_gemma_gguf_surgery_adapter(replace(discovery, contract_version=2))
    with pytest.raises(GemmaGGUFSurgeryError, match="divisible"):
        build_gemma_gguf_surgery_adapter(
            replace(discovery, shape=replace(discovery.shape, kv_heads=3))
        )

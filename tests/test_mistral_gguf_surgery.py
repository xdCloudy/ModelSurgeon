"""Tests for native and legacy-prefix Mistral GGUF surgery mappings."""

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
    MistralGGUFSurgeryError,
    build_mistral_gguf_surgery_adapter,
    discover_gguf_components,
    load_mistral_gguf_surgery_adapter,
    open_gguf,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "mistral_gguf_surgery_v1.json"


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _entry(key: str, value: str | int) -> bytes:
    if isinstance(value, str):
        return _string(key) + struct.pack("<I", 8) + _string(value)
    return _string(key) + struct.pack("<II", 4, value)


def _fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _write(path: Path, architecture: str, *, include_window: bool = True) -> None:
    fixture = _fixture()
    values = dict(fixture["metadata"])
    if not include_window:
        values.pop("attention.sliding_window")
    metadata = _entry("general.architecture", architecture) + _entry(
        "general.alignment", 32
    )
    for suffix, value in values.items():
        metadata += _entry(f"{architecture}.{suffix}", value)
    descriptors = bytearray()
    offset = 0
    ranges: list[tuple[int, int]] = []
    for name, dimensions in fixture["tensors"]:
        size = reduce(mul, dimensions, 1) * 4
        descriptors.extend(_string(name))
        descriptors.extend(struct.pack("<I", len(dimensions)))
        descriptors.extend(struct.pack(f"<{len(dimensions)}Q", *dimensions))
        descriptors.extend(struct.pack("<IQ", 0, offset))
        ranges.append((offset, size))
        offset = (offset + size + 31) // 32 * 32
    metadata_count = 2 + len(values)
    data = bytearray(
        struct.pack("<4sIQQ", b"GGUF", 3, len(fixture["tensors"]), metadata_count)
    )
    data.extend(metadata)
    data.extend(descriptors)
    data.extend(bytes((-len(data)) % 32))
    data.extend(bytes(max(start + size for start, size in ranges)))
    path.write_bytes(data)


@pytest.mark.parametrize(
    ("architecture", "legacy"), [("mistral", False), ("llama", True)]
)
def test_supported_mistral_prefix_fixtures_build_valid_views(
    tmp_path: Path, architecture: str, legacy: bool
) -> None:
    path = tmp_path / f"{architecture}.gguf"
    _write(path, architecture)
    with open_gguf(path) as source:
        adapter = load_mistral_gguf_surgery_adapter(source.container)

    assert adapter.legacy_llama_prefix is legacy
    assert adapter.attention.sliding_window == 4096
    assert adapter.attention.head_width == 2
    assert adapter.attention.query_heads_per_kv == 2
    assert [group.kind for group in adapter.layer(0).coupling_groups] == [
        CouplingKind.ATTENTION_HEAD,
        CouplingKind.KV_HEAD,
        CouplingKind.MLP_CHANNEL,
    ]
    assert adapter.to_record()["metadata_prefix"] == architecture


def test_missing_sliding_window_fails_before_surgery_view(tmp_path: Path) -> None:
    path = tmp_path / "mistral.gguf"
    _write(path, "mistral", include_window=False)
    with (
        open_gguf(path) as source,
        pytest.raises(MistralGGUFSurgeryError, match="sliding_window"),
    ):
        load_mistral_gguf_surgery_adapter(source.container)


def test_mistral_version_and_gqa_constraints_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "mistral.gguf"
    _write(path, "mistral")
    with open_gguf(path) as source:
        discovery = discover_gguf_components(source.container, family=ModelFamily.MISTRAL)
    with pytest.raises(MistralGGUFSurgeryError, match="contract version 2"):
        build_mistral_gguf_surgery_adapter(
            replace(discovery, contract_version=2), sliding_window=4096
        )
    with pytest.raises(MistralGGUFSurgeryError, match="divisible"):
        build_mistral_gguf_surgery_adapter(
            replace(discovery, shape=replace(discovery.shape, kv_heads=3)),
            sliding_window=4096,
        )

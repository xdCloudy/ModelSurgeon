"""Tests for logical attention-head output masking."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.head_mask import (
    AttentionHeadMask,
    AttentionHeadMaskError,
    AttentionHeadMaskPoint,
    AttentionHeadMaskSpec,
)
from modelsurgeon.surgery.transaction import InMemoryMutationTransaction


class SnapshotTarget:
    def snapshot(self) -> object:
        return b"stable"

    def restore(self, snapshot: object) -> None:
        assert snapshot == b"stable"


@dataclass
class Handle:
    module: Module
    hook: object

    def remove(self) -> None:
        self.module.hooks.remove(self.hook)


class Module:
    def __init__(self) -> None:
        self.hooks: list[object] = []

    def register_forward_hook(self, hook: object) -> Handle:
        self.hooks.append(hook)
        return Handle(self, hook)

    def emit(self, output: object) -> object:
        result = output
        for hook in tuple(self.hooks):
            result = hook(self, (), result)  # type: ignore[operator]
        return result


def _transaction(owner: object) -> InMemoryMutationTransaction:
    return InMemoryMutationTransaction(owner, {"hook": SnapshotTarget()}, ("hook",))


def _component() -> ComponentId:
    return ComponentId.parse("model.layers.0.self_attn")


def test_only_selected_query_head_contribution_is_zeroed_and_reversible() -> None:
    module = Module()
    source = ((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),)
    spec = AttentionHeadMaskSpec(_component(), 1, 4, 4, 2)

    with _transaction(object()) as transaction, AttentionHeadMask(
        spec, module, transaction
    ) as mask:
        assert mask.active
        assert module.emit(source) == ((1.0, 2.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0),)

    assert module.emit(source) == source
    assert module.hooks == []


@pytest.mark.parametrize("kv_heads", [2, 1])
def test_gqa_and_mqa_query_output_masking_remains_isolatable(kv_heads: int) -> None:
    spec = AttentionHeadMaskSpec(_component(), 3, 4, kv_heads, 2)
    assert spec.mask_indices == (6, 7)
    assert spec.expected_output_width == 8


@pytest.mark.parametrize("kv_heads, mode", [(2, "GQA"), (1, "MQA")])
def test_shared_kv_masking_rejects_gqa_and_mqa(kv_heads: int, mode: str) -> None:
    with pytest.raises(AttentionHeadMaskError, match=mode):
        AttentionHeadMaskSpec(
            _component(),
            0,
            4,
            kv_heads,
            2,
            AttentionHeadMaskPoint.SHARED_KV_OUTPUT,
        )


def test_mha_shared_kv_point_maps_one_head_directly() -> None:
    spec = AttentionHeadMaskSpec(
        _component(),
        2,
        4,
        4,
        3,
        AttentionHeadMaskPoint.SHARED_KV_OUTPUT,
    )
    assert spec.mask_indices == (6, 7, 8)
    assert spec.expected_output_width == 12


def test_invalid_head_geometry_fails_closed() -> None:
    with pytest.raises(AttentionHeadMaskError, match="divisible"):
        AttentionHeadMaskSpec(_component(), 0, 3, 2, 4)
    with pytest.raises(AttentionHeadMaskError, match="outside"):
        AttentionHeadMaskSpec(_component(), 4, 4, 2, 4)
    with pytest.raises(AttentionHeadMaskError, match="positive"):
        AttentionHeadMaskSpec(_component(), 0, 4, 2, 0)

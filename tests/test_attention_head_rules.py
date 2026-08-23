"""Tests for fail-closed native GGUF MHA/GQA/MQA removal rules."""

from __future__ import annotations

from functools import reduce
from operator import mul

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    GGMLQuantizationType,
    GGUFDiscovery,
    GGUFModelShape,
    GGUFTensorComponent,
    GGUFTensorDescriptor,
    plan_axis_edit,
    resolve_gguf_architecture,
)
from modelsurgeon.graph import ComponentGraph, ComponentId, GraphNode
from modelsurgeon.surgery import (
    AttentionHeadEditStrategy,
    AttentionHeadMode,
    NativeGGUFAttentionHeadRuleError,
    resolve_native_gguf_attention_head_removal_rules,
)


def _discovery(
    family: ModelFamily,
    *,
    query_heads: int,
    kv_heads: int,
) -> GGUFDiscovery:
    architecture_name = "llama" if family is ModelFamily.LLAMA else "qwen2"
    architecture = resolve_gguf_architecture(architecture_name, family=family)
    embedding = 1024
    head_width = embedding // query_heads
    shapes = (
        ("token_embd.weight", (embedding, 256)),
        ("output_norm.weight", (embedding,)),
        ("blk.0.attn_norm.weight", (embedding,)),
        ("blk.0.ffn_norm.weight", (embedding,)),
        ("blk.0.attn_q.weight", (embedding, query_heads * head_width)),
        ("blk.0.attn_k.weight", (embedding, kv_heads * head_width)),
        ("blk.0.attn_v.weight", (embedding, kv_heads * head_width)),
        ("blk.0.attn_output.weight", (query_heads * head_width, embedding)),
        ("blk.0.ffn_gate.weight", (embedding, 512)),
        ("blk.0.ffn_up.weight", (embedding, 512)),
        ("blk.0.ffn_down.weight", (512, embedding)),
    )
    tensors: list[GGUFTensorComponent] = []
    offset = 0
    for ordinal, (name, shape) in enumerate(shapes):
        quant_type = GGMLQuantizationType.Q4_K
        size = plan_axis_edit(quant_type, shape, 0).tensor_bytes
        descriptor = GGUFTensorDescriptor(
            name,
            shape,
            12,
            quant_type,
            offset,
            offset,
            size,
            ordinal,
            1,
        )
        mapping = architecture.map_tensor(name)
        tensors.append(
            GGUFTensorComponent(
                descriptor,
                mapping,
                mapping.component_id.child("weight"),
                reduce(mul, shape, 1),
            )
        )
        offset += size
    graph = ComponentGraph.build((GraphNode(ComponentId.parse("model"), "model"),))
    return GGUFDiscovery(
        family,
        architecture_name,
        1,
        GGUFModelShape(1, embedding, 512, query_heads, kv_heads),
        tuple(tensors),
        sum(item.element_count for item in tensors),
        sum(item.descriptor.byte_size for item in tensors),
        graph,
    )


@pytest.mark.parametrize("family", [ModelFamily.LLAMA, ModelFamily.QWEN])
def test_gqa_removes_equal_local_query_pattern_on_llama_and_qwen(
    family: ModelFamily,
) -> None:
    rules = resolve_native_gguf_attention_head_removal_rules(
        _discovery(family, query_heads=8, kv_heads=2),
        removed_query_heads=(1, 5),
    )

    assert rules.mode is AttentionHeadMode.GQA
    assert (rules.new_query_heads, rules.new_kv_heads) == (6, 2)
    assert rules.removed_kv_heads == ()
    by_role = {item.role.value: item for item in rules.tensor_rules}
    assert by_role["attention_q"].new_shape == (1024, 768)
    assert by_role["attention_k"].changed is False
    assert by_role["attention_v"].changed is False
    assert by_role["attention_o"].new_shape == (768, 1024)
    assert by_role["attention_o"].strategy is (
        AttentionHeadEditStrategy.REPACK_CONTIGUOUS_AXIS
    )
    prefix = "llama" if family is ModelFamily.LLAMA else "qwen2"
    assert dict(rules.metadata_updates) == {
        f"{prefix}.attention.head_count": 6,
        f"{prefix}.attention.head_count_kv": 2,
        f"{prefix}.attention.key_length": 128,
        f"{prefix}.attention.value_length": 128,
    }


def test_mha_removes_matching_qkvo_heads_and_preserves_width() -> None:
    rules = resolve_native_gguf_attention_head_removal_rules(
        _discovery(ModelFamily.LLAMA, query_heads=4, kv_heads=4),
        removed_query_heads=(1, 3),
    )

    assert rules.mode is AttentionHeadMode.MHA
    assert rules.removed_kv_heads == (1, 3)
    assert (rules.new_query_heads, rules.new_kv_heads) == (2, 2)
    assert all(item.changed for item in rules.tensor_rules)
    assert {item.head_width for item in rules.tensor_rules} == {256}
    assert next(item for item in rules.tensor_rules if item.axis == 0).strategy is (
        AttentionHeadEditStrategy.DIRECT_BLOCK_COPY
    )


def test_mqa_can_remove_aligned_query_heads_but_never_its_only_kv_head() -> None:
    discovery = _discovery(ModelFamily.QWEN, query_heads=8, kv_heads=1)
    rules = resolve_native_gguf_attention_head_removal_rules(
        discovery,
        removed_query_heads=(0, 1),
    )
    assert rules.mode is AttentionHeadMode.MQA
    assert (rules.new_query_heads, rules.new_kv_heads) == (6, 1)
    assert rules.removed_kv_heads == ()

    with pytest.raises(NativeGGUFAttentionHeadRuleError, match="retain a head"):
        resolve_native_gguf_attention_head_removal_rules(
            discovery,
            removed_query_heads=tuple(range(8)),
        )


def test_uneven_gqa_patterns_and_unaligned_output_width_fail_before_payloads() -> None:
    discovery = _discovery(ModelFamily.LLAMA, query_heads=8, kv_heads=2)
    with pytest.raises(NativeGGUFAttentionHeadRuleError, match="same local"):
        resolve_native_gguf_attention_head_removal_rules(
            discovery,
            removed_query_heads=(1,),
        )
    mqa = _discovery(ModelFamily.LLAMA, query_heads=8, kv_heads=1)
    with pytest.raises(NativeGGUFAttentionHeadRuleError, match="codec-representable"):
        resolve_native_gguf_attention_head_removal_rules(
            mqa,
            removed_query_heads=(0,),
        )

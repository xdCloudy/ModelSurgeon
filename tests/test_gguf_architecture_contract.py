"""Tests for versioned GGUF architecture and tensor mutation mapping."""

from __future__ import annotations

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    AmbiguousGGUFArchitectureError,
    AxisSemantic,
    CouplingKind,
    GGUFArchitectureError,
    MetadataSemantic,
    TensorRole,
    UnknownGGUFArchitectureError,
    UnknownTensorMappingError,
    resolve_gguf_architecture,
)


@pytest.mark.parametrize(
    ("architecture", "family"),
    [
        ("llama", ModelFamily.LLAMA),
        ("mistral", ModelFamily.MISTRAL),
        ("qwen2", ModelFamily.QWEN),
        ("qwen2moe", ModelFamily.QWEN),
        ("qwen3", ModelFamily.QWEN),
        ("qwen3_moe", ModelFamily.QWEN),
        ("gemma", ModelFamily.GEMMA),
        ("gemma2", ModelFamily.GEMMA),
        ("gemma3", ModelFamily.GEMMA),
    ],
)
def test_known_architecture_aliases_resolve_explicitly(
    architecture: str,
    family: ModelFamily,
) -> None:
    resolved = resolve_gguf_architecture(architecture, family=family)

    assert resolved.contract.family is family
    assert resolved.contract.version == 1


def test_unknown_and_ambiguous_architectures_fail_closed() -> None:
    with pytest.raises(UnknownGGUFArchitectureError, match="future-model"):
        resolve_gguf_architecture("future-model")
    with pytest.raises(AmbiguousGGUFArchitectureError, match="explicit family"):
        resolve_gguf_architecture("llama")
    assert (
        resolve_gguf_architecture("llama", family=ModelFamily.MISTRAL).contract.family
        is ModelFamily.MISTRAL
    )


@pytest.mark.parametrize(
    ("tensor_name", "component", "role", "axes"),
    [
        (
            "blk.7.attn_q.weight",
            "model.layers.7.self_attn.q_proj",
            TensorRole.ATTENTION_Q,
            (AxisSemantic.INPUT_FEATURE, AxisSemantic.ATTENTION_HEAD),
        ),
        (
            "blk.7.attn_k.weight",
            "model.layers.7.self_attn.k_proj",
            TensorRole.ATTENTION_K,
            (AxisSemantic.INPUT_FEATURE, AxisSemantic.KV_HEAD),
        ),
        (
            "blk.7.attn_output.weight",
            "model.layers.7.self_attn.o_proj",
            TensorRole.ATTENTION_O,
            (AxisSemantic.ATTENTION_HEAD, AxisSemantic.OUTPUT_FEATURE),
        ),
        (
            "blk.7.ffn_gate.weight",
            "model.layers.7.mlp.gate_proj",
            TensorRole.MLP_GATE,
            (AxisSemantic.INPUT_FEATURE, AxisSemantic.MLP_CHANNEL),
        ),
        (
            "blk.7.ffn_down.weight",
            "model.layers.7.mlp.down_proj",
            TensorRole.MLP_DOWN,
            (AxisSemantic.MLP_CHANNEL, AxisSemantic.OUTPUT_FEATURE),
        ),
        (
            "blk.7.attn_k.bias",
            "model.layers.7.self_attn.k_proj",
            TensorRole.ATTENTION_K,
            (AxisSemantic.KV_HEAD,),
        ),
    ],
)
def test_tensor_names_map_to_canonical_components_and_axis_semantics(
    tensor_name: str,
    component: str,
    role: TensorRole,
    axes: tuple[AxisSemantic, ...],
) -> None:
    contract = resolve_gguf_architecture("qwen3", family=ModelFamily.QWEN)

    mapping = contract.map_tensor(tensor_name)

    assert str(mapping.component_id) == component
    assert mapping.role is role
    assert tuple(axis.semantic for axis in mapping.axes) == axes


def test_coupling_groups_cover_attention_kv_and_mlp_axes() -> None:
    resolved = resolve_gguf_architecture("gemma2", family=ModelFamily.GEMMA)

    groups = resolved.coupling_groups(3)

    assert [group.kind for group in groups] == [
        CouplingKind.ATTENTION_HEAD,
        CouplingKind.KV_HEAD,
        CouplingKind.MLP_CHANNEL,
    ]
    assert [(target.role, target.axis) for target in groups[2].targets] == [
        (TensorRole.MLP_GATE, 1),
        (TensorRole.MLP_UP, 1),
        (TensorRole.MLP_DOWN, 0),
    ]


def test_metadata_updates_and_block_renames_are_explicit() -> None:
    resolved = resolve_gguf_architecture("qwen3", family=ModelFamily.QWEN)

    updates = resolved.metadata_updates(
        (
            (MetadataSemantic.BLOCK_COUNT, 31),
            (MetadataSemantic.FEED_FORWARD_LENGTH, 8192),
        )
    )

    assert updates == (("qwen3.block_count", 31), ("qwen3.feed_forward_length", 8192))
    remap = ((0, 0), (1, None), (2, 1))
    assert resolved.rename_tensor_blocks("blk.2.attn_q.weight", remap) == "blk.1.attn_q.weight"
    assert resolved.rename_tensor_blocks("blk.1.attn_q.weight", remap) is None
    assert resolved.rename_tensor_blocks("token_embd.weight", remap) == "token_embd.weight"


def test_unknown_tensors_and_incomplete_remaps_fail_before_mutation() -> None:
    resolved = resolve_gguf_architecture("llama", family=ModelFamily.LLAMA)

    with pytest.raises(UnknownTensorMappingError, match="mystery"):
        resolved.map_tensor("blk.0.mystery.weight")
    with pytest.raises(GGUFArchitectureError, match="no disposition"):
        resolved.rename_tensor_blocks("blk.4.attn_q.weight", ((0, 0),))
    with pytest.raises(GGUFArchitectureError, match="positive"):
        resolved.metadata_updates(((MetadataSemantic.BLOCK_COUNT, 0),))

"""Tests for coupled native Llama/Qwen GGUF MLP-channel planning."""

from __future__ import annotations

import json
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
    NativeGGUFMLPPlanError,
    QuantizedEditStrategy,
    plan_native_gguf_mlp_channel_removal,
    plan_native_gguf_model_mlp_channel_removal,
)


def _discovery(family: ModelFamily, *, layers: int = 1) -> GGUFDiscovery:
    architecture_name = "llama" if family is ModelFamily.LLAMA else "qwen2"
    architecture = resolve_gguf_architecture(architecture_name, family=family)
    shapes = [
        ("token_embd.weight", (256, 1024), GGMLQuantizationType.F32),
        ("output_norm.weight", (256,), GGMLQuantizationType.F32),
    ]
    for layer in range(layers):
        shapes.extend(
            (
                (f"blk.{layer}.attn_norm.weight", (256,), GGMLQuantizationType.F32),
                (f"blk.{layer}.ffn_norm.weight", (256,), GGMLQuantizationType.F32),
                (f"blk.{layer}.attn_q.weight", (256, 256), GGMLQuantizationType.F32),
                (f"blk.{layer}.attn_k.weight", (256, 128), GGMLQuantizationType.F32),
                (f"blk.{layer}.attn_v.weight", (256, 128), GGMLQuantizationType.F32),
                (f"blk.{layer}.attn_output.weight", (256, 256), GGMLQuantizationType.F32),
                (f"blk.{layer}.ffn_gate.weight", (256, 512), GGMLQuantizationType.Q4_K),
                (f"blk.{layer}.ffn_up.weight", (256, 512), GGMLQuantizationType.Q4_K),
                (f"blk.{layer}.ffn_down.weight", (512, 256), GGMLQuantizationType.Q4_K),
            )
        )
    tensors: list[GGUFTensorComponent] = []
    offset = 0
    for ordinal, (name, shape, quant_type) in enumerate(shapes):
        size = plan_axis_edit(quant_type, shape, 0).tensor_bytes
        descriptor = GGUFTensorDescriptor(
            name,
            shape,
            0,
            quant_type,
            offset,
            offset,
            size,
            ordinal,
            1,
        )
        mapping = architecture.map_tensor(name)
        component = mapping.component_id.child("weight")
        tensors.append(
            GGUFTensorComponent(descriptor, mapping, component, reduce(mul, shape, 1))
        )
        offset += size
    graph = ComponentGraph.build((GraphNode(ComponentId.parse("model"), "model"),))
    return GGUFDiscovery(
        family,
        architecture_name,
        1,
        GGUFModelShape(layers, 256, 512, 8, 4),
        tuple(tensors),
        sum(tensor.element_count for tensor in tensors),
        sum(tensor.descriptor.byte_size for tensor in tensors),
        graph,
    )


@pytest.mark.parametrize("family", [ModelFamily.LLAMA, ModelFamily.QWEN])
def test_selected_families_compile_only_coupled_mlp_tensors(family: ModelFamily) -> None:
    plan = plan_native_gguf_mlp_channel_removal(
        _discovery(family),
        layer_index=0,
        removed_channels=tuple(range(256)),
    )
    assert plan.coupled_tensor_names == (
        "blk.0.ffn_down.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
    )
    assert {edit.locator for edit in plan.physical_plan.tensor_edits} == set(
        plan.coupled_tensor_names
    )
    assert [edit.new_shape for edit in plan.physical_plan.tensor_edits] == [
        (256, 256),
        (256, 256),
        (256, 256),
    ]
    strategies = {
        edit.component_id: edit.axis_edits[0].strategy
        for edit in plan.quantized_plan.tensor_edits
    }
    down = next(
        edit.component_id
        for edit in plan.physical_plan.tensor_edits
        if edit.locator == "blk.0.ffn_down.weight"
    )
    assert strategies[down] is QuantizedEditStrategy.DIRECT_BLOCK_COPY
    assert list(strategies.values()).count(QuantizedEditStrategy.WHOLE_SLICE_COPY) == 2


@pytest.mark.parametrize(
    ("family", "metadata_key"),
    [
        (ModelFamily.LLAMA, "llama.feed_forward_length"),
        (ModelFamily.QWEN, "qwen2.feed_forward_length"),
    ],
)
def test_shapes_metadata_parameters_storage_and_identities_reconcile(
    family: ModelFamily, metadata_key: str
) -> None:
    plan = plan_native_gguf_mlp_channel_removal(
        _discovery(family), layer_index=0, removed_channels=tuple(range(256))
    )
    assert plan.physical_plan.metadata_updates[0].key == metadata_key
    assert plan.physical_plan.metadata_updates[0].value == 256
    assert plan.expected_parameter_delta == -(256 * 256 * 3)
    assert plan.expected_storage_delta == -(36_864 * 3)
    assert sum(edit.storage_delta for edit in plan.physical_plan.tensor_edits) == (
        plan.expected_storage_delta
    )
    removed = ComponentId.parse("model.layers.0.mlp.channel.0")
    renumbered = ComponentId.parse("model.layers.0.mlp.channel.256")
    assert plan.physical_plan.identity_remap.mapping(removed).removed is True
    assert plan.physical_plan.identity_remap.resolve(renumbered) == (
        ComponentId.parse("model.layers.0.mlp.channel.0"),
    )
    json.dumps(plan.to_record())


def test_noncanonical_and_block_unrepresentable_channel_sets_fail_before_load() -> None:
    discovery = _discovery(ModelFamily.LLAMA)
    with pytest.raises(NativeGGUFMLPPlanError, match="canonical"):
        plan_native_gguf_mlp_channel_removal(
            discovery, layer_index=0, removed_channels=(2, 1)
        )
    with pytest.raises(NativeGGUFMLPPlanError, match="block-representable"):
        plan_native_gguf_mlp_channel_removal(
            discovery, layer_index=0, removed_channels=(0,)
        )


def test_model_wide_plan_updates_every_layer_and_single_layer_fails_closed() -> None:
    discovery = _discovery(ModelFamily.LLAMA, layers=2)
    with pytest.raises(NativeGGUFMLPPlanError, match="model-wide"):
        plan_native_gguf_mlp_channel_removal(
            discovery,
            layer_index=0,
            removed_channels=tuple(range(256)),
        )

    plan = plan_native_gguf_model_mlp_channel_removal(
        discovery,
        removed_channels=tuple(range(256)),
    )

    assert plan.layer_indices == (0, 1)
    assert len(plan.coupled_tensor_names) == 6
    assert plan.expected_parameter_delta == -(2 * 256 * 256 * 3)
    assert plan.physical_plan.metadata_updates[0].value == 256

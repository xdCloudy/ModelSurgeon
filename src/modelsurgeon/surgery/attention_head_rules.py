"""Fail-closed native GGUF MHA/GQA/MQA head-removal rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    QUANT_LAYOUTS,
    GGMLQuantizationType,
    GGUFDiscovery,
    MetadataSemantic,
    TensorRole,
    build_llama_gguf_surgery_adapter,
    build_qwen_gguf_surgery_adapter,
    plan_axis_edit,
    resolve_gguf_architecture,
)


class NativeGGUFAttentionHeadRuleError(ValueError):
    """Raised when requested head removal would change or corrupt KV grouping."""


class AttentionHeadMode(StrEnum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"


class AttentionHeadEditStrategy(StrEnum):
    UNCHANGED = "unchanged"
    WHOLE_HEAD_SLICE_COPY = "whole_head_slice_copy"
    DIRECT_BLOCK_COPY = "direct_block_copy"
    REPACK_CONTIGUOUS_AXIS = "repack_contiguous_axis"


@dataclass(frozen=True, slots=True)
class AttentionHeadTensorRule:
    tensor_name: str
    role: TensorRole
    axis: int
    head_width: int
    removed_heads: tuple[int, ...]
    removed_indices: tuple[int, ...]
    old_shape: tuple[int, ...]
    new_shape: tuple[int, ...]
    quant_type: GGMLQuantizationType
    strategy: AttentionHeadEditStrategy

    @property
    def changed(self) -> bool:
        return bool(self.removed_indices)


@dataclass(frozen=True, slots=True)
class NativeGGUFAttentionHeadRules:
    family: ModelFamily
    architecture: str
    mode: AttentionHeadMode
    old_query_heads: int
    new_query_heads: int
    old_kv_heads: int
    new_kv_heads: int
    key_head_width: int
    value_head_width: int
    removed_query_heads: tuple[int, ...]
    removed_kv_heads: tuple[int, ...]
    tensor_rules: tuple[AttentionHeadTensorRule, ...]
    metadata_updates: tuple[tuple[str, int], ...]
    upstream_revision: str = "c060ca974c773c7c3d17fd1b66dc9d312bc292c0"

    def to_record(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "architecture": self.architecture,
            "mode": self.mode.value,
            "old_query_heads": self.old_query_heads,
            "new_query_heads": self.new_query_heads,
            "old_kv_heads": self.old_kv_heads,
            "new_kv_heads": self.new_kv_heads,
            "key_head_width": self.key_head_width,
            "value_head_width": self.value_head_width,
            "removed_query_heads": list(self.removed_query_heads),
            "removed_kv_heads": list(self.removed_kv_heads),
            "metadata_updates": dict(self.metadata_updates),
            "upstream_revision": self.upstream_revision,
        }


def _head_widths(discovery: GGUFDiscovery) -> tuple[int, int]:
    return discovery.shape.key_length, discovery.shape.value_length


def _mode(query_heads: int, kv_heads: int) -> AttentionHeadMode:
    if query_heads == kv_heads:
        return AttentionHeadMode.MHA
    if kv_heads == 1:
        return AttentionHeadMode.MQA
    return AttentionHeadMode.GQA


def _indices(heads: tuple[int, ...], width: int) -> tuple[int, ...]:
    return tuple(index for head in heads for index in range(head * width, (head + 1) * width))


def _strategy(
    axis: int,
    removed: tuple[int, ...],
    quant_type: GGMLQuantizationType,
) -> AttentionHeadEditStrategy:
    if not removed:
        return AttentionHeadEditStrategy.UNCHANGED
    if axis > 0:
        return AttentionHeadEditStrategy.WHOLE_HEAD_SLICE_COPY
    block = QUANT_LAYOUTS[quant_type].block_size
    selected = set(removed)
    touched = {index // block for index in removed}
    complete = all(
        all(index in selected for index in range(item * block, (item + 1) * block))
        for item in touched
    )
    return (
        AttentionHeadEditStrategy.DIRECT_BLOCK_COPY
        if complete
        else AttentionHeadEditStrategy.REPACK_CONTIGUOUS_AXIS
    )


def resolve_native_gguf_attention_head_removal_rules(
    discovery: GGUFDiscovery,
    *,
    removed_query_heads: tuple[int, ...],
) -> NativeGGUFAttentionHeadRules:
    """Resolve model-wide fixed-width Q/K/V/O edits without loading payloads."""

    if discovery.family is ModelFamily.LLAMA:
        build_llama_gguf_surgery_adapter(discovery)
    elif discovery.family is ModelFamily.QWEN:
        build_qwen_gguf_surgery_adapter(discovery)
    else:
        raise NativeGGUFAttentionHeadRuleError(
            "native attention-head rules support Llama and dense Qwen only"
        )
    query_heads = discovery.shape.attention_heads
    kv_heads = discovery.shape.kv_heads
    if (
        not removed_query_heads
        or removed_query_heads != tuple(sorted(set(removed_query_heads)))
        or removed_query_heads[0] < 0
        or removed_query_heads[-1] >= query_heads
        or len(removed_query_heads) == query_heads
    ):
        raise NativeGGUFAttentionHeadRuleError(
            "removed query heads must be non-empty, canonical, in-range, and retain a head"
        )
    if query_heads % kv_heads:
        raise NativeGGUFAttentionHeadRuleError(
            "query head count must be divisible by KV head count"
        )
    per_kv = query_heads // kv_heads
    selected = set(removed_query_heads)
    patterns = tuple(
        tuple(local for local in range(per_kv) if group * per_kv + local in selected)
        for group in range(kv_heads)
    )
    removed_kv = tuple(index for index, pattern in enumerate(patterns) if len(pattern) == per_kv)
    retained_patterns = tuple(pattern for pattern in patterns if len(pattern) != per_kv)
    if not retained_patterns:
        raise NativeGGUFAttentionHeadRuleError("head removal cannot remove every KV group")
    if len(set(retained_patterns)) != 1:
        raise NativeGGUFAttentionHeadRuleError(
            "retained KV groups must remove the same local query-head pattern"
        )
    retained_per_kv = per_kv - len(retained_patterns[0])
    new_kv = kv_heads - len(removed_kv)
    new_query = new_kv * retained_per_kv
    key_width, value_width = _head_widths(discovery)
    specs = (
        (TensorRole.ATTENTION_Q, 1, key_width, removed_query_heads),
        (TensorRole.ATTENTION_K, 1, key_width, removed_kv),
        (TensorRole.ATTENTION_V, 1, value_width, removed_kv),
        (TensorRole.ATTENTION_O, 0, value_width, removed_query_heads),
    )
    rules: list[AttentionHeadTensorRule] = []
    for layer in range(discovery.shape.layers):
        tensor_by_role = {
            tensor.mapping.role: tensor
            for tensor in discovery.tensors
            if tensor.descriptor.name.startswith(f"blk.{layer}.")
        }
        for role, axis, width, heads in specs:
            try:
                tensor = tensor_by_role[role]
            except KeyError as error:
                raise NativeGGUFAttentionHeadRuleError(
                    f"layer {layer} is missing required {role.value} weight"
                ) from error
            old_shape = tensor.descriptor.dimensions
            removed = _indices(heads, width)
            new_shape = list(old_shape)
            new_shape[axis] -= len(removed)
            try:
                plan_axis_edit(tensor.descriptor.quant_type, tuple(new_shape), 0)
            except ValueError as error:
                raise NativeGGUFAttentionHeadRuleError(
                    f"{role.value} new shape {tuple(new_shape)} is not "
                    "codec-representable"
                ) from error
            rules.append(
                AttentionHeadTensorRule(
                    tensor.descriptor.name,
                    role,
                    axis,
                    width,
                    heads,
                    removed,
                    old_shape,
                    tuple(new_shape),
                    tensor.descriptor.quant_type,
                    _strategy(axis, removed, tensor.descriptor.quant_type),
                )
            )
    architecture = resolve_gguf_architecture(
        discovery.architecture, family=discovery.family
    )
    updates = architecture.metadata_updates(
        (
            (MetadataSemantic.HEAD_COUNT, new_query),
            (MetadataSemantic.KV_HEAD_COUNT, new_kv),
        )
    )
    updates += tuple(
        sorted(
            (
                (f"{architecture.metadata_prefix}.attention.key_length", key_width),
                (f"{architecture.metadata_prefix}.attention.value_length", value_width),
            )
        )
    )
    return NativeGGUFAttentionHeadRules(
        discovery.family,
        discovery.architecture,
        _mode(query_heads, kv_heads),
        query_heads,
        new_query,
        kv_heads,
        new_kv,
        key_width,
        value_width,
        removed_query_heads,
        removed_kv,
        tuple(rules),
        tuple(sorted(updates)),
    )

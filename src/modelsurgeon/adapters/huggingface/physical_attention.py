"""Physical Hugging Face MHA/GQA attention-head removal."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Final

HF_PHYSICAL_ATTENTION_SCHEMA_VERSION: Final[int] = 1


class HuggingFacePhysicalAttentionError(ValueError):
    """Raised when attention head removal cannot preserve MHA/GQA grouping."""


@dataclass(frozen=True, slots=True)
class HuggingFaceAttentionRemovalResult:
    old_attention_heads: int
    new_attention_heads: int
    old_kv_heads: int
    new_kv_heads: int
    head_dim: int
    removed_query_heads: tuple[int, ...]
    removed_kv_heads: tuple[int, ...]
    old_parameter_count: int
    new_parameter_count: int
    schema_version: int = HF_PHYSICAL_ATTENTION_SCHEMA_VERSION

    @property
    def parameter_delta(self) -> int:
        return self.new_parameter_count - self.old_parameter_count


def remove_huggingface_attention_heads(
    model: Any, removed_query_heads: tuple[int, ...]
) -> HuggingFaceAttentionRemovalResult:
    """Remove complete query/KV groups from every adapter-defined attention layer."""

    config = getattr(model, "config", None)
    query_heads = getattr(config, "num_attention_heads", None)
    kv_heads = getattr(config, "num_key_value_heads", query_heads)
    hidden_size = getattr(config, "hidden_size", None)
    layers = getattr(config, "num_hidden_layers", None)
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (query_heads, kv_heads, hidden_size, layers)
    ):
        raise HuggingFacePhysicalAttentionError("model config has invalid attention dimensions")
    assert config is not None
    assert isinstance(query_heads, int) and isinstance(kv_heads, int)
    assert isinstance(hidden_size, int) and isinstance(layers, int)
    head_dim = getattr(config, "head_dim", hidden_size // query_heads)
    if not isinstance(head_dim, int) or head_dim <= 0 or query_heads % kv_heads != 0:
        raise HuggingFacePhysicalAttentionError("attention/KV head divisibility is invalid")
    if (
        not removed_query_heads
        or removed_query_heads != tuple(sorted(set(removed_query_heads)))
        or removed_query_heads[0] < 0
        or removed_query_heads[-1] >= query_heads
        or len(removed_query_heads) >= query_heads
    ):
        raise HuggingFacePhysicalAttentionError(
            "removed query heads must be canonical and leave heads"
        )
    group_size = query_heads // kv_heads
    removed_set = set(removed_query_heads)
    removed_kv: list[int] = []
    for kv_head in range(kv_heads):
        group = set(range(kv_head * group_size, (kv_head + 1) * group_size))
        overlap = group & removed_set
        if overlap and overlap != group:
            raise HuggingFacePhysicalAttentionError(
                "query removals must contain complete KV groups"
            )
        if overlap:
            removed_kv.append(kv_head)
    if len(removed_kv) >= kv_heads:
        raise HuggingFacePhysicalAttentionError("attention removal cannot remove every KV head")
    keep_query = _kept_scalar_indices(query_heads, head_dim, removed_set)
    keep_kv = _kept_scalar_indices(kv_heads, head_dim, set(removed_kv))
    modules = dict(model.named_modules())
    attentions: list[Any] = []
    for layer in range(layers):
        path = f"model.layers.{layer}.self_attn"
        attention = modules.get(path)
        if attention is None:
            raise HuggingFacePhysicalAttentionError(f"model has no adapter-defined module {path!r}")
        _validate_attention(layer, attention, query_heads, kv_heads, head_dim)
        attentions.append(attention)
    old_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    for attention in attentions:
        attention.q_proj = _slice_linear(attention.q_proj, keep_query, output_axis=True)
        attention.k_proj = _slice_linear(attention.k_proj, keep_kv, output_axis=True)
        attention.v_proj = _slice_linear(attention.v_proj, keep_kv, output_axis=True)
        attention.o_proj = _slice_linear(attention.o_proj, keep_query, output_axis=False)
        new_query_heads = query_heads - len(removed_query_heads)
        new_kv_heads = kv_heads - len(removed_kv)
        for name, value in (
            ("num_heads", new_query_heads),
            ("num_attention_heads", new_query_heads),
            ("num_key_value_heads", new_kv_heads),
            ("num_key_value_groups", new_query_heads // new_kv_heads),
        ):
            if hasattr(attention, name):
                setattr(attention, name, value)
    config.num_attention_heads = query_heads - len(removed_query_heads)
    config.num_key_value_heads = kv_heads - len(removed_kv)
    new_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    return HuggingFaceAttentionRemovalResult(
        query_heads,
        config.num_attention_heads,
        kv_heads,
        config.num_key_value_heads,
        head_dim,
        removed_query_heads,
        tuple(removed_kv),
        old_parameters,
        new_parameters,
    )


def _kept_scalar_indices(heads: int, head_dim: int, removed: set[int]) -> list[int]:
    return [
        head * head_dim + offset
        for head in range(heads)
        if head not in removed
        for offset in range(head_dim)
    ]


def _validate_attention(
    layer: int, attention: Any, query_heads: int, kv_heads: int, head_dim: int
) -> None:
    expectations = (
        ("q_proj", query_heads * head_dim, 0),
        ("k_proj", kv_heads * head_dim, 0),
        ("v_proj", kv_heads * head_dim, 0),
        ("o_proj", query_heads * head_dim, 1),
    )
    for name, size, axis in expectations:
        module = getattr(attention, name, None)
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", None) != 2 or int(weight.shape[axis]) != size:
            raise HuggingFacePhysicalAttentionError(
                f"layer {layer} {name} shape disagrees with attention metadata"
            )


def _slice_linear(module: Any, keep: list[int], *, output_axis: bool) -> Any:
    torch = __import__("torch")
    result = copy.deepcopy(module)
    sliced = module.weight.detach()[keep, :] if output_axis else module.weight.detach()[:, keep]
    result.weight = torch.nn.Parameter(sliced.clone(), requires_grad=module.weight.requires_grad)
    if output_axis and getattr(module, "bias", None) is not None:
        result.bias = torch.nn.Parameter(
            module.bias.detach()[keep].clone(), requires_grad=module.bias.requires_grad
        )
    if output_axis and hasattr(result, "out_features"):
        result.out_features = len(keep)
    if not output_axis and hasattr(result, "in_features"):
        result.in_features = len(keep)
    return result

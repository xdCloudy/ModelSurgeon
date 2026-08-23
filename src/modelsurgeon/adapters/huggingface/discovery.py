"""Deterministic, metadata-only Hugging Face component discovery."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from modelsurgeon.adapters import ComponentDescriptor, ModelFamily
from modelsurgeon.graph import ComponentId


class HuggingFaceComponentKind(StrEnum):
    MODEL = "model"
    MODULE = "module"
    TRANSFORMER_LAYER = "transformer_layer"
    ATTENTION = "attention"
    ATTENTION_HEAD = "attention_head"
    KV_HEAD = "kv_head"
    PROJECTION = "projection"
    MLP = "mlp"
    MLP_CHANNEL = "mlp_channel"
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    MOE_EXPERT = "moe_expert"
    MOE_ROUTER = "moe_router"
    PARAMETER = "parameter"


class ParameterReconciliationError(ValueError):
    """Raised when discovered parameter counts do not cover the model exactly."""


class ParameterLike(Protocol):
    def numel(self) -> int: ...


class HuggingFaceModelProvider(Protocol):
    config: object

    def named_modules(self) -> Iterable[tuple[str, object]]: ...

    def named_parameters(self) -> Iterable[tuple[str, ParameterLike]]: ...


@dataclass(frozen=True, slots=True)
class TransformerShape:
    layers: int
    attention_heads: int
    kv_heads: int
    intermediate_size: int

    @classmethod
    def from_config(cls, config: object) -> TransformerShape:
        attention_heads = _positive_config_int(config, "num_attention_heads")
        kv_heads = getattr(config, "num_key_value_heads", attention_heads)
        if not isinstance(kv_heads, int) or isinstance(kv_heads, bool) or kv_heads <= 0:
            raise ValueError("num_key_value_heads must be a positive integer")
        return cls(
            layers=_positive_config_int(config, "num_hidden_layers"),
            attention_heads=attention_heads,
            kv_heads=kv_heads,
            intermediate_size=_positive_config_int(config, "intermediate_size"),
        )


@dataclass(frozen=True, slots=True)
class HuggingFaceDiscovery:
    """Bounded physical metadata plus lazily generated logical components."""

    family: ModelFamily
    shape: TransformerShape
    modules: tuple[ComponentDescriptor, ...]
    parameters: tuple[ComponentDescriptor, ...]
    parameter_count: int

    @property
    def logical_component_count(self) -> int:
        per_layer = (
            3 + self.shape.attention_heads + self.shape.kv_heads + self.shape.intermediate_size
        )
        return self.shape.layers * per_layer

    def components(self) -> Iterator[ComponentDescriptor]:
        yield from self.modules
        yield from self.parameters
        yield from _logical_components(self.shape)

    def to_record(self) -> dict[str, str | int]:
        return {
            "family": self.family.value,
            "module_count": len(self.modules),
            "parameter_tensor_count": len(self.parameters),
            "parameter_count": self.parameter_count,
            "logical_component_count": self.logical_component_count,
        }


def _positive_config_int(config: object, name: str) -> int:
    value = getattr(config, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parameter_size(parameter: ParameterLike, name: str) -> int:
    try:
        size = int(parameter.numel())
    except (AttributeError, TypeError, ValueError) as exc:
        raise ParameterReconciliationError(
            f"parameter {name!r} does not expose a valid element count"
        ) from exc
    if size < 0:
        raise ParameterReconciliationError(
            f"parameter {name!r} has a negative element count"
        )
    return size


def _kind_for_module(name: str) -> HuggingFaceComponentKind:
    if not name:
        return HuggingFaceComponentKind.MODEL
    segments = name.split(".")
    leaf = segments[-1]
    if len(segments) >= 2 and segments[-2] == "layers" and leaf.isdigit():
        return HuggingFaceComponentKind.TRANSFORMER_LAYER
    if leaf in {"self_attn", "attention", "attn"}:
        return HuggingFaceComponentKind.ATTENTION
    if leaf == "mlp":
        return HuggingFaceComponentKind.MLP
    if leaf in {"embed_tokens", "embed_positions", "lm_head"}:
        return HuggingFaceComponentKind.EMBEDDING
    if leaf.endswith("_proj"):
        return HuggingFaceComponentKind.PROJECTION
    if leaf == "norm" or leaf.endswith("_norm"):
        return HuggingFaceComponentKind.NORMALIZATION
    if len(segments) >= 2 and segments[-2] == "experts" and leaf.isdigit():
        return HuggingFaceComponentKind.MOE_EXPERT
    if leaf in {"router", "gate"}:
        return HuggingFaceComponentKind.MOE_ROUTER
    return HuggingFaceComponentKind.MODULE


def _module_descriptors(model: HuggingFaceModelProvider) -> tuple[ComponentDescriptor, ...]:
    descriptors: dict[ComponentId, ComponentDescriptor] = {}
    for name, module in model.named_modules():
        component_id = ComponentId.from_module_name(name)
        if component_id in descriptors:
            if name == "model":
                continue
            raise ValueError(f"module name {name!r} collides at canonical ID {component_id}")
        descriptors[component_id] = ComponentDescriptor(
            component_id=component_id,
            kind=_kind_for_module(name).value,
            parent=component_id.parent,
            attributes=(("module_type", type(module).__name__),),
        )
    return tuple(descriptors[key] for key in sorted(descriptors))


def _parameter_descriptors(
    model: HuggingFaceModelProvider,
) -> tuple[tuple[ComponentDescriptor, ...], int]:
    descriptors: dict[ComponentId, ComponentDescriptor] = {}
    seen_parameters: set[int] = set()
    parameter_count = 0
    for name, parameter in model.named_parameters():
        identity = id(parameter)
        if identity in seen_parameters:
            continue
        seen_parameters.add(identity)
        size = _parameter_size(parameter, name)
        component_id = ComponentId.from_module_name(name)
        if component_id in descriptors:
            raise ParameterReconciliationError(
                f"parameter name {name!r} collides at canonical ID {component_id}"
            )
        descriptors[component_id] = ComponentDescriptor(
            component_id=component_id,
            kind=HuggingFaceComponentKind.PARAMETER.value,
            parent=component_id.parent,
            attributes=(("element_count", size),),
        )
        parameter_count += size
    ordered = tuple(descriptors[key] for key in sorted(descriptors))
    discovered_count = 0
    for item in ordered:
        element_count = dict(item.attributes).get("element_count")
        if not isinstance(element_count, int) or isinstance(element_count, bool):
            raise ParameterReconciliationError("parameter element count is not an integer")
        discovered_count += element_count
    if discovered_count != parameter_count:
        raise ParameterReconciliationError("discovered parameter counts do not reconcile")
    return ordered, parameter_count


def _logical_components(shape: TransformerShape) -> Iterator[ComponentDescriptor]:
    for layer_index in range(shape.layers):
        attention = ComponentId.parse(f"model.layers.{layer_index}.self_attn")
        head_group = attention.child("head")
        yield ComponentDescriptor(
            component_id=head_group,
            kind=HuggingFaceComponentKind.MODULE.value,
            parent=head_group.parent,
            attributes=(("layer_index", layer_index),),
        )
        for head_index in range(shape.attention_heads):
            component_id = head_group.child(head_index)
            yield ComponentDescriptor(
                component_id=component_id,
                kind=HuggingFaceComponentKind.ATTENTION_HEAD.value,
                parent=component_id.parent,
                attributes=(("layer_index", layer_index), ("head_index", head_index)),
            )
        kv_head_group = attention.child("kv_head")
        yield ComponentDescriptor(
            component_id=kv_head_group,
            kind=HuggingFaceComponentKind.MODULE.value,
            parent=kv_head_group.parent,
            attributes=(("layer_index", layer_index),),
        )
        for head_index in range(shape.kv_heads):
            component_id = kv_head_group.child(head_index)
            yield ComponentDescriptor(
                component_id=component_id,
                kind=HuggingFaceComponentKind.KV_HEAD.value,
                parent=component_id.parent,
                attributes=(("layer_index", layer_index), ("head_index", head_index)),
            )
        mlp = ComponentId.parse(f"model.layers.{layer_index}.mlp")
        channel_group = mlp.child("channel")
        yield ComponentDescriptor(
            component_id=channel_group,
            kind=HuggingFaceComponentKind.MODULE.value,
            parent=channel_group.parent,
            attributes=(("layer_index", layer_index),),
        )
        for channel_index in range(shape.intermediate_size):
            component_id = channel_group.child(channel_index)
            yield ComponentDescriptor(
                component_id=component_id,
                kind=HuggingFaceComponentKind.MLP_CHANNEL.value,
                parent=component_id.parent,
                attributes=(("layer_index", layer_index), ("channel_index", channel_index)),
            )


def discover_huggingface_components(
    model: HuggingFaceModelProvider,
    family: ModelFamily,
) -> HuggingFaceDiscovery:
    """Discover deterministic physical and logical component metadata."""
    modules = _module_descriptors(model)
    parameters, parameter_count = _parameter_descriptors(model)
    return HuggingFaceDiscovery(
        family=family,
        shape=TransformerShape.from_config(model.config),
        modules=modules,
        parameters=parameters,
        parameter_count=parameter_count,
    )

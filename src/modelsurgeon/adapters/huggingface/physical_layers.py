"""Physical Hugging Face transformer-layer removal and canonical renumbering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

HF_PHYSICAL_LAYER_SCHEMA_VERSION: Final[int] = 1


class HuggingFacePhysicalLayerError(ValueError):
    """Raised when a model does not expose a safe standard layer container."""


@dataclass(frozen=True, slots=True)
class HuggingFaceLayerRemovalResult:
    old_layer_count: int
    new_layer_count: int
    removed_indices: tuple[int, ...]
    old_to_new: tuple[tuple[int, int | None], ...]
    old_parameter_count: int
    new_parameter_count: int
    schema_version: int = HF_PHYSICAL_LAYER_SCHEMA_VERSION

    @property
    def parameter_delta(self) -> int:
        return self.new_parameter_count - self.old_parameter_count


def remove_huggingface_transformer_layers(
    model: Any, removed_indices: tuple[int, ...]
) -> HuggingFaceLayerRemovalResult:
    """Remove full blocks while preserving every retained module object and parameter."""

    torch = __import__("torch")
    config = getattr(model, "config", None)
    configured = getattr(config, "num_hidden_layers", None)
    layer_types = getattr(config, "layer_types", None)
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if (
        not isinstance(configured, int)
        or isinstance(configured, bool)
        or configured <= 0
        or not isinstance(layers, torch.nn.ModuleList)
        or len(layers) != configured
    ):
        raise HuggingFacePhysicalLayerError(
            "model must expose a ModuleList matching config.num_hidden_layers"
        )
    if layer_types is not None and (
        not isinstance(layer_types, (list, tuple)) or len(layer_types) != configured
    ):
        raise HuggingFacePhysicalLayerError(
            "model config layer_types must match config.num_hidden_layers"
        )
    if (
        not removed_indices
        or removed_indices != tuple(sorted(set(removed_indices)))
        or removed_indices[0] < 0
        or removed_indices[-1] >= configured
        or len(removed_indices) >= configured
    ):
        raise HuggingFacePhysicalLayerError("removed layers must be canonical and leave a layer")
    assert config is not None and base is not None
    old_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    removed = set(removed_indices)
    retained = [layer for index, layer in enumerate(layers) if index not in removed]
    mapping: list[tuple[int, int | None]] = []
    new_index = 0
    for old_index in range(configured):
        if old_index in removed:
            mapping.append((old_index, None))
        else:
            mapping.append((old_index, new_index))
            new_index += 1
    base.layers = torch.nn.ModuleList(retained)
    for index, layer in enumerate(retained):
        attention = getattr(layer, "self_attn", None)
        if attention is not None and hasattr(attention, "layer_idx"):
            attention.layer_idx = index
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = index
    config.num_hidden_layers = len(retained)
    if layer_types is not None:
        retained_layer_types = [
            layer_types[index] for index in range(configured) if index not in removed
        ]
        config.layer_types = (
            tuple(retained_layer_types) if isinstance(layer_types, tuple) else retained_layer_types
        )
    new_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    return HuggingFaceLayerRemovalResult(
        configured,
        len(retained),
        removed_indices,
        tuple(mapping),
        old_parameters,
        new_parameters,
    )

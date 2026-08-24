"""Physical Hugging Face gated-MLP channel removal."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Final

HF_PHYSICAL_MLP_SCHEMA_VERSION: Final[int] = 1


class HuggingFacePhysicalMLPError(ValueError):
    """Raised before or during an incompatible physical MLP resize."""


@dataclass(frozen=True, slots=True)
class HuggingFaceMLPLayerResize:
    layer_index: int
    gate_shape: tuple[int, int]
    up_shape: tuple[int, int]
    down_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class HuggingFaceMLPRemovalResult:
    old_intermediate_size: int
    new_intermediate_size: int
    removed_indices: tuple[int, ...]
    old_parameter_count: int
    new_parameter_count: int
    layers: tuple[HuggingFaceMLPLayerResize, ...]
    schema_version: int = HF_PHYSICAL_MLP_SCHEMA_VERSION

    @property
    def parameter_delta(self) -> int:
        return self.new_parameter_count - self.old_parameter_count

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "old_intermediate_size": self.old_intermediate_size,
            "new_intermediate_size": self.new_intermediate_size,
            "removed_indices": list(self.removed_indices),
            "old_parameter_count": self.old_parameter_count,
            "new_parameter_count": self.new_parameter_count,
            "parameter_delta": self.parameter_delta,
            "layers": [
                {
                    "layer_index": item.layer_index,
                    "gate_shape": list(item.gate_shape),
                    "up_shape": list(item.up_shape),
                    "down_shape": list(item.down_shape),
                }
                for item in self.layers
            ],
        }


def remove_huggingface_mlp_channels(
    model: Any, removed_indices: tuple[int, ...]
) -> HuggingFaceMLPRemovalResult:
    """Resize every gate/up/down MLP consistently and update global configuration."""

    config = getattr(model, "config", None)
    old_width = getattr(config, "intermediate_size", None)
    layers = getattr(config, "num_hidden_layers", None)
    if not isinstance(old_width, int) or isinstance(old_width, bool) or old_width <= 0:
        raise HuggingFacePhysicalMLPError("model config has no valid intermediate_size")
    if not isinstance(layers, int) or isinstance(layers, bool) or layers <= 0:
        raise HuggingFacePhysicalMLPError("model config has no valid num_hidden_layers")
    assert config is not None
    if (
        not removed_indices
        or removed_indices != tuple(sorted(set(removed_indices)))
        or removed_indices[0] < 0
        or removed_indices[-1] >= old_width
        or len(removed_indices) >= old_width
    ):
        raise HuggingFacePhysicalMLPError("removed MLP channels must be canonical and leave width")
    modules = dict(model.named_modules())
    projections: list[tuple[int, Any, Any, Any, Any]] = []
    for layer in range(layers):
        path = f"model.layers.{layer}.mlp"
        mlp = modules.get(path)
        if mlp is None:
            raise HuggingFacePhysicalMLPError(f"model has no adapter-defined module {path!r}")
        gate = getattr(mlp, "gate_proj", None)
        up = getattr(mlp, "up_proj", None)
        down = getattr(mlp, "down_proj", None)
        _validate_projection_shapes(layer, gate, up, down, old_width)
        projections.append((layer, mlp, gate, up, down))
    old_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    removed = set(removed_indices)
    keep = [index for index in range(old_width) if index not in removed]
    resized: list[HuggingFaceMLPLayerResize] = []
    for layer, mlp, gate, up, down in projections:
        new_gate = _slice_projection(gate, keep, output_axis=True)
        new_up = _slice_projection(up, keep, output_axis=True)
        new_down = _slice_projection(down, keep, output_axis=False)
        mlp.gate_proj = new_gate
        mlp.up_proj = new_up
        mlp.down_proj = new_down
        resized.append(
            HuggingFaceMLPLayerResize(
                layer,
                (int(new_gate.weight.shape[0]), int(new_gate.weight.shape[1])),
                (int(new_up.weight.shape[0]), int(new_up.weight.shape[1])),
                (int(new_down.weight.shape[0]), int(new_down.weight.shape[1])),
            )
        )
    config.intermediate_size = len(keep)
    new_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
    return HuggingFaceMLPRemovalResult(
        old_width,
        len(keep),
        removed_indices,
        old_parameters,
        new_parameters,
        tuple(resized),
    )


def _validate_projection_shapes(layer: int, gate: Any, up: Any, down: Any, width: int) -> None:
    for name, module in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", None) != 2:
            raise HuggingFacePhysicalMLPError(f"layer {layer} {name} has no rank-2 weight")
    if int(gate.weight.shape[0]) != width or int(up.weight.shape[0]) != width:
        raise HuggingFacePhysicalMLPError(f"layer {layer} gate/up width disagrees with config")
    if int(down.weight.shape[1]) != width:
        raise HuggingFacePhysicalMLPError(f"layer {layer} down width disagrees with config")


def _slice_projection(module: Any, keep: list[int], *, output_axis: bool) -> Any:
    torch = __import__("torch")
    result = copy.deepcopy(module)
    weight = module.weight.detach()
    sliced = weight[keep, :] if output_axis else weight[:, keep]
    result.weight = torch.nn.Parameter(sliced.clone(), requires_grad=module.weight.requires_grad)
    if output_axis and getattr(module, "bias", None) is not None:
        result.bias = torch.nn.Parameter(
            module.bias.detach()[keep].clone(), requires_grad=module.bias.requires_grad
        )
    if hasattr(result, "out_features") and output_axis:
        result.out_features = len(keep)
    if hasattr(result, "in_features") and not output_axis:
        result.in_features = len(keep)
    return result

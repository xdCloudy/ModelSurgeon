from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelsurgeon.adapters.huggingface import remove_huggingface_transformer_layers

torch = pytest.importorskip("torch")


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(torch.nn.Linear(3, 3) for _ in range(4))


class _LayerTypesModel(_Model):
    def __init__(self) -> None:
        super().__init__()
        self.config.layer_types = [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]


def test_layer_removal_preserves_retained_modules_weights_and_mapping() -> None:
    model = _Model()
    retained = (model.model.layers[0], model.model.layers[2])
    weights = tuple(layer.weight.detach().clone() for layer in retained)

    result = remove_huggingface_transformer_layers(model, (1, 3))

    assert result.old_to_new == ((0, 0), (1, None), (2, 1), (3, None))
    assert model.config.num_hidden_layers == 2
    assert model.model.layers[0] is retained[0]
    assert model.model.layers[1] is retained[1]
    assert torch.equal(model.model.layers[0].weight, weights[0])
    assert torch.equal(model.model.layers[1].weight, weights[1])
    assert result.parameter_delta == -24


def test_layer_removal_keeps_layer_types_aligned_for_reloadable_config() -> None:
    model = _LayerTypesModel()

    remove_huggingface_transformer_layers(model, (1, 3))

    assert model.config.num_hidden_layers == 2
    assert model.config.layer_types == ["full_attention", "full_attention"]


def test_layer_removal_rejects_misaligned_layer_types_before_mutation() -> None:
    model = _LayerTypesModel()
    model.config.layer_types.pop()
    original_layers = tuple(model.model.layers)

    with pytest.raises(ValueError, match="layer_types must match"):
        remove_huggingface_transformer_layers(model, (1,))

    assert model.config.num_hidden_layers == 4
    assert tuple(model.model.layers) == original_layers

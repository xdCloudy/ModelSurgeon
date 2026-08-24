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

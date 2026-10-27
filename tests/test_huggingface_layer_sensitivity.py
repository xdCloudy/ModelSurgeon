"""CPU correctness tests for first-order physical layer-deletion sensitivity."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from modelsurgeon.adapters.huggingface.layer_sensitivity import (
    HuggingFaceLayerSensitivityError,
    measure_huggingface_layer_sensitivity,
)

torch = pytest.importorskip("torch")
nn = torch.nn


class _ScaleBlock(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden * self.scale,)


class _TinyCausalModel(nn.Module):
    def __init__(self, layer_count: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layer_count)
        self.embedding = nn.Embedding(8, 4)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            _ScaleBlock(1.1 + index * 0.1) for index in range(layer_count)
        )
        self.lm_head = nn.Linear(4, 8)

    def forward(self, *, input_ids: torch.Tensor, use_cache: bool = False) -> Any:
        del use_cache
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_first_order_sensitivity_is_finite_token_normalized_and_restores_hooks_and_modes() -> None:
    model = _TinyCausalModel()
    model.train()
    model.model.layers[0].eval()
    states_before = tuple((module, module.training) for module in model.modules())

    scores = measure_huggingface_layer_sensitivity(
        model,
        ((1, 2, 3, 4), (4, 3, 2)),
        input_device=torch.device("cpu"),
    )

    assert [score.layer_index for score in scores] == [0, 1]
    assert all(score.token_count == 5 for score in scores)
    assert all(torch.isfinite(torch.tensor(score.predicted_nll_delta)) for score in scores)
    assert all(torch.isfinite(torch.tensor(score.absolute_token_sensitivity)) for score in scores)
    assert all(
        score.absolute_token_sensitivity >= abs(score.predicted_nll_delta) for score in scores
    )
    assert all(len(layer._forward_hooks) == 0 for layer in model.model.layers)
    assert all(module.training is state for module, state in states_before)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_first_order_sensitivity_rejects_short_or_malformed_chunks_without_hooks() -> None:
    model = _TinyCausalModel()
    with pytest.raises(HuggingFaceLayerSensitivityError, match="at least two"):
        measure_huggingface_layer_sensitivity(model, ((1,),), input_device=torch.device("cpu"))
    with pytest.raises(HuggingFaceLayerSensitivityError, match="valid tokens"):
        measure_huggingface_layer_sensitivity(model, ((1, True),), input_device=torch.device("cpu"))
    assert all(len(layer._forward_hooks) == 0 for layer in model.model.layers)

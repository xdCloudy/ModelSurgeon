"""Tests for ShortGPT-style layer influence measurement."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelsurgeon.adapters.huggingface.block_influence import (
    HuggingFaceBlockInfluenceError,
    measure_huggingface_block_influence,
)

torch = pytest.importorskip("torch")


class _IdentityBlock(torch.nn.Module):
    def forward(self, hidden_states: object) -> object:
        return hidden_states


class _NegatingBlock(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return -hidden_states


class _ZeroFirstRowBlock(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states.clone()
        output[:, 0, :] = 0
        return output


class _NonFiniteBlock(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = hidden_states.clone()
        output[:, 0, 0] = self.value
        return output


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=2)
        self.embed = torch.nn.Embedding(16, 4)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList((_IdentityBlock(), _NegatingBlock()))

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        del use_cache
        hidden_states = self.embed(input_ids)
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_block_influence_measures_corresponding_token_rows_and_removes_hooks() -> None:
    model = _TinyModel().train()
    model.model.layers[0].eval()
    module_training_states = tuple(module.training for module in model.modules())

    scores = measure_huggingface_block_influence(
        model,
        ((1, 2, 3), (4, 5)),
        input_device=torch.device("cpu"),
    )

    assert scores[0].score == pytest.approx(0.0, abs=1e-6)
    assert scores[1].score == pytest.approx(2.0, abs=1e-6)
    assert tuple(item.token_count for item in scores) == (5, 5)
    assert tuple(module.training for module in model.modules()) == module_training_states
    assert all(not block._forward_hooks for block in model.model.layers)


def test_block_influence_matches_shortgpt_zero_norm_fallback_and_keeps_rows() -> None:
    model = _TinyModel()
    model.model.layers = torch.nn.ModuleList((_ZeroFirstRowBlock(),))
    model.config.num_hidden_layers = 1

    scores = measure_huggingface_block_influence(
        model,
        ((1, 2),),
        input_device=torch.device("cpu"),
    )

    assert scores[0].score == pytest.approx(0.25, abs=1e-6)
    assert scores[0].token_count == 2


def test_block_influence_uses_shortgpt_fallback_when_every_hidden_pair_is_zero() -> None:
    class ZeroBlock(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(hidden_states)

    model = _TinyModel()
    model.model.layers = torch.nn.ModuleList((ZeroBlock(), ZeroBlock()))
    model.embed.weight.data.zero_()

    scores = measure_huggingface_block_influence(
        model,
        ((1, 2),),
        input_device=torch.device("cpu"),
    )

    assert tuple(score.score for score in scores) == pytest.approx((0.5, 0.5))
    assert tuple(score.token_count for score in scores) == (2, 2)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_block_influence_rejects_nonfinite_hidden_activations(value: float) -> None:
    model = _TinyModel()
    model.model.layers = torch.nn.ModuleList((_NonFiniteBlock(value),))
    model.config.num_hidden_layers = 1

    with pytest.raises(HuggingFaceBlockInfluenceError, match="non-finite hidden states"):
        measure_huggingface_block_influence(
            model,
            ((1, 2),),
            input_device=torch.device("cpu"),
        )

    assert all(not block._forward_hooks for block in model.model.layers)

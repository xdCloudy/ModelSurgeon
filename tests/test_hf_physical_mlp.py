from __future__ import annotations

import copy
import io
from types import SimpleNamespace

import pytest

from modelsurgeon.adapters.huggingface import remove_huggingface_mlp_channels

torch = pytest.importorskip("torch")


class _MLP(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)
        self.masked: tuple[int, ...] = ()

    def forward(self, values):
        intermediate = torch.nn.functional.silu(self.gate_proj(values)) * self.up_proj(values)
        if self.masked:
            intermediate = intermediate.clone()
            intermediate[..., list(self.masked)] = 0
        return self.down_proj(intermediate)


class _Layer(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.mlp = _MLP(hidden, intermediate)

    def forward(self, values):
        return values + self.mlp(values)


class _TinyModel(torch.nn.Module):
    def __init__(self, hidden: int = 4, intermediate: int = 6, layers: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(intermediate_size=intermediate, num_hidden_layers=layers)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(_Layer(hidden, intermediate) for _ in range(layers))

    def forward(self, values):
        for layer in self.model.layers:
            values = layer(values)
        return values


def test_physical_mlp_removal_matches_mask_and_state_reloads() -> None:
    torch.manual_seed(7)
    original = _TinyModel()
    masked = copy.deepcopy(original)
    physical = copy.deepcopy(original)
    removed = (1, 4)
    for layer in masked.model.layers:
        layer.mlp.masked = removed
    values = torch.randn(2, 3, 4)
    expected = masked(values)

    result = remove_huggingface_mlp_channels(physical, removed)
    actual = physical(values)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert result.new_intermediate_size == 4
    assert result.parameter_delta == -(2 * 2 * 3 * 4)
    assert all(item.gate_shape == (4, 4) for item in result.layers)
    buffer = io.BytesIO()
    torch.save(physical.state_dict(), buffer)
    buffer.seek(0)
    reloaded = _TinyModel(intermediate=physical.config.intermediate_size)
    reloaded.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.allclose(reloaded(values), actual, rtol=1e-6, atol=1e-7)

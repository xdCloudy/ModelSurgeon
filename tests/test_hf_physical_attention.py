from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelsurgeon.adapters.huggingface import (
    HuggingFacePhysicalAttentionError,
    remove_huggingface_attention_heads,
)

torch = pytest.importorskip("torch")


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.v_proj = torch.nn.Linear(8, 4, bias=False)
        self.o_proj = torch.nn.Linear(8, 8, bias=False)
        self.num_heads = 4
        self.num_key_value_heads = 2
        self.num_key_value_groups = 2


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=8,
            head_dim=2,
            num_hidden_layers=2,
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList((_Layer(), _Layer()))


def test_complete_gqa_group_resizes_qkvo_and_metadata() -> None:
    model = _Model()

    result = remove_huggingface_attention_heads(model, (0, 1))

    assert result.removed_kv_heads == (0,)
    assert result.parameter_delta == -192
    assert model.config.num_attention_heads == 2
    assert model.config.num_key_value_heads == 1
    for layer in model.model.layers:
        assert tuple(layer.self_attn.q_proj.weight.shape) == (4, 8)
        assert tuple(layer.self_attn.k_proj.weight.shape) == (2, 8)
        assert tuple(layer.self_attn.v_proj.weight.shape) == (2, 8)
        assert tuple(layer.self_attn.o_proj.weight.shape) == (8, 4)
        assert layer.self_attn.num_key_value_groups == 2


def test_partial_gqa_group_is_rejected_before_mutation() -> None:
    model = _Model()
    original = tuple(model.model.layers[0].self_attn.q_proj.weight.shape)

    with pytest.raises(HuggingFacePhysicalAttentionError, match="complete KV groups"):
        remove_huggingface_attention_heads(model, (0,))

    assert tuple(model.model.layers[0].self_attn.q_proj.weight.shape) == original

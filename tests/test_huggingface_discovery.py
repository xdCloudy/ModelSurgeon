"""Tests for deterministic Hugging Face component discovery."""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.huggingface import (
    HuggingFaceComponentKind,
    discover_huggingface_components,
)


class Parameter:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class Layer:
    pass


class TinyModel:
    config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=3,
    )

    def __init__(self) -> None:
        self.weight = Parameter(12)
        self.bias = Parameter(3)

    def named_modules(self) -> Iterable[tuple[str, object]]:
        return [
            ("model.layers.1.mlp", Layer()),
            ("", self),
            ("model", Layer()),
            ("model.layers.0", Layer()),
            ("model.layers.0.mlp", Layer()),
            ("model.layers.0.self_attn", Layer()),
            ("model.layers.0.self_attn.q_proj", Layer()),
        ]

    def named_parameters(self) -> Iterable[tuple[str, Parameter]]:
        return [
            ("model.layers.0.self_attn.q_proj.weight", self.weight),
            ("model.layers.0.self_attn.q_proj.bias", self.bias),
        ]


def test_discovery_is_deterministic_and_canonical() -> None:
    first = discover_huggingface_components(TinyModel(), ModelFamily.LLAMA)
    second = discover_huggingface_components(TinyModel(), ModelFamily.LLAMA)

    first_records = [item.to_record() for item in first.components()]
    second_records = [item.to_record() for item in second.components()]
    assert first_records == second_records
    assert [str(item.component_id) for item in first.modules] == [
        "model",
        "model.layers.0",
        "model.layers.0.mlp",
        "model.layers.0.self_attn",
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.mlp",
    ]
    assert all(item.kind for item in first.components())


def test_parameter_counts_reconcile_exactly() -> None:
    discovery = discover_huggingface_components(TinyModel(), ModelFamily.QWEN)

    assert discovery.parameter_count == 15
    assert sum(dict(item.attributes)["element_count"] for item in discovery.parameters) == 15
    assert discovery.to_record() == {
        "family": "qwen",
        "module_count": 6,
        "parameter_tensor_count": 2,
        "parameter_count": 15,
        "logical_component_count": 18,
    }


def test_heads_kv_heads_and_channels_are_discovered_lazily() -> None:
    discovery = discover_huggingface_components(TinyModel(), ModelFamily.GEMMA)
    logical = list(discovery.components())[len(discovery.modules) + len(discovery.parameters) :]

    kinds = [item.kind for item in logical]
    assert kinds.count(HuggingFaceComponentKind.ATTENTION_HEAD.value) == 4
    assert kinds.count(HuggingFaceComponentKind.KV_HEAD.value) == 2
    assert kinds.count(HuggingFaceComponentKind.MLP_CHANNEL.value) == 6
    attention_heads = [
        item for item in logical if item.kind == HuggingFaceComponentKind.ATTENTION_HEAD.value
    ]
    assert str(attention_heads[0].component_id) == "model.layers.0.self_attn.head.0"
    assert str(logical[-1].component_id) == "model.layers.1.mlp.channel.2"


def test_shared_parameters_are_counted_once() -> None:
    model = TinyModel()

    def named_parameters() -> Iterable[tuple[str, Parameter]]:
        return [
            ("model.embed_tokens.weight", model.weight),
            ("lm_head.weight", model.weight),
        ]

    model.named_parameters = named_parameters  # type: ignore[method-assign]
    discovery = discover_huggingface_components(model, ModelFamily.MISTRAL)

    assert discovery.parameter_count == 12
    assert len(discovery.parameters) == 1


@pytest.mark.parametrize(
    "field",
    ["num_hidden_layers", "num_attention_heads", "intermediate_size"],
)
def test_invalid_or_missing_shape_fields_fail_closed(field: str) -> None:
    model = TinyModel()
    setattr(model.config, field, 0)

    with pytest.raises(ValueError, match=field):
        discover_huggingface_components(model, ModelFamily.LLAMA)

    setattr(model.config, field, 2 if field != "intermediate_size" else 3)

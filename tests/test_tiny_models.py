"""Tests for deterministic, dependency-free transformer fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.huggingface import discover_huggingface_components
from modelsurgeon.testing import TinyParameter, tiny_transformer

MANIFEST = Path(__file__).parent / "fixtures" / "tiny_hf_models_v1.json"


@pytest.mark.parametrize("family", list(ModelFamily))
def test_offline_doubles_are_deterministic_and_discoverable(family: ModelFamily) -> None:
    first = tiny_transformer(family)
    second = tiny_transformer(family)

    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64
    assert [name for name, _ in first.named_modules()] == [
        name for name, _ in second.named_modules()
    ]
    assert [type(module).__name__ for _, module in first.named_modules()] == [
        type(module).__name__ for _, module in second.named_modules()
    ]
    first_parameters = list(first.named_parameters())
    second_parameters = list(second.named_parameters())
    assert [(name, parameter.shape) for name, parameter in first_parameters] == [
        (name, parameter.shape) for name, parameter in second_parameters
    ]
    assert first_parameters[0][1].sample(0) == second_parameters[0][1].sample(0)

    discovery = discover_huggingface_components(first, family)
    expected = sum(parameter.numel() for _, parameter in first_parameters)
    assert discovery.parameter_count == expected
    assert discovery.shape.layers == 2


def test_seed_changes_values_and_identity_without_changing_structure() -> None:
    first = tiny_transformer(ModelFamily.LLAMA, seed=1)
    second = tiny_transformer(ModelFamily.LLAMA, seed=2)
    first_parameters = list(first.named_parameters())
    second_parameters = list(second.named_parameters())

    assert first.fingerprint() != second.fingerprint()
    assert [parameter.shape for _, parameter in first_parameters] == [
        parameter.shape for _, parameter in second_parameters
    ]
    assert first_parameters[0][1].sample(0) != second_parameters[0][1].sample(0)


def test_tied_embeddings_share_identity_and_discovery_counts_once() -> None:
    model = tiny_transformer(ModelFamily.GEMMA, tie_word_embeddings=True)
    parameters = dict(model.named_parameters())
    embedding = parameters["model.embed_tokens.weight"]
    head = parameters["lm_head.weight"]

    assert embedding is head
    raw_total = sum(parameter.numel() for parameter in parameters.values())
    discovery = discover_huggingface_components(model, ModelFamily.GEMMA)
    assert discovery.parameter_count == raw_total - head.numel()


def test_parameter_samples_are_bounded_and_validate_indices() -> None:
    parameter = TinyParameter("weight", (2, 3), 7)

    assert -1.0 <= parameter.sample(0) <= 1.0
    assert parameter.sample(0) != parameter.sample(1)
    with pytest.raises(IndexError, match="outside"):
        parameter.sample(-1)
    with pytest.raises(IndexError, match="outside"):
        parameter.sample(6)


def test_pinned_hf_manifest_has_no_weights_and_exact_revisions() -> None:
    payload = cast(dict[str, object], json.loads(MANIFEST.read_text(encoding="utf-8")))
    models = cast(list[dict[str, object]], payload["models"])
    fingerprints = cast(dict[str, str], payload["offline_fingerprints"])

    assert payload["schema_version"] == 1
    assert payload["weights_committed"] is False
    assert fingerprints == {
        family.value: tiny_transformer(family).fingerprint() for family in ModelFamily
    }
    assert {model["family"] for model in models} == {family.value for family in ModelFamily}
    assert all(len(cast(str, model["revision"])) == 40 for model in models)
    assert all(
        all(character in "0123456789abcdef" for character in cast(str, model["revision"]))
        for model in models
    )
    assert not any(
        path.suffix in {".bin", ".safetensors", ".pt"}
        for path in MANIFEST.parent.iterdir()
    )

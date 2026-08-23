import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from modelsurgeon.graph import ComponentId, ComponentSegment

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "component_id_conformance.json").read_text(
        encoding="utf-8"
    )
)


def test_parse_round_trip_and_typed_indices() -> None:
    component_id = ComponentId.parse("model.layers.17.self_attn.head.4")

    assert str(component_id) == "model.layers.17.self_attn.head.4"
    assert component_id.segments[2].value == 17
    assert component_id.parent == ComponentId.parse("model.layers.17.self_attn.head")


@pytest.mark.parametrize("value", ["", ".model", "model.", "model..layers", "model.-1"])
def test_parse_rejects_invalid_ids(value: str) -> None:
    with pytest.raises(ValueError):
        ComponentId.parse(value)


def test_child_does_not_mutate_parent() -> None:
    parent = ComponentId.parse("model.layers.2")

    assert parent.child("mlp").child("up_proj") == ComponentId.parse(
        "model.layers.2.mlp.up_proj"
    )
    assert str(parent) == "model.layers.2"


@pytest.mark.parametrize("vector", VECTORS["valid"], ids=lambda value: value["canonical"])
def test_valid_conformance_vectors_round_trip(vector: dict[str, str]) -> None:
    component_id = ComponentId.parse(vector["canonical"])

    assert str(component_id) == vector["canonical"]
    assert ComponentId.from_json(component_id.to_json()) == component_id


@pytest.mark.parametrize("vector", VECTORS["invalid"], ids=lambda value: value["reason"])
def test_invalid_conformance_vectors_are_rejected(vector: dict[str, str]) -> None:
    with pytest.raises(ValueError, match=r".+"):
        ComponentId.parse(vector["text"])


def test_provider_names_are_canonically_escaped() -> None:
    component_id = ComponentId.parse("model.layers.0").child("with.dot").child("snowman☃")

    assert str(component_id) == "model.layers.0.~with%2Edot.~snowman%E2%98%83"
    assert ComponentId.parse(str(component_id)) == component_id


def test_numeric_provider_name_is_distinct_from_index() -> None:
    parent = ComponentId.parse("model.module")

    assert str(parent.child("123")) == "model.module.~123"
    assert str(parent.child(123)) == "model.module.123"
    assert parent.child("123") != parent.child(123)


def test_component_ids_are_hashable_ordered_and_immutable() -> None:
    earlier = ComponentId.parse("model.layers.2")
    later = ComponentId.parse("model.layers.10")

    assert len({earlier, ComponentId.parse("model.layers.2")}) == 1
    assert sorted([later, earlier]) == [later, earlier]
    with pytest.raises(FrozenInstanceError):
        earlier.segments = (ComponentSegment("model"),)  # type: ignore[misc]


def test_module_names_are_placed_under_model_root() -> None:
    assert ComponentId.from_module_name("") == ComponentId.parse("model")
    assert ComponentId.from_module_name("transformer.h.0") == ComponentId.parse(
        "model.transformer.h.0"
    )


def test_json_representation_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        ComponentId.from_json({"component_id": "model"})

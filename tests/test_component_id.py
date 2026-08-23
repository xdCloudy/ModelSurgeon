import pytest

from modelsurgeon.graph import ComponentId


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


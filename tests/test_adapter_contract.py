import json
from dataclasses import fields
from typing import Any

import pytest

from modelsurgeon.adapters import (
    AdapterCapability,
    AdapterIdentity,
    ComponentDescriptor,
    ModelFormat,
    ModelSource,
    MutationSupport,
    OpenOptions,
    TensorDescriptor,
    UnsupportedCapabilityError,
    require_capability,
)
from modelsurgeon.graph import ComponentId


def test_capabilities_are_independent_and_fail_explicitly() -> None:
    identity = AdapterIdentity("gguf", "1")
    capabilities = frozenset(
        {AdapterCapability.DISCOVER, AdapterCapability.TENSOR_METADATA}
    )

    require_capability(identity, capabilities, AdapterCapability.DISCOVER)
    with pytest.raises(UnsupportedCapabilityError, match="native_quantized_surgery") as raised:
        require_capability(
            identity,
            capabilities,
            AdapterCapability.NATIVE_QUANTIZED_SURGERY,
            reason="codec is read-only",
        )

    assert raised.value.adapter == identity
    assert raised.value.capability is AdapterCapability.NATIVE_QUANTIZED_SURGERY
    assert raised.value.reason == "codec is read-only"


def test_persisted_records_are_json_serializable_primitives() -> None:
    parent = ComponentId.parse("model.layers.2")
    component = ComponentDescriptor(
        component_id=parent.child("mlp"),
        kind="mlp",
        parent=parent,
        attributes=(("intermediate_size", 4096),),
    )
    tensor = TensorDescriptor(
        component_id=component.component_id.child("up_proj"),
        tensor_name="blk.2.ffn_up.weight",
        shape=(4096, 1024),
        dtype="Q4_K",
        storage_bytes=2_359_296,
        quantization="Q4_K_M",
    )
    support = MutationSupport(
        supported=True,
        reason="channel axis is adapter-defined",
        constraints=(("block_size", 256),),
    )
    source = ModelSource(
        format=ModelFormat.GGUF,
        locator="model.gguf",
        content_digest="sha256:abc",
    )

    payload = {
        "adapter": AdapterIdentity("qwen-gguf", "1").to_record(),
        "source": source.to_record(),
        "component": component.to_record(),
        "tensor": tensor.to_record(),
        "support": support.to_record(),
    }

    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize(
    "record_type",
    [AdapterIdentity, ModelSource, ComponentDescriptor, TensorDescriptor, MutationSupport],
)
def test_persisted_record_fields_do_not_expose_any_or_object(record_type: type[Any]) -> None:
    annotations = {field.name: field.type for field in fields(record_type)}

    assert Any not in annotations.values()
    assert object not in annotations.values()


def test_component_descriptor_rejects_inconsistent_parent() -> None:
    with pytest.raises(ValueError, match="parent"):
        ComponentDescriptor(
            component_id=ComponentId.parse("model.layers.2.mlp"),
            kind="mlp",
            parent=ComponentId.parse("model.layers.1"),
        )


@pytest.mark.parametrize(
    ("option_name", "value"),
    [
        ("max_ram_bytes", 0),
        ("max_vram_bytes", -1),
    ],
)
def test_open_options_reject_non_positive_limits(
    option_name: str,
    value: int,
) -> None:
    with pytest.raises(ValueError, match=option_name):
        OpenOptions(**{option_name: value})

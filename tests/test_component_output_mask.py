"""Tests for reversible graph-addressed component output masking."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.component_mask import (
    ComponentMaskError,
    ComponentOutputMask,
    OutputSignature,
    SequenceOutputMaskBackend,
)
from modelsurgeon.surgery.transaction import InMemoryMutationTransaction


class SnapshotTarget:
    def snapshot(self) -> object:
        return b"stable"

    def restore(self, snapshot: object) -> None:
        assert snapshot == b"stable"


@dataclass
class Handle:
    module: Module
    hook: object

    def remove(self) -> None:
        self.module.hooks.remove(self.hook)


class Module:
    def __init__(self) -> None:
        self.hooks: list[object] = []

    def register_forward_hook(self, hook: object) -> Handle:
        self.hooks.append(hook)
        return Handle(self, hook)

    def emit(self, output: object) -> object:
        result = output
        for hook in tuple(self.hooks):
            result = hook(self, (), result)  # type: ignore[operator]
        return result


def _transaction(owner: object) -> InMemoryMutationTransaction:
    return InMemoryMutationTransaction(owner, {"hook_state": SnapshotTarget()}, ("hook_state",))


def test_selected_last_axis_mask_is_deterministic_and_reversible() -> None:
    owner = object()
    module = Module()
    component = ComponentId.parse("model.layers.0.mlp")
    source = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

    with _transaction(owner) as transaction:
        with ComponentOutputMask(component, module, transaction, indices=(1,)) as mask:
            first = module.emit(source)
            second = module.emit(source)
            assert mask.active
            assert first == ((1.0, 0, 3.0), (4.0, 0, 6.0))
            assert second == first
        transaction.commit()

    assert module.hooks == []
    assert module.emit(source) == source


def test_full_output_mask_preserves_shape_container_and_dtype_signature() -> None:
    owner = object()
    module = Module()
    component = ComponentId.parse("model.layers.0.self_attn")
    source = [[1.5, -2.0], [3.0, 4.0]]
    backend = SequenceOutputMaskBackend()
    before = backend.signature(source)

    with _transaction(owner) as transaction, ComponentOutputMask(
        component, module, transaction, backend=backend
    ):
        masked = module.emit(source)
        assert masked == [[0, 0], [0, 0]]
        assert backend.signature(masked) == before


def test_out_of_range_and_unsupported_outputs_fail_with_component_context() -> None:
    owner = object()
    module = Module()
    component = ComponentId.parse("model.layers.2.mlp")

    with _transaction(owner) as transaction, ComponentOutputMask(
        component, module, transaction, indices=(3,)
    ):
        with pytest.raises(ComponentMaskError, match=r"model\.layers\.2\.mlp"):
            module.emit(((1.0, 2.0),))

    with _transaction(object()) as transaction, ComponentOutputMask(
        component, module, transaction
    ):
        with pytest.raises(ComponentMaskError, match="unsupported"):
            module.emit({"hidden": (1.0, 2.0)})


def test_mask_requires_prepared_transaction_ownership() -> None:
    owner = object()
    module = Module()
    component = ComponentId.parse("model.layers.0.mlp")
    transaction = _transaction(owner)

    with pytest.raises(ValueError, match="prepared transaction"):
        ComponentOutputMask(component, module, transaction).__enter__()


def test_backend_shape_or_dtype_change_fails_closed() -> None:
    class BadBackend:
        calls = 0

        def signature(self, output: object) -> OutputSignature:
            del output
            self.calls += 1
            return OutputSignature((1, 2) if self.calls == 1 else (1, 3), "float32")

        def mask_last_axis(
            self, output: object, indices: tuple[int, ...] | None
        ) -> object:
            del indices
            return output

    owner = object()
    module = Module()
    component = ComponentId.parse("model.layers.0.mlp")
    with _transaction(owner) as transaction, ComponentOutputMask(
        component, module, transaction, backend=BadBackend()
    ):
        with pytest.raises(ComponentMaskError, match="shape or dtype"):
            module.emit(((1.0, 2.0),))


def test_mask_index_contract_rejects_duplicates_noncanonical_and_negative() -> None:
    owner = object()
    module = Module()
    component = ComponentId.parse("model.layers.0.mlp")
    transaction = _transaction(owner)
    for indices in ((1, 1), (2, 1), (-1,)):
        with pytest.raises(ComponentMaskError):
            ComponentOutputMask(component, module, transaction, indices=indices)

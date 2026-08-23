"""Tests for reversible transformer-layer bypass execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pytest

from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.layer_bypass import (
    DirectResidualBypassBackend,
    LayerBypassError,
    TransformerLayerBypass,
)
from modelsurgeon.surgery.transaction import InMemoryMutationTransaction


class SnapshotTarget:
    def snapshot(self) -> object:
        return b"stable"

    def restore(self, snapshot: object) -> None:
        assert snapshot == b"stable"


@dataclass
class Handle:
    layer: Layer

    def remove(self) -> None:
        self.layer.replacement = None


class Layer:
    def __init__(self) -> None:
        self.replacement: Callable[[tuple[object, ...], Mapping[str, object]], object] | None = None
        self.executions = 0

    def install_bypass(
        self,
        replacement: Callable[[tuple[object, ...], Mapping[str, object]], object],
    ) -> Handle:
        if self.replacement is not None:
            raise RuntimeError("already bypassed")
        self.replacement = replacement
        return Handle(self)

    def __call__(self, residual: object) -> object:
        if self.replacement is not None:
            return self.replacement((residual,), {})
        self.executions += 1
        if not isinstance(residual, tuple):
            return residual
        return tuple(
            tuple(float(value) + 1.0 for value in row) if isinstance(row, tuple) else row
            for row in residual
        )


def _transaction() -> InMemoryMutationTransaction:
    return InMemoryMutationTransaction(object(), {"layer": SnapshotTarget()}, ("layer",))


def _component() -> ComponentId:
    return ComponentId.parse("model.layers.1")


def test_bypass_skips_block_body_and_returns_compatible_residual() -> None:
    layer = Layer()
    residual = ((1.0, 2.0), (3.0, 4.0))
    assert layer(residual) == ((2.0, 3.0), (4.0, 5.0))
    assert layer.executions == 1

    with _transaction() as transaction, TransformerLayerBypass(
        _component(), layer, transaction
    ) as bypass:
        assert bypass.active
        assert layer(residual) is residual
        assert layer.executions == 1

    assert layer.replacement is None
    assert layer(residual) == ((2.0, 3.0), (4.0, 5.0))
    assert layer.executions == 2


def test_transaction_rejection_scope_restores_original_execution() -> None:
    layer = Layer()
    residual = ((1.0, 2.0),)
    transaction = _transaction()

    with transaction, TransformerLayerBypass(_component(), layer, transaction):
        assert layer(residual) is residual
        transaction.mark_applied()

    assert transaction.state.value == "rolled_back"
    assert layer.replacement is None
    assert layer(residual) == ((2.0, 3.0),)


def test_unsupported_residual_layout_fails_closed_with_component_context() -> None:
    layer = Layer()
    with (
        _transaction() as transaction,
        TransformerLayerBypass(_component(), layer, transaction),
        pytest.raises(LayerBypassError, match=r"model\.layers\.1"),
    ):
        layer({"hidden": (1.0, 2.0)})


def test_backend_requires_positional_residual() -> None:
    with pytest.raises(LayerBypassError, match="positional residual"):
        DirectResidualBypassBackend().bypass((), {})


def test_bypass_requires_prepared_transaction() -> None:
    layer = Layer()
    transaction = _transaction()
    with pytest.raises(ValueError, match="prepared transaction"):
        TransformerLayerBypass(_component(), layer, transaction).__enter__()

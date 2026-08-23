"""Tests for individual and grouped MLP intermediate-channel masking."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.mlp_mask import (
    MLPChannelMask,
    MLPChannelMaskError,
    MLPChannelMaskSpec,
    MLPPathPoint,
    MLPPathRole,
    MLPVariant,
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
    def __init__(self, *, fail: bool = False) -> None:
        self.hooks: list[object] = []
        self.fail = fail

    def register_forward_hook(self, hook: object) -> Handle:
        if self.fail:
            raise RuntimeError("registration failed")
        self.hooks.append(hook)
        return Handle(self, hook)

    def emit(self, output: object) -> object:
        result = output
        for hook in tuple(self.hooks):
            result = hook(self, (), result)  # type: ignore[operator]
        return result


def _transaction() -> InMemoryMutationTransaction:
    return InMemoryMutationTransaction(object(), {"hook": SnapshotTarget()}, ("hook",))


def _point(name: str, module: Module) -> MLPPathPoint:
    return MLPPathPoint(ComponentId.parse(f"model.layers.0.mlp.{name}"), module)


def test_gated_path_masks_same_group_across_gate_up_and_down_input() -> None:
    gate = Module()
    up = Module()
    down = Module()
    points = {
        MLPPathRole.GATE: _point("gate_proj", gate),
        MLPPathRole.UP: _point("up_proj", up),
        MLPPathRole.DOWN_INPUT: _point("down_input", down),
    }
    spec = MLPChannelMaskSpec((1, 3), 5, MLPVariant.GATED)
    source = ((10.0, 20.0, 30.0, 40.0, 50.0),)

    with _transaction() as transaction, MLPChannelMask(spec, points, transaction) as mask:
        assert mask.active
        expected = ((10.0, 0.0, 30.0, 0.0, 50.0),)
        assert gate.emit(source) == expected
        assert up.emit(source) == expected
        assert down.emit(source) == expected

    assert gate.emit(source) == source
    assert up.emit(source) == source
    assert down.emit(source) == source
    assert all(not module.hooks for module in (gate, up, down))


def test_ungated_variant_requires_up_and_down_only() -> None:
    up = Module()
    down = Module()
    points = {
        MLPPathRole.UP: _point("up_proj", up),
        MLPPathRole.DOWN_INPUT: _point("down_input", down),
    }
    spec = MLPChannelMaskSpec((2,), 4, MLPVariant.UNGATED)

    with _transaction() as transaction, MLPChannelMask(spec, points, transaction):
        assert up.emit(((1.0, 2.0, 3.0, 4.0),)) == ((1.0, 2.0, 0.0, 4.0),)
        assert down.emit(((5.0, 6.0, 7.0, 8.0),)) == ((5.0, 6.0, 0.0, 8.0),)


def test_incomplete_or_wrong_variant_path_fails_before_hook_registration() -> None:
    gate = Module()
    up = Module()
    with pytest.raises(MLPChannelMaskError, match="missing"):
        MLPChannelMask(
            MLPChannelMaskSpec((0,), 4),
            {
                MLPPathRole.GATE: _point("gate_proj", gate),
                MLPPathRole.UP: _point("up_proj", up),
            },
            _transaction(),
        )
    assert gate.hooks == []
    assert up.hooks == []

    down = Module()
    with pytest.raises(MLPChannelMaskError, match="unsupported"):
        MLPChannelMask(
            MLPChannelMaskSpec((0,), 4, MLPVariant.UNGATED),
            {
                MLPPathRole.GATE: _point("gate_proj", gate),
                MLPPathRole.UP: _point("up_proj", up),
                MLPPathRole.DOWN_INPUT: _point("down_input", down),
            },
            _transaction(),
        )


def test_partial_registration_failure_removes_previously_installed_masks() -> None:
    gate = Module()
    up = Module(fail=True)
    down = Module()
    points = {
        MLPPathRole.GATE: _point("gate_proj", gate),
        MLPPathRole.UP: _point("up_proj", up),
        MLPPathRole.DOWN_INPUT: _point("down_input", down),
    }
    spec = MLPChannelMaskSpec((1,), 4)

    with _transaction() as transaction, pytest.raises(ValueError, match="failed to register"):
        MLPChannelMask(spec, points, transaction).__enter__()

    assert gate.hooks == []
    assert up.hooks == []
    assert down.hooks == []


def test_channel_group_contract_rejects_invalid_indices() -> None:
    for channels in ((), (1, 1), (2, 1), (-1,), (4,)):
        with pytest.raises(MLPChannelMaskError):
            MLPChannelMaskSpec(channels, 4)

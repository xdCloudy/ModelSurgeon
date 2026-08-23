"""Versioned golden regressions for reversible masking experiments."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.head_mask import AttentionHeadMask, AttentionHeadMaskSpec
from modelsurgeon.surgery.layer_bypass import TransformerLayerBypass
from modelsurgeon.surgery.mlp_mask import (
    MLPChannelMask,
    MLPChannelMaskSpec,
    MLPPathPoint,
    MLPPathRole,
)
from modelsurgeon.surgery.transaction import InMemoryMutationTransaction

GOLDEN_FIXTURE_VERSION = 1
GOLDEN_PATH = Path(__file__).parent / "golden" / "masking_v1.json"


class SnapshotTarget:
    def snapshot(self) -> object:
        return b"stable"

    def restore(self, snapshot: object) -> None:
        assert snapshot == b"stable"


@dataclass
class HookHandle:
    module: HookModule
    hook: object

    def remove(self) -> None:
        self.module.hooks.remove(self.hook)


class HookModule:
    def __init__(self) -> None:
        self.hooks: list[object] = []

    def register_forward_hook(self, hook: object) -> HookHandle:
        self.hooks.append(hook)
        return HookHandle(self, hook)

    def emit(self, output: object) -> object:
        result = output
        for hook in tuple(self.hooks):
            result = hook(self, (), result)  # type: ignore[operator]
        return result


@dataclass
class BypassHandle:
    layer: TinyLayer

    def remove(self) -> None:
        self.layer.replacement = None


class TinyLayer:
    def __init__(self) -> None:
        self.replacement: Callable[[tuple[object, ...], Mapping[str, object]], object] | None = None
        self.executions = 0

    def install_bypass(
        self,
        replacement: Callable[[tuple[object, ...], Mapping[str, object]], object],
    ) -> BypassHandle:
        if self.replacement is not None:
            raise RuntimeError("already bypassed")
        self.replacement = replacement
        return BypassHandle(self)

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
    return InMemoryMutationTransaction(
        object(),
        {"golden": SnapshotTarget()},
        ("golden",),
    )


def _fixture() -> dict[str, Any]:
    raw = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AssertionError("golden masking fixture must be an object")
    assert raw["fixture_version"] == GOLDEN_FIXTURE_VERSION
    return raw


def _matrix(value: object) -> tuple[tuple[float, ...], ...]:
    assert isinstance(value, list)
    return tuple(tuple(float(item) for item in row) for row in value)


def test_attention_head_mask_matches_versioned_golden_and_preserves_neighbor() -> None:
    golden = _fixture()["head"]
    assert isinstance(golden, dict)
    target = HookModule()
    neighbor = HookModule()
    source = _matrix(golden["input"])
    component = ComponentId.parse(str(golden["affected_ids"][0]))
    spec = AttentionHeadMaskSpec(
        component,
        int(golden["head_index"]),
        int(golden["query_heads"]),
        int(golden["key_value_heads"]),
        int(golden["head_width"]),
    )

    assert [str(spec.component_id)] == golden["affected_ids"]
    assert str(ComponentId.parse(str(golden["neighbor_id"]))) == golden["neighbor_id"]
    with _transaction() as transaction, AttentionHeadMask(spec, target, transaction):
        assert target.emit(source) == _matrix(golden["expected_masked"])
        assert neighbor.emit(source) == _matrix(golden["expected_neighbor"])

    assert target.emit(source) == source
    assert neighbor.emit(source) == source
    assert target.hooks == []
    assert neighbor.hooks == []


def test_mlp_channel_mask_matches_versioned_golden_and_preserves_neighbor() -> None:
    golden = _fixture()["mlp"]
    assert isinstance(golden, dict)
    gate = HookModule()
    up = HookModule()
    down = HookModule()
    neighbor = HookModule()
    affected = tuple(str(value) for value in golden["affected_ids"])
    points = {
        MLPPathRole.GATE: MLPPathPoint(ComponentId.parse(affected[0]), gate),
        MLPPathRole.UP: MLPPathPoint(ComponentId.parse(affected[1]), up),
        MLPPathRole.DOWN_INPUT: MLPPathPoint(ComponentId.parse(affected[2]), down),
    }
    spec = MLPChannelMaskSpec(
        tuple(int(value) for value in golden["channels"]),
        int(golden["width"]),
    )
    source = _matrix(golden["input"])

    assert tuple(str(points[role].component_id) for role in points) == affected
    assert str(ComponentId.parse(str(golden["neighbor_id"]))) == golden["neighbor_id"]
    with _transaction() as transaction, MLPChannelMask(spec, points, transaction):
        expected = _matrix(golden["expected_masked"])
        assert gate.emit(source) == expected
        assert up.emit(source) == expected
        assert down.emit(source) == expected
        assert neighbor.emit(source) == _matrix(golden["expected_neighbor"])

    assert all(module.emit(source) == source for module in (gate, up, down, neighbor))
    assert all(not module.hooks for module in (gate, up, down, neighbor))


def test_layer_bypass_matches_versioned_golden_and_preserves_neighbor() -> None:
    golden = _fixture()["layer"]
    assert isinstance(golden, dict)
    target = TinyLayer()
    neighbor = TinyLayer()
    residual = _matrix(golden["input"])
    component = ComponentId.parse(str(golden["affected_ids"][0]))

    assert [str(component)] == golden["affected_ids"]
    assert str(ComponentId.parse(str(golden["neighbor_id"]))) == golden["neighbor_id"]
    assert target(residual) == _matrix(golden["expected_normal"])
    executions_before = target.executions

    with _transaction() as transaction, TransformerLayerBypass(
        component, target, transaction
    ):
        assert target(residual) == _matrix(golden["expected_bypass"])
        assert target.executions == executions_before
        assert neighbor(residual) == _matrix(golden["expected_neighbor"])

    assert target.replacement is None
    assert target(residual) == _matrix(golden["expected_normal"])

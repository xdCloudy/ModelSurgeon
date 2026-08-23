"""Tests for transformer graph construction from discovery records."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pytest

from modelsurgeon.graph import (
    ComponentId,
    EdgeKind,
    GraphBuildError,
    build_component_graph,
)


@dataclass(frozen=True)
class Record:
    component_id: ComponentId
    kind: str
    attributes: tuple[tuple[str, str | int | float | bool | None], ...] = ()


def _record(path: str, kind: str = "module") -> Record:
    return Record(ComponentId.parse(path), kind)


def _fixture_records() -> tuple[Record, ...]:
    paths = (
        ("model", "model"),
        ("model.layers", "module"),
        ("model.layers.0", "transformer_layer"),
        ("model.layers.0.input_layernorm", "normalization"),
        ("model.layers.0.self_attn", "attention"),
        ("model.layers.0.self_attn.q_proj", "projection"),
        ("model.layers.0.self_attn.k_proj", "projection"),
        ("model.layers.0.self_attn.v_proj", "projection"),
        ("model.layers.0.self_attn.o_proj", "projection"),
        ("model.layers.0.post_attention_layernorm", "normalization"),
        ("model.layers.0.mlp", "mlp"),
        ("model.layers.0.mlp.gate_proj", "projection"),
        ("model.layers.0.mlp.up_proj", "projection"),
        ("model.layers.0.mlp.down_proj", "projection"),
    )
    return tuple(_record(path, kind) for path, kind in reversed(paths))


def test_builder_constructs_complete_hierarchy_and_dataflow() -> None:
    graph = build_component_graph(_fixture_records())
    edge_keys = {(edge.kind, str(edge.source), str(edge.target)) for edge in graph.edges}

    assert (
        EdgeKind.CHILD,
        "model.layers.0",
        "model.layers.0.residual_attn",
    ) in edge_keys
    assert (
        EdgeKind.PARENT,
        "model.layers.0.residual_attn",
        "model.layers.0",
    ) in edge_keys
    assert (
        EdgeKind.PRODUCES,
        "model.layers.0.input_layernorm",
        "model.layers.0.self_attn.q_proj",
    ) in edge_keys
    assert (
        EdgeKind.CONSUMES,
        "model.layers.0.mlp.down_proj",
        "model.layers.0.mlp.gate_proj",
    ) in edge_keys
    assert all(
        edge.source in {node.component_id for node in graph.nodes}
        and edge.target in {node.component_id for node in graph.nodes}
        for edge in graph.edges
    )


def test_projection_coupling_closures_are_complete() -> None:
    graph = build_component_graph(_fixture_records())
    coupled = {
        frozenset((str(edge.source), str(edge.target)))
        for edge in graph.edges
        if edge.kind is EdgeKind.COUPLED
    }
    attention = {
        f"model.layers.0.self_attn.{name}"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    mlp = {
        f"model.layers.0.mlp.{name}" for name in ("gate_proj", "up_proj", "down_proj")
    }

    assert {frozenset(pair) for pair in combinations(attention, 2)} <= coupled
    assert {frozenset(pair) for pair in combinations(mlp, 2)} <= coupled
    assert [constraint.constraint_id for constraint in graph.constraints] == [
        "model.layers.0:attention-projections:v1",
        "model.layers.0:mlp-projections:v1",
    ]


def test_build_is_deterministic_for_reordered_discovery() -> None:
    records = _fixture_records()

    first = build_component_graph(records).to_record()
    second = build_component_graph(tuple(reversed(records))).to_record()

    assert first == second


def test_incomplete_supported_projection_group_fails_closed() -> None:
    records = tuple(
        record
        for record in _fixture_records()
        if str(record.component_id) != "model.layers.0.self_attn.k_proj"
    )

    with pytest.raises(GraphBuildError, match="attention projection group is incomplete"):
        build_component_graph(records)


def test_missing_parent_and_duplicate_components_fail_closed() -> None:
    with pytest.raises(GraphBuildError, match="missing parent"):
        build_component_graph((_record("model", "model"), _record("model.orphan.child")))
    root = _record("model", "model")
    with pytest.raises(GraphBuildError, match="duplicate component ID"):
        build_component_graph((root, root))

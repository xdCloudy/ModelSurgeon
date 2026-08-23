"""Tests for the versioned dependency and coupling graph schema."""

from __future__ import annotations

import json

import pytest

from modelsurgeon.graph import (
    ComponentGraph,
    ComponentId,
    ConstraintKind,
    EdgeKind,
    GraphEdge,
    GraphNode,
    MutationConstraint,
)

ROOT = ComponentId.parse("model")
LAYER = ComponentId.parse("model.layers")
LAYER_ZERO = ComponentId.parse("model.layers.0")
ATTENTION = ComponentId.parse("model.layers.0.self_attn")
MLP = ComponentId.parse("model.layers.0.mlp")


def _nodes() -> tuple[GraphNode, ...]:
    return (
        GraphNode(MLP, "mlp"),
        GraphNode(ROOT, "model"),
        GraphNode(ATTENTION, "attention"),
        GraphNode(LAYER_ZERO, "transformer_layer"),
        GraphNode(LAYER, "module_list"),
    )


def test_graph_records_are_canonical_and_json_serializable() -> None:
    constraint = MutationConstraint(
        constraint_id="layer-width",
        kind=ConstraintKind.SAME_HIDDEN_SIZE,
        members=(MLP, ATTENTION),
        parameters=(("axis", 1),),
    )
    graph = ComponentGraph.build(
        _nodes(),
        edges=(
            GraphEdge(MLP, ATTENTION, EdgeKind.CONSUMES),
            GraphEdge(ROOT, LAYER, EdgeKind.CHILD),
            GraphEdge(
                ATTENTION,
                MLP,
                EdgeKind.CONSTRAINED,
                attributes=(("constraint_id", "layer-width"),),
            ),
            GraphEdge(MLP, ATTENTION, EdgeKind.COUPLED),
        ),
        constraints=(constraint,),
    )

    payload = graph.to_record()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["schema_version"] == 1
    assert payload["edge_semantics_version"] == 1
    assert payload["constraint_schema_version"] == 1
    assert [node["component_id"] for node in payload["nodes"]] == [
        "model",
        "model.layers",
        "model.layers.0",
        "model.layers.0.mlp",
        "model.layers.0.self_attn",
    ]


@pytest.mark.parametrize(
    "edge",
    [
        GraphEdge(LAYER_ZERO, LAYER, EdgeKind.PARENT),
        GraphEdge(LAYER, LAYER_ZERO, EdgeKind.CHILD),
        GraphEdge(ATTENTION, MLP, EdgeKind.CONSUMES),
        GraphEdge(ATTENTION, MLP, EdgeKind.PRODUCES),
        GraphEdge(MLP, ATTENTION, EdgeKind.COUPLED),
        GraphEdge(
            ATTENTION,
            MLP,
            EdgeKind.CONSTRAINED,
            attributes=(("constraint_id", "width"),),
        ),
    ],
)
def test_all_versioned_edge_kinds_serialize(edge: GraphEdge) -> None:
    assert edge.to_record()["semantics_version"] == 1
    assert edge.to_record()["kind"] == edge.kind.value


def test_hierarchy_and_undirected_coupling_semantics_are_enforced() -> None:
    with pytest.raises(ValueError, match="canonical parent"):
        GraphEdge(ATTENTION, ROOT, EdgeKind.PARENT)
    with pytest.raises(ValueError, match="canonical child"):
        GraphEdge(ROOT, ATTENTION, EdgeKind.CHILD)
    with pytest.raises(ValueError, match="canonical endpoint ordering"):
        GraphEdge(ATTENTION, MLP, EdgeKind.COUPLED)


def test_graph_rejects_dangling_duplicate_and_unknown_constraint_references() -> None:
    with pytest.raises(ValueError, match="existing nodes"):
        ComponentGraph.build(
            (GraphNode(ROOT, "model"),),
            (GraphEdge(ROOT, LAYER, EdgeKind.CHILD),),
        )
    duplicate = GraphNode(ROOT, "model")
    with pytest.raises(ValueError, match="unique"):
        ComponentGraph.build((duplicate, duplicate))
    with pytest.raises(ValueError, match="existing string constraint_id"):
        ComponentGraph.build(
            _nodes(),
            (GraphEdge(ATTENTION, MLP, EdgeKind.CONSTRAINED),),
        )


def test_constraints_are_versioned_canonical_and_reference_graph_nodes() -> None:
    with pytest.raises(ValueError, match="canonical ordering"):
        MutationConstraint(
            constraint_id="reversed",
            kind=ConstraintKind.GROUPED_MUTATION,
            members=(ATTENTION, MLP),
        )
    unknown = MutationConstraint(
        constraint_id="unknown",
        kind=ConstraintKind.CUSTOM,
        members=(ComponentId.parse("model.missing"),),
    )
    with pytest.raises(ValueError, match="existing nodes"):
        ComponentGraph.build(_nodes(), constraints=(unknown,))

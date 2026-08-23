"""Tests for canonical versioned component graph persistence."""

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
    GraphProvenance,
    GraphSerializationError,
    MutationConstraint,
    dump_component_graph,
    load_component_graph,
)

ROOT = ComponentId.parse("model")
LEFT = ComponentId.parse("model.left")
RIGHT = ComponentId.parse("model.right")


def _graph() -> ComponentGraph:
    constraint = MutationConstraint(
        "pair",
        ConstraintKind.GROUPED_MUTATION,
        (LEFT, RIGHT),
    )
    return ComponentGraph.build(
        (
            GraphNode(RIGHT, "projection", (("width", 16),)),
            GraphNode(ROOT, "model"),
            GraphNode(LEFT, "projection", (("width", 16),)),
        ),
        (
            GraphEdge(LEFT, RIGHT, EdgeKind.COUPLED),
            GraphEdge(
                LEFT,
                RIGHT,
                EdgeKind.CONSTRAINED,
                attributes=(("constraint_id", "pair"),),
            ),
            GraphEdge(ROOT, LEFT, EdgeKind.CHILD),
            GraphEdge(LEFT, ROOT, EdgeKind.PARENT),
            GraphEdge(ROOT, RIGHT, EdgeKind.CHILD),
            GraphEdge(RIGHT, ROOT, EdgeKind.PARENT),
        ),
        (constraint,),
    )


def test_round_trip_preserves_graph_order_ids_edges_constraints_and_provenance() -> None:
    graph = _graph()
    provenance = GraphProvenance("llama-hf", "1", "a" * 40)

    serialized = dump_component_graph(graph, provenance)
    restored = load_component_graph(serialized)

    assert restored.graph == graph
    assert restored.provenance == provenance
    assert restored.canonical_json() == serialized
    assert " " not in serialized
    assert "\n" not in serialized


@pytest.mark.parametrize(
    ("path", "version_name"),
    [
        (("artifact_schema_version",), "graph artifact schema version"),
        (("graph", "schema_version"), "graph schema version"),
        (("graph", "edge_semantics_version"), "edge semantics version"),
        (("graph", "constraint_schema_version"), "constraint schema version"),
    ],
)
def test_unknown_schema_versions_are_rejected(
    path: tuple[str, ...],
    version_name: str,
) -> None:
    payload = json.loads(dump_component_graph(_graph(), GraphProvenance("hf", "1", "rev")))
    target = payload
    for segment in path[:-1]:
        target = target[segment]
    target[path[-1]] = 99

    with pytest.raises(GraphSerializationError, match=version_name):
        load_component_graph(json.dumps(payload))


def test_noncanonical_order_and_unknown_fields_are_rejected() -> None:
    payload = json.loads(dump_component_graph(_graph(), GraphProvenance("hf", "1", "rev")))
    payload["graph"]["nodes"].reverse()
    with pytest.raises(GraphSerializationError, match="canonical ordering"):
        load_component_graph(json.dumps(payload))

    payload = json.loads(dump_component_graph(_graph(), GraphProvenance("hf", "1", "rev")))
    payload["provenance"]["unexpected"] = True
    with pytest.raises(GraphSerializationError, match="unknown"):
        load_component_graph(json.dumps(payload))

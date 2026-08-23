"""Canonical JSON persistence for versioned component graphs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from modelsurgeon.graph.component_id import ComponentId
from modelsurgeon.graph.schema import (
    CONSTRAINT_SCHEMA_VERSION,
    EDGE_SEMANTICS_VERSION,
    GRAPH_SCHEMA_VERSION,
    ComponentGraph,
    ConstraintKind,
    EdgeKind,
    GraphEdge,
    GraphNode,
    GraphPrimitive,
    MutationConstraint,
)

GRAPH_ARTIFACT_SCHEMA_VERSION: Literal[1] = 1


class GraphSerializationError(ValueError):
    """Raised when a persisted graph is malformed or has an unknown version."""


@dataclass(frozen=True, slots=True)
class GraphProvenance:
    """Adapter and immutable model identity attached to a graph artifact."""

    adapter_name: str
    adapter_version: str
    model_revision: str

    def __post_init__(self) -> None:
        if not self.adapter_name or not self.adapter_version or not self.model_revision:
            raise ValueError("adapter name, adapter version, and model revision must be non-empty")

    def to_record(self) -> dict[str, str]:
        return {
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "model_revision": self.model_revision,
        }


@dataclass(frozen=True, slots=True)
class PersistedComponentGraph:
    """Reloaded graph together with its required provenance."""

    graph: ComponentGraph
    provenance: GraphProvenance
    artifact_schema_version: Literal[1] = GRAPH_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.artifact_schema_version != GRAPH_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported graph artifact schema version: {self.artifact_schema_version}"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "provenance": self.provenance.to_record(),
            "graph": self.graph.to_record(),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_record(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise GraphSerializationError(f"{location} must be a JSON object")
    return cast(Mapping[str, object], value)


def _array(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise GraphSerializationError(f"{location} must be a JSON array")
    return cast(list[object], value)


def _exact_keys(record: Mapping[str, object], expected: set[str], location: str) -> None:
    actual = set(record)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise GraphSerializationError(
            f"{location} keys do not match schema; missing={missing}, unknown={unknown}"
        )


def _string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise GraphSerializationError(f"{location} must be a non-empty string")
    return value


def _version(value: object, expected: int, location: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise GraphSerializationError(
            f"unsupported {location}: {value!r}; expected version {expected}"
        )


def _attributes(value: object, location: str) -> tuple[tuple[str, GraphPrimitive], ...]:
    record = _mapping(value, location)
    attributes: list[tuple[str, GraphPrimitive]] = []
    for key, item in sorted(record.items()):
        if not isinstance(item, str | int | float | bool | type(None)):
            raise GraphSerializationError(f"{location}.{key} must be a primitive value")
        attributes.append((key, item))
    return tuple(attributes)


def _node(value: object, index: int) -> GraphNode:
    location = f"graph.nodes[{index}]"
    record = _mapping(value, location)
    _exact_keys(record, {"component_id", "kind", "attributes"}, location)
    return GraphNode(
        ComponentId.parse(_string(record["component_id"], f"{location}.component_id")),
        _string(record["kind"], f"{location}.kind"),
        _attributes(record["attributes"], f"{location}.attributes"),
    )


def _edge(value: object, index: int) -> GraphEdge:
    location = f"graph.edges[{index}]"
    record = _mapping(value, location)
    _exact_keys(
        record,
        {"source", "target", "kind", "semantics_version", "attributes"},
        location,
    )
    _version(record["semantics_version"], EDGE_SEMANTICS_VERSION, "edge semantics version")
    try:
        kind = EdgeKind(_string(record["kind"], f"{location}.kind"))
    except ValueError as exc:
        raise GraphSerializationError(f"{location}.kind is unknown") from exc
    return GraphEdge(
        ComponentId.parse(_string(record["source"], f"{location}.source")),
        ComponentId.parse(_string(record["target"], f"{location}.target")),
        kind,
        attributes=_attributes(record["attributes"], f"{location}.attributes"),
    )


def _constraint(value: object, index: int) -> MutationConstraint:
    location = f"graph.constraints[{index}]"
    record = _mapping(value, location)
    _exact_keys(
        record,
        {"constraint_id", "kind", "members", "schema_version", "parameters"},
        location,
    )
    _version(record["schema_version"], CONSTRAINT_SCHEMA_VERSION, "constraint schema version")
    try:
        kind = ConstraintKind(_string(record["kind"], f"{location}.kind"))
    except ValueError as exc:
        raise GraphSerializationError(f"{location}.kind is unknown") from exc
    members = tuple(
        ComponentId.parse(_string(member, f"{location}.members[{member_index}]"))
        for member_index, member in enumerate(_array(record["members"], f"{location}.members"))
    )
    return MutationConstraint(
        _string(record["constraint_id"], f"{location}.constraint_id"),
        kind,
        members,
        parameters=_attributes(record["parameters"], f"{location}.parameters"),
    )


def dump_component_graph(graph: ComponentGraph, provenance: GraphProvenance) -> str:
    """Serialize a graph and adapter provenance as canonical UTF-8 JSON text."""
    canonical_graph = ComponentGraph.build(graph.nodes, graph.edges, graph.constraints)
    return PersistedComponentGraph(canonical_graph, provenance).canonical_json()


def load_component_graph(serialized: str | bytes) -> PersistedComponentGraph:
    """Strictly reload one canonical graph artifact and reject unknown schemas."""
    try:
        raw = json.loads(serialized)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GraphSerializationError("graph artifact is not valid UTF-8 JSON") from exc
    root = _mapping(raw, "artifact")
    _exact_keys(root, {"artifact_schema_version", "provenance", "graph"}, "artifact")
    _version(
        root["artifact_schema_version"],
        GRAPH_ARTIFACT_SCHEMA_VERSION,
        "graph artifact schema version",
    )

    provenance_record = _mapping(root["provenance"], "provenance")
    _exact_keys(
        provenance_record,
        {"adapter_name", "adapter_version", "model_revision"},
        "provenance",
    )
    provenance = GraphProvenance(
        _string(provenance_record["adapter_name"], "provenance.adapter_name"),
        _string(provenance_record["adapter_version"], "provenance.adapter_version"),
        _string(provenance_record["model_revision"], "provenance.model_revision"),
    )

    graph_record = _mapping(root["graph"], "graph")
    _exact_keys(
        graph_record,
        {
            "schema_version",
            "edge_semantics_version",
            "constraint_schema_version",
            "nodes",
            "edges",
            "constraints",
        },
        "graph",
    )
    _version(graph_record["schema_version"], GRAPH_SCHEMA_VERSION, "graph schema version")
    _version(
        graph_record["edge_semantics_version"],
        EDGE_SEMANTICS_VERSION,
        "edge semantics version",
    )
    _version(
        graph_record["constraint_schema_version"],
        CONSTRAINT_SCHEMA_VERSION,
        "constraint schema version",
    )
    nodes = tuple(
        _node(value, index)
        for index, value in enumerate(_array(graph_record["nodes"], "graph.nodes"))
    )
    edges = tuple(
        _edge(value, index)
        for index, value in enumerate(_array(graph_record["edges"], "graph.edges"))
    )
    constraints = tuple(
        _constraint(value, index)
        for index, value in enumerate(
            _array(graph_record["constraints"], "graph.constraints")
        )
    )
    graph = ComponentGraph(nodes, edges, constraints)
    canonical = ComponentGraph.build(nodes, edges, constraints)
    if graph.to_record() != canonical.to_record():
        raise GraphSerializationError("graph records are not in canonical ordering")
    return PersistedComponentGraph(graph, provenance)

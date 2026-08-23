"""Versioned, framework-neutral component dependency and coupling graph schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from modelsurgeon.graph.component_id import ComponentId

type GraphPrimitive = str | int | float | bool | None

GRAPH_SCHEMA_VERSION: Literal[1] = 1
EDGE_SEMANTICS_VERSION: Literal[1] = 1
CONSTRAINT_SCHEMA_VERSION: Literal[1] = 1


class EdgeKind(StrEnum):
    """Directed edge meanings fixed by ``EDGE_SEMANTICS_VERSION``."""

    PARENT = "parent"
    CHILD = "child"
    CONSUMES = "consumes"
    PRODUCES = "produces"
    COUPLED = "coupled"
    CONSTRAINED = "constrained"


class ConstraintKind(StrEnum):
    """Stable mutation constraint categories."""

    GROUPED_MUTATION = "grouped_mutation"
    SAME_HIDDEN_SIZE = "same_hidden_size"
    SAME_HEAD_SET = "same_head_set"
    SHAPE_EQUALITY = "shape_equality"
    CUSTOM = "custom"


def _validate_attributes(attributes: tuple[tuple[str, GraphPrimitive], ...]) -> None:
    keys = [key for key, _ in attributes]
    if any(not key for key in keys):
        raise ValueError("attribute keys must be non-empty")
    if len(keys) != len(set(keys)):
        raise ValueError("attribute keys must be unique")


@dataclass(frozen=True, slots=True)
class GraphNode:
    """Persisted component node with no model or framework object."""

    component_id: ComponentId
    kind: str
    attributes: tuple[tuple[str, GraphPrimitive], ...] = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("node kind must be non-empty")
        _validate_attributes(self.attributes)

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "kind": self.kind,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """One versioned relationship between canonical component nodes."""

    source: ComponentId
    target: ComponentId
    kind: EdgeKind
    semantics_version: Literal[1] = EDGE_SEMANTICS_VERSION
    attributes: tuple[tuple[str, GraphPrimitive], ...] = ()

    def __post_init__(self) -> None:
        if self.semantics_version != EDGE_SEMANTICS_VERSION:
            raise ValueError(f"unsupported edge semantics version: {self.semantics_version}")
        if self.source == self.target:
            raise ValueError("graph edges cannot be self-referential")
        if self.kind is EdgeKind.PARENT and self.source.parent != self.target:
            raise ValueError("parent edge target must be the source's canonical parent")
        if self.kind is EdgeKind.CHILD and self.target.parent != self.source:
            raise ValueError("child edge target must be a canonical child of the source")
        if self.kind is EdgeKind.COUPLED and self.target < self.source:
            raise ValueError("coupled edges must use canonical endpoint ordering")
        _validate_attributes(self.attributes)

    def to_record(self) -> dict[str, object]:
        return {
            "source": str(self.source),
            "target": str(self.target),
            "kind": self.kind.value,
            "semantics_version": self.semantics_version,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class MutationConstraint:
    """Versioned rule requiring a component set to mutate consistently."""

    constraint_id: str
    kind: ConstraintKind
    members: tuple[ComponentId, ...]
    schema_version: Literal[1] = CONSTRAINT_SCHEMA_VERSION
    parameters: tuple[tuple[str, GraphPrimitive], ...] = ()

    def __post_init__(self) -> None:
        if not self.constraint_id:
            raise ValueError("constraint_id must be non-empty")
        if self.schema_version != CONSTRAINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported constraint schema version: {self.schema_version}")
        if not self.members:
            raise ValueError("mutation constraints require at least one member")
        if len(self.members) != len(set(self.members)):
            raise ValueError("mutation constraint members must be unique")
        if tuple(sorted(self.members)) != self.members:
            raise ValueError("mutation constraint members must use canonical ordering")
        _validate_attributes(self.parameters)

    def to_record(self) -> dict[str, object]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind.value,
            "members": [str(member) for member in self.members],
            "schema_version": self.schema_version,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class ComponentGraph:
    """Serializable graph record with canonical ordering and integrity checks."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...] = ()
    constraints: tuple[MutationConstraint, ...] = ()
    schema_version: Literal[1] = GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GRAPH_SCHEMA_VERSION:
            raise ValueError(f"unsupported graph schema version: {self.schema_version}")
        node_ids = [node.component_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node component IDs must be unique")
        known_nodes = set(node_ids)
        edge_keys = [(edge.kind, edge.source, edge.target) for edge in self.edges]
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("graph edges must be unique by kind and endpoints")
        for edge in self.edges:
            if edge.source not in known_nodes or edge.target not in known_nodes:
                raise ValueError("graph edge endpoints must reference existing nodes")
        constraint_ids = [constraint.constraint_id for constraint in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint IDs must be unique")
        for constraint in self.constraints:
            if any(member not in known_nodes for member in constraint.members):
                raise ValueError("constraint members must reference existing nodes")
        known_constraints = set(constraint_ids)
        for edge in self.edges:
            if edge.kind is not EdgeKind.CONSTRAINED:
                continue
            constraint_id = dict(edge.attributes).get("constraint_id")
            if constraint_id not in known_constraints:
                raise ValueError(
                    "constrained edges require an existing string constraint_id attribute"
                )

    @classmethod
    def build(
        cls,
        nodes: tuple[GraphNode, ...],
        edges: tuple[GraphEdge, ...] = (),
        constraints: tuple[MutationConstraint, ...] = (),
    ) -> ComponentGraph:
        """Build a graph with deterministic record ordering."""
        return cls(
            nodes=tuple(sorted(nodes, key=lambda node: node.component_id)),
            edges=tuple(
                sorted(
                    edges,
                    key=lambda edge: (edge.kind.value, edge.source, edge.target),
                )
            ),
            constraints=tuple(sorted(constraints, key=lambda item: item.constraint_id)),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "edge_semantics_version": EDGE_SEMANTICS_VERSION,
            "constraint_schema_version": CONSTRAINT_SCHEMA_VERSION,
            "nodes": [node.to_record() for node in self.nodes],
            "edges": [edge.to_record() for edge in self.edges],
            "constraints": [constraint.to_record() for constraint in self.constraints],
        }

"""Build a canonical transformer dependency graph from discovery records."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from typing import Protocol

from modelsurgeon.graph.component_id import ComponentId
from modelsurgeon.graph.schema import (
    ComponentGraph,
    ConstraintKind,
    EdgeKind,
    GraphEdge,
    GraphNode,
    GraphPrimitive,
    MutationConstraint,
)


class GraphBuildError(ValueError):
    """Raised when discovery records cannot form a consistent supported graph."""


class ComponentRecordLike(Protocol):
    component_id: ComponentId
    kind: str
    attributes: tuple[tuple[str, GraphPrimitive], ...]


_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _projection_group(
    nodes: dict[ComponentId, GraphNode],
    parent: ComponentId,
    names: tuple[str, ...],
    label: str,
) -> tuple[ComponentId, ...]:
    members = tuple(parent.child(name) for name in names)
    present = tuple(member for member in members if member in nodes)
    if present and len(present) != len(members):
        missing = ", ".join(str(member) for member in members if member not in nodes)
        raise GraphBuildError(f"{label} projection group is incomplete; missing: {missing}")
    return present


def _constraint_edges(
    members: tuple[ComponentId, ...],
    constraint_id: str,
) -> Iterable[GraphEdge]:
    for left, right in combinations(sorted(members), 2):
        yield GraphEdge(left, right, EdgeKind.COUPLED)
        yield GraphEdge(
            left,
            right,
            EdgeKind.CONSTRAINED,
            attributes=(("constraint_id", constraint_id),),
        )


def _flow_edges(upstream: ComponentId, downstream: ComponentId) -> tuple[GraphEdge, GraphEdge]:
    return (
        GraphEdge(downstream, upstream, EdgeKind.CONSUMES),
        GraphEdge(upstream, downstream, EdgeKind.PRODUCES),
    )


def _add_edge(
    edges: dict[tuple[EdgeKind, ComponentId, ComponentId], GraphEdge],
    edge: GraphEdge,
) -> None:
    key = (edge.kind, edge.source, edge.target)
    existing = edges.get(key)
    if existing is not None and existing != edge:
        raise GraphBuildError(
            f"conflicting duplicate edge: {edge.kind.value} {edge.source} -> {edge.target}"
        )
    edges[key] = edge


def build_component_graph(records: Iterable[ComponentRecordLike]) -> ComponentGraph:
    """Construct hierarchy, transformer dataflow, coupling, and constraints."""
    nodes: dict[ComponentId, GraphNode] = {}
    for record in records:
        node = GraphNode(record.component_id, record.kind, record.attributes)
        if node.component_id in nodes:
            raise GraphBuildError(f"duplicate component ID: {node.component_id}")
        nodes[node.component_id] = node

    if ComponentId.parse("model") not in nodes:
        raise GraphBuildError("discovery records must contain the canonical model root")

    edges: dict[tuple[EdgeKind, ComponentId, ComponentId], GraphEdge] = {}
    constraints: list[MutationConstraint] = []
    layer_ids = sorted(
        node.component_id for node in nodes.values() if node.kind == "transformer_layer"
    )

    for layer_id in layer_ids:
        attention = layer_id.child("self_attn")
        mlp = layer_id.child("mlp")
        attention_members = _projection_group(
            nodes,
            attention,
            _ATTENTION_PROJECTIONS,
            "attention",
        )
        mlp_members = _projection_group(nodes, mlp, _MLP_PROJECTIONS, "MLP")
        if bool(attention_members) != bool(mlp_members):
            raise GraphBuildError(
                "transformer block must expose both attention and MLP projection groups"
            )

        if attention_members:
            constraint_id = f"{layer_id}:attention-projections:v1"
            constraints.append(
                MutationConstraint(
                    constraint_id=constraint_id,
                    kind=ConstraintKind.SAME_HEAD_SET,
                    members=tuple(sorted(attention_members)),
                )
            )
            for edge in _constraint_edges(attention_members, constraint_id):
                _add_edge(edges, edge)

        if mlp_members:
            constraint_id = f"{layer_id}:mlp-projections:v1"
            constraints.append(
                MutationConstraint(
                    constraint_id=constraint_id,
                    kind=ConstraintKind.SAME_HIDDEN_SIZE,
                    members=tuple(sorted(mlp_members)),
                )
            )
            for edge in _constraint_edges(mlp_members, constraint_id):
                _add_edge(edges, edge)

        required_flow = (*attention_members, *mlp_members)
        if required_flow:
            input_norm = layer_id.child("input_layernorm")
            post_norm = layer_id.child("post_attention_layernorm")
            missing_norms = [item for item in (input_norm, post_norm) if item not in nodes]
            if missing_norms:
                missing = ", ".join(str(item) for item in missing_norms)
                raise GraphBuildError(f"transformer layer norms are incomplete; missing: {missing}")

            residual_attention = layer_id.child("residual_attn")
            residual_mlp = layer_id.child("residual_mlp")
            if residual_attention in nodes or residual_mlp in nodes:
                raise GraphBuildError("synthetic residual component ID collides with discovery")
            nodes[residual_attention] = GraphNode(residual_attention, "residual_path")
            nodes[residual_mlp] = GraphNode(residual_mlp, "residual_path")

            q_proj, k_proj, v_proj, o_proj = attention_members
            gate_proj, up_proj, down_proj = mlp_members
            for target in (q_proj, k_proj, v_proj):
                for edge in _flow_edges(input_norm, target):
                    _add_edge(edges, edge)
            for source in (q_proj, k_proj, v_proj):
                for edge in _flow_edges(source, o_proj):
                    _add_edge(edges, edge)
            for edge in _flow_edges(o_proj, residual_attention):
                _add_edge(edges, edge)
            for edge in _flow_edges(residual_attention, post_norm):
                _add_edge(edges, edge)
            for target in (gate_proj, up_proj):
                for edge in _flow_edges(post_norm, target):
                    _add_edge(edges, edge)
            for source in (gate_proj, up_proj):
                for edge in _flow_edges(source, down_proj):
                    _add_edge(edges, edge)
            for edge in _flow_edges(down_proj, residual_mlp):
                _add_edge(edges, edge)

    for component_id in sorted(nodes):
        parent = component_id.parent
        if parent is None:
            continue
        if parent not in nodes:
            raise GraphBuildError(f"component {component_id} has missing parent {parent}")
        _add_edge(edges, GraphEdge(component_id, parent, EdgeKind.PARENT))
        _add_edge(edges, GraphEdge(parent, component_id, EdgeKind.CHILD))

    return ComponentGraph.build(
        tuple(nodes.values()),
        tuple(edges.values()),
        tuple(constraints),
    )

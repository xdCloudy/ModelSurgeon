"""Exact invariant validation for component graphs and raw graph records."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

from modelsurgeon.graph.component_id import ComponentId
from modelsurgeon.graph.schema import (
    ComponentGraph,
    EdgeKind,
    GraphEdge,
    GraphNode,
    MutationConstraint,
)


class GraphValidationRule(StrEnum):
    DANGLING_EDGE = "dangling_edge"
    DANGLING_CONSTRAINT_MEMBER = "dangling_constraint_member"
    HIERARCHY_CYCLE = "hierarchy_cycle"
    DATAFLOW_CYCLE = "dataflow_cycle"
    MISSING_RECIPROCAL_EDGE = "missing_reciprocal_edge"
    INCOMPLETE_COUPLING_CLOSURE = "incomplete_coupling_closure"
    UNKNOWN_CONSTRAINT = "unknown_constraint"
    CONSTRAINT_MEMBERSHIP_MISMATCH = "constraint_membership_mismatch"


@dataclass(frozen=True, slots=True)
class GraphViolation:
    """One deterministic rule violation naming every involved component."""

    rule: GraphValidationRule
    nodes: tuple[ComponentId, ...]
    detail: str

    def to_record(self) -> dict[str, object]:
        return {
            "rule": self.rule.value,
            "nodes": [str(node) for node in self.nodes],
            "detail": self.detail,
        }


class GraphValidationError(ValueError):
    """Raised on demand for a complete validation report."""

    def __init__(self, violations: tuple[GraphViolation, ...]) -> None:
        summary = "; ".join(
            f"{violation.rule.value}({', '.join(map(str, violation.nodes))}): "
            f"{violation.detail}"
            for violation in violations
        )
        super().__init__(summary)
        self.violations = violations


@dataclass(frozen=True, slots=True)
class GraphValidationReport:
    """Complete validation result; callers choose whether to raise."""

    violations: tuple[GraphViolation, ...]

    @property
    def valid(self) -> bool:
        return not self.violations

    def raise_for_errors(self) -> None:
        if self.violations:
            raise GraphValidationError(self.violations)

    def to_record(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "violations": [violation.to_record() for violation in self.violations],
        }


def _cycle_nodes(
    edges: Iterable[GraphEdge],
    kind: EdgeKind,
) -> tuple[ComponentId, ...]:
    adjacency: dict[ComponentId, set[ComponentId]] = {}
    reverse: dict[ComponentId, set[ComponentId]] = {}
    for edge in edges:
        if edge.kind is kind:
            adjacency.setdefault(edge.source, set()).add(edge.target)
            reverse.setdefault(edge.target, set()).add(edge.source)
            adjacency.setdefault(edge.target, set())
            reverse.setdefault(edge.source, set())

    visited: set[ComponentId] = set()
    finish_order: list[ComponentId] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack: list[tuple[ComponentId, bool]] = [(start, False)]
        while stack:
            node, finishing = stack.pop()
            if finishing:
                finish_order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            stack.extend(
                (target, False)
                for target in sorted(adjacency[node], reverse=True)
                if target not in visited
            )

    assigned: set[ComponentId] = set()
    cycle_members: set[ComponentId] = set()
    for start in reversed(finish_order):
        if start in assigned:
            continue
        component: set[ComponentId] = set()
        stack = [(start, False)]
        while stack:
            node, _ = stack.pop()
            if node in assigned:
                continue
            assigned.add(node)
            component.add(node)
            stack.extend((source, False) for source in reverse[node] if source not in assigned)
        if len(component) > 1 or any(node in adjacency[node] for node in component):
            cycle_members.update(component)
    return tuple(sorted(cycle_members))


def _reciprocal(kind: EdgeKind) -> EdgeKind | None:
    return {
        EdgeKind.PARENT: EdgeKind.CHILD,
        EdgeKind.CHILD: EdgeKind.PARENT,
        EdgeKind.CONSUMES: EdgeKind.PRODUCES,
        EdgeKind.PRODUCES: EdgeKind.CONSUMES,
    }.get(kind)


def validate_graph_records(
    nodes: Iterable[GraphNode],
    edges: Iterable[GraphEdge],
    constraints: Iterable[MutationConstraint] = (),
) -> GraphValidationReport:
    """Validate raw records, including states rejected by ``ComponentGraph``."""
    node_tuple = tuple(nodes)
    edge_tuple = tuple(edges)
    constraint_tuple = tuple(constraints)
    node_ids = {node.component_id for node in node_tuple}
    edge_keys = {(edge.kind, edge.source, edge.target) for edge in edge_tuple}
    constraint_by_id = {constraint.constraint_id: constraint for constraint in constraint_tuple}
    violations: list[GraphViolation] = []

    for edge in edge_tuple:
        missing = tuple(sorted({edge.source, edge.target} - node_ids))
        if missing:
            violations.append(
                GraphViolation(
                    GraphValidationRule.DANGLING_EDGE,
                    missing,
                    f"{edge.kind.value} edge references nodes absent from the graph",
                )
            )
        reciprocal = _reciprocal(edge.kind)
        if reciprocal is not None and (reciprocal, edge.target, edge.source) not in edge_keys:
            violations.append(
                GraphViolation(
                    GraphValidationRule.MISSING_RECIPROCAL_EDGE,
                    tuple(sorted((edge.source, edge.target))),
                    f"{edge.kind.value} edge lacks inverse {reciprocal.value} edge",
                )
            )

    for constraint in constraint_tuple:
        missing = tuple(sorted(set(constraint.members) - node_ids))
        if missing:
            violations.append(
                GraphViolation(
                    GraphValidationRule.DANGLING_CONSTRAINT_MEMBER,
                    missing,
                    f"constraint {constraint.constraint_id!r} references absent nodes",
                )
            )
        for left, right in combinations(constraint.members, 2):
            coupled_key = (EdgeKind.COUPLED, left, right)
            constrained = next(
                (
                    edge
                    for edge in edge_tuple
                    if edge.kind is EdgeKind.CONSTRAINED
                    and edge.source == left
                    and edge.target == right
                    and dict(edge.attributes).get("constraint_id") == constraint.constraint_id
                ),
                None,
            )
            if coupled_key not in edge_keys or constrained is None:
                violations.append(
                    GraphViolation(
                        GraphValidationRule.INCOMPLETE_COUPLING_CLOSURE,
                        (left, right),
                        f"constraint {constraint.constraint_id!r} lacks coupled/constrained edges",
                    )
                )

    for edge in edge_tuple:
        if edge.kind is not EdgeKind.CONSTRAINED:
            continue
        constraint_id = dict(edge.attributes).get("constraint_id")
        if not isinstance(constraint_id, str) or constraint_id not in constraint_by_id:
            violations.append(
                GraphViolation(
                    GraphValidationRule.UNKNOWN_CONSTRAINT,
                    tuple(sorted((edge.source, edge.target))),
                    f"constrained edge references unknown constraint {constraint_id!r}",
                )
            )
            continue
        members = set(constraint_by_id[constraint_id].members)
        if edge.source not in members or edge.target not in members:
            violations.append(
                GraphViolation(
                    GraphValidationRule.CONSTRAINT_MEMBERSHIP_MISMATCH,
                    tuple(sorted((edge.source, edge.target))),
                    f"edge endpoints are not both members of constraint {constraint_id!r}",
                )
            )

    for kind, rule in (
        (EdgeKind.CHILD, GraphValidationRule.HIERARCHY_CYCLE),
        (EdgeKind.PRODUCES, GraphValidationRule.DATAFLOW_CYCLE),
    ):
        cycle_nodes = _cycle_nodes(edge_tuple, kind)
        if cycle_nodes:
            violations.append(
                GraphViolation(rule, cycle_nodes, f"{kind.value} edges contain a directed cycle")
            )

    ordered = tuple(
        sorted(
            violations,
            key=lambda item: (
                item.rule.value,
                tuple(str(node) for node in item.nodes),
                item.detail,
            ),
        )
    )
    return GraphValidationReport(ordered)


def validate_component_graph(graph: ComponentGraph) -> GraphValidationReport:
    """Validate all cross-record invariants of a constructed component graph."""
    return validate_graph_records(graph.nodes, graph.edges, graph.constraints)

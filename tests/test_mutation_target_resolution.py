"""Tests for deterministic transitive mutation coupling closure."""

from __future__ import annotations

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
from modelsurgeon.surgery import (
    MutationDelta,
    MutationKind,
    MutationPrecondition,
    MutationRequest,
    MutationTargetResolutionError,
    resolve_mutation_targets,
)
from modelsurgeon.surgery.target_resolution import MutationTargetResolver

A = ComponentId.parse("model.a")
B = ComponentId.parse("model.b")
C = ComponentId.parse("model.c")
D = ComponentId.parse("model.d")


def _pair_edges(left: ComponentId, right: ComponentId, constraint_id: str) -> tuple[GraphEdge, ...]:
    return (
        GraphEdge(left, right, EdgeKind.COUPLED),
        GraphEdge(
            left,
            right,
            EdgeKind.CONSTRAINED,
            attributes=(("constraint_id", constraint_id),),
        ),
    )


def _graph() -> ComponentGraph:
    first = MutationConstraint("a-b", ConstraintKind.SAME_HEAD_SET, (A, B))
    second = MutationConstraint("b-c", ConstraintKind.SAME_HIDDEN_SIZE, (B, C))
    return ComponentGraph.build(
        tuple(GraphNode(item, "module") for item in (A, B, C, D)),
        (*_pair_edges(A, B, "a-b"), *_pair_edges(B, C, "b-c")),
        (first, second),
    )


def test_overlapping_constraints_resolve_transitive_closure_with_reasons() -> None:
    request = MutationRequest(MutationKind.REMOVE, (A,))
    resolved = resolve_mutation_targets(request, _graph())

    assert resolved.affected_components == (A, B, C)
    assert resolved.constraint_ids == ("a-b", "b-c")
    assert resolved.targets[0].requested is True
    assert resolved.targets[2].requested is False
    assert "constraint:b-c:same_hidden_size" in resolved.targets[2].reasons
    assert resolved.to_record()["mutation_id"] == request.mutation_id


def test_resolved_closure_builds_complete_base_mutation_plan() -> None:
    resolved = resolve_mutation_targets(MutationRequest(MutationKind.MASK, (B,)), _graph())
    plan = resolved.to_plan(
        preconditions=(MutationPrecondition("revision", "abc"),),
        expected_delta=MutationDelta(memory_bytes=-10),
    )
    assert plan.affected_components == (A, B, C)
    assert plan.expected_delta.memory_bytes == -10


def test_unknown_target_fails_before_planning() -> None:
    unknown = ComponentId.parse("model.unknown")
    with pytest.raises(MutationTargetResolutionError, match=r"model\.unknown"):
        resolve_mutation_targets(MutationRequest(MutationKind.REMOVE, (unknown,)), _graph())


def test_invalid_incomplete_constraint_graph_is_rejected() -> None:
    constraint = MutationConstraint("a-b", ConstraintKind.GROUPED_MUTATION, (A, B))
    invalid = ComponentGraph.build(
        (GraphNode(A, "module"), GraphNode(B, "module")), constraints=(constraint,)
    )
    with pytest.raises(MutationTargetResolutionError, match=r"invalid.*incomplete"):
        resolve_mutation_targets(MutationRequest(MutationKind.REMOVE, (A,)), invalid)


def test_unconstrained_coupled_edge_is_still_closed() -> None:
    graph = ComponentGraph.build(
        (GraphNode(A, "module"), GraphNode(B, "module")),
        (GraphEdge(A, B, EdgeKind.COUPLED),),
    )
    resolved = resolve_mutation_targets(MutationRequest(MutationKind.REMOVE, (B,)), graph)
    assert resolved.affected_components == (A, B)
    assert any(reason.startswith("coupled:") for reason in resolved.targets[0].reasons)


def test_indexed_resolver_reuses_one_validated_graph_for_multiple_requests() -> None:
    resolver = MutationTargetResolver(_graph())

    assert resolver.resolve(MutationRequest(MutationKind.REMOVE, (A,))).affected_components == (
        A,
        B,
        C,
    )
    assert resolver.resolve(MutationRequest(MutationKind.MASK, (D,))).affected_components == (D,)

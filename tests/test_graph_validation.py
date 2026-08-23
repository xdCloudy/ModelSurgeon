"""Tests for graph invariant and mutation-closure validation."""

from __future__ import annotations

import pytest

from modelsurgeon.graph import (
    ComponentGraph,
    ComponentId,
    ConstraintKind,
    EdgeKind,
    GraphEdge,
    GraphNode,
    GraphValidationError,
    GraphValidationRule,
    MutationConstraint,
    validate_component_graph,
    validate_graph_records,
)

ROOT = ComponentId.parse("model")
LEFT = ComponentId.parse("model.left")
RIGHT = ComponentId.parse("model.right")


def _nodes() -> tuple[GraphNode, ...]:
    return (GraphNode(ROOT, "model"), GraphNode(LEFT, "module"), GraphNode(RIGHT, "module"))


def _valid_graph() -> ComponentGraph:
    constraint = MutationConstraint(
        "pair",
        ConstraintKind.GROUPED_MUTATION,
        (LEFT, RIGHT),
    )
    return ComponentGraph.build(
        _nodes(),
        (
            GraphEdge(ROOT, LEFT, EdgeKind.CHILD),
            GraphEdge(LEFT, ROOT, EdgeKind.PARENT),
            GraphEdge(ROOT, RIGHT, EdgeKind.CHILD),
            GraphEdge(RIGHT, ROOT, EdgeKind.PARENT),
            GraphEdge(LEFT, RIGHT, EdgeKind.PRODUCES),
            GraphEdge(RIGHT, LEFT, EdgeKind.CONSUMES),
            GraphEdge(LEFT, RIGHT, EdgeKind.COUPLED),
            GraphEdge(
                LEFT,
                RIGHT,
                EdgeKind.CONSTRAINED,
                attributes=(("constraint_id", "pair"),),
            ),
        ),
        (constraint,),
    )


def test_supported_tiny_graph_validates_successfully() -> None:
    report = validate_component_graph(_valid_graph())

    assert report.valid
    assert report.to_record() == {"valid": True, "violations": []}
    report.raise_for_errors()


def test_dangling_and_missing_reciprocal_edges_name_exact_nodes() -> None:
    missing = ComponentId.parse("model.missing")
    report = validate_graph_records(
        _nodes(),
        (GraphEdge(ROOT, missing, EdgeKind.CHILD),),
    )

    rules = {violation.rule for violation in report.violations}
    assert GraphValidationRule.DANGLING_EDGE in rules
    assert GraphValidationRule.MISSING_RECIPROCAL_EDGE in rules
    dangling = next(
        item for item in report.violations if item.rule is GraphValidationRule.DANGLING_EDGE
    )
    assert dangling.nodes == (missing,)
    with pytest.raises(GraphValidationError, match=r"model\.missing"):
        report.raise_for_errors()


def test_dataflow_cycles_report_exact_cycle_members() -> None:
    dataflow = (
        GraphEdge(LEFT, RIGHT, EdgeKind.PRODUCES),
        GraphEdge(RIGHT, LEFT, EdgeKind.PRODUCES),
    )

    report = validate_graph_records(_nodes(), dataflow)

    dataflow_error = next(
        item for item in report.violations if item.rule is GraphValidationRule.DATAFLOW_CYCLE
    )
    assert set(dataflow_error.nodes) == {LEFT, RIGHT}


def test_incomplete_mutation_closure_reports_missing_pair() -> None:
    constraint = MutationConstraint(
        "pair",
        ConstraintKind.SAME_HIDDEN_SIZE,
        (LEFT, RIGHT),
    )

    report = validate_graph_records(_nodes(), (), (constraint,))

    violation = next(
        item
        for item in report.violations
        if item.rule is GraphValidationRule.INCOMPLETE_COUPLING_CLOSURE
    )
    assert violation.nodes == (LEFT, RIGHT)
    assert "pair" in violation.detail


def test_constrained_edge_must_reference_matching_constraint_members() -> None:
    other = MutationConstraint(
        "root-left",
        ConstraintKind.CUSTOM,
        (ROOT, LEFT),
    )
    edges = (
        GraphEdge(
            LEFT,
            RIGHT,
            EdgeKind.CONSTRAINED,
            attributes=(("constraint_id", "root-left"),),
        ),
    )

    report = validate_graph_records(_nodes(), edges, (other,))

    assert GraphValidationRule.CONSTRAINT_MEMBERSHIP_MISMATCH in {
        item.rule for item in report.violations
    }

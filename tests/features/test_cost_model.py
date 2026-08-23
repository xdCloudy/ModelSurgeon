import pytest

from modelsurgeon.features.cost_model import (
    ComponentCostSpec,
    CostAssumptions,
    CostModelError,
    CostOperation,
    SequenceShape,
    estimate_component_costs,
    graph_parameter_count,
)
from modelsurgeon.graph import ComponentGraph, ComponentId, GraphNode


def _linear_graph() -> ComponentGraph:
    projection = ComponentId.parse("model.proj")
    weight = projection.child("weight")
    bias = projection.child("bias")
    return ComponentGraph.build(
        (
            GraphNode(ComponentId.parse("model"), "model"),
            GraphNode(projection, "projection"),
            GraphNode(weight, "parameter", (("element_count", 12),)),
            GraphNode(bias, "parameter", (("element_count", 4),)),
        )
    )


def test_linear_costs_and_parameter_totals_reconcile_exactly() -> None:
    graph = _linear_graph()
    projection = ComponentId.parse("model.proj")
    report = estimate_component_costs(
        graph,
        (
            ComponentCostSpec(
                projection,
                CostOperation.LINEAR,
                parameter_count=16,
                input_width=3,
                output_width=4,
            ),
        ),
        SequenceShape(batch_size=2, sequence_length=5),
    )

    assert graph_parameter_count(graph) == 16
    assert report.graph_parameter_count == 16
    assert report.specified_parameter_count == 16
    estimate = report.estimates[0]
    assert estimate.forward_flops == 240
    assert estimate.output_activation_bytes == 80
    assert estimate.peak_working_activation_bytes == 80
    assert report.to_record()["parameter_counts_reconciled"] is True


def test_parameter_mismatch_fails_closed() -> None:
    graph = _linear_graph()
    projection = ComponentId.parse("model.proj")

    with pytest.raises(CostModelError, match="do not reconcile"):
        estimate_component_costs(
            graph,
            (
                ComponentCostSpec(
                    projection,
                    CostOperation.LINEAR,
                    parameter_count=15,
                    input_width=3,
                    output_width=4,
                ),
            ),
            SequenceShape(1, 1),
        )


def test_gated_mlp_and_attention_emit_memory_assumptions() -> None:
    model = ComponentId.parse("model")
    mlp = ComponentId.parse("model.mlp")
    attention = ComponentId.parse("model.attn")
    p0 = ComponentId.parse("model.p0")
    p1 = ComponentId.parse("model.p1")
    graph = ComponentGraph.build(
        (
            GraphNode(model, "model"),
            GraphNode(mlp, "mlp"),
            GraphNode(attention, "attention"),
            GraphNode(p0, "parameter", (("element_count", 96),)),
            GraphNode(p1, "parameter", (("element_count", 64),)),
        )
    )
    assumptions = CostAssumptions(activation_bytes_per_element=4)
    report = estimate_component_costs(
        graph,
        (
            ComponentCostSpec(
                mlp,
                CostOperation.GATED_MLP,
                parameter_count=96,
                input_width=4,
                output_width=4,
                intermediate_width=8,
            ),
            ComponentCostSpec(
                attention,
                CostOperation.ATTENTION,
                parameter_count=64,
                input_width=4,
                output_width=4,
                heads=2,
                head_dim=2,
            ),
        ),
        SequenceShape(1, 3),
        assumptions,
    )

    by_id = {estimate.component_id: estimate for estimate in report.estimates}
    assert by_id[mlp].forward_flops == 576
    assert by_id[mlp].peak_working_activation_bytes == 240
    assert by_id[attention].forward_flops == 264
    assert by_id[attention].peak_working_activation_bytes == 264
    assert report.to_record()["assumptions"] == assumptions.to_record()


def test_parameter_nodes_require_valid_element_counts() -> None:
    graph = ComponentGraph.build(
        (
            GraphNode(ComponentId.parse("model"), "model"),
            GraphNode(ComponentId.parse("model.bad"), "parameter"),
        )
    )

    with pytest.raises(CostModelError, match="element_count"):
        graph_parameter_count(graph)

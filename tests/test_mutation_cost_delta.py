"""Tests for analytical mutation parameter, FLOP, and activation-memory deltas."""

from __future__ import annotations

import pytest

from modelsurgeon.features.cost_model import (
    ComponentCostEstimate,
    CostAssumptions,
    CostModelReport,
    CostOperation,
    SequenceShape,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.contracts import (
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationRequest,
)
from modelsurgeon.surgery.cost_delta import (
    AffectedShape,
    MutationCostDeltaError,
    attach_expected_cost_delta,
)


def _component() -> ComponentId:
    return ComponentId.parse("model.layers.0.mlp")


def _plan() -> MutationPlan:
    component = _component()
    return MutationPlan(
        MutationRequest(MutationKind.REMOVE, (component,)),
        (component,),
        (),
        MutationDelta(storage_bytes=-16),
    )


def _report(parameters: int, flops: int, output_bytes: int, peak_bytes: int) -> CostModelReport:
    component = _component()
    estimate = ComponentCostEstimate(
        component,
        CostOperation.GATED_MLP,
        parameters,
        flops,
        output_bytes,
        peak_bytes,
    )
    return CostModelReport(
        "1",
        SequenceShape(1, 8),
        CostAssumptions(),
        parameters,
        parameters,
        (estimate,),
    )


def test_physical_delta_is_candidate_minus_baseline_and_preserves_storage_delta() -> None:
    report = attach_expected_cost_delta(
        _plan(),
        _report(100, 1000, 200, 500),
        candidate=_report(80, 760, 160, 420),
        affected_shapes=(AffectedShape(_component(), (10, 10), (8, 10)),),
    )

    assert report.parameter_delta == -20
    assert report.forward_flop_delta == -240
    assert report.output_activation_delta_bytes == -40
    assert report.peak_working_activation_delta_bytes == -80
    assert report.plan.expected_delta == MutationDelta(-20, -240, -40, -16)
    record = report.to_record()
    assert dict(report.assumptions)["delta_sign"] == "candidate_minus_baseline"
    assert record["affected_shapes"] == [
        {
            "component_id": "model.layers.0.mlp",
            "old_shape": [10, 10],
            "new_shape": [8, 10],
        }
    ]


def test_mask_only_reports_zero_physical_reduction() -> None:
    report = attach_expected_cost_delta(
        _plan(),
        _report(100, 1000, 200, 500),
        affected_shapes=(AffectedShape(_component(), (10, 10), (10, 10)),),
        mask_only=True,
    )

    assert report.mask_only
    assert report.plan.expected_delta == MutationDelta()
    assert report.parameter_delta == 0
    assert report.forward_flop_delta == 0
    assert report.output_activation_delta_bytes == 0
    assert report.peak_working_activation_delta_bytes == 0


def test_physical_delta_requires_candidate_and_comparable_context() -> None:
    shape = (AffectedShape(_component(), (10, 10), (8, 10)),)
    baseline = _report(100, 1000, 200, 500)
    with pytest.raises(MutationCostDeltaError, match="candidate"):
        attach_expected_cost_delta(_plan(), baseline, affected_shapes=shape)

    candidate = CostModelReport(
        "1",
        SequenceShape(2, 8),
        CostAssumptions(),
        80,
        80,
        (
            ComponentCostEstimate(
                _component(), CostOperation.GATED_MLP, 80, 760, 160, 420
            ),
        ),
    )
    with pytest.raises(MutationCostDeltaError, match="sequence shapes"):
        attach_expected_cost_delta(
            _plan(), baseline, affected_shapes=shape, candidate=candidate
        )


def test_affected_shapes_must_be_canonical_and_inside_mutation_closure() -> None:
    other = ComponentId.parse("model.layers.1.mlp")
    with pytest.raises(MutationCostDeltaError, match="closure"):
        attach_expected_cost_delta(
            _plan(),
            _report(100, 1000, 200, 500),
            affected_shapes=(AffectedShape(other, (2, 2), (1, 2)),),
            mask_only=True,
        )

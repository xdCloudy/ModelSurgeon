from __future__ import annotations

from dataclasses import replace

from tests.test_deployable_state import _state

from modelsurgeon.graph import ComponentId, ComponentIdentityMapping, ComponentIdentityRemap
from modelsurgeon.search import (
    ArchitectureAxis,
    ArchitectureMutationRequest,
    MultiAxisCompilationOutcome,
    compile_multi_axis_sequence,
)
from modelsurgeon.surgery import (
    AlignmentDecision,
    AlignmentDecisionOutcome,
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationPrecondition,
    MutationRequest,
)


def _plan(component: ComponentId, kind: MutationKind, delta: MutationDelta) -> MutationPlan:
    request = MutationRequest(kind, (component,))
    return MutationPlan(request, (component,), (MutationPrecondition("state", "ready"),), delta)


def _target_state(
    base,
    *,
    mutation_id: str,
    layer_widths,
    storage_bytes: int,
    parameter_count: int,
    quantization=None,
):
    artifact = replace(
        base.artifact,
        materialized_artifact_digest=("d" if quantization is None else "e") * 64,
        size_bytes=storage_bytes,
        physical_outcome_id="outcome_" + ("f" if quantization is None else "0") * 64,
    )
    return replace(
        base,
        layer_widths=layer_widths,
        quantization=base.quantization if quantization is None else quantization,
        artifact=artifact,
        parameter_count=parameter_count,
        storage_bytes=storage_bytes,
        mutation_order=(*base.mutation_order, mutation_id),
    )


def test_multi_axis_sequence_reconciles_state_identity_and_deltas() -> None:
    root = _state()
    component = ComponentId.parse("model.layers.0")
    first_plan = _plan(component, MutationKind.REMOVE, MutationDelta(-100, -10, -20, -400))
    first_target = _target_state(
        root,
        mutation_id=first_plan.request.mutation_id,
        layer_widths=(replace(root.layer_widths[0], mlp_width=96), root.layer_widths[1]),
        storage_bytes=3_696,
        parameter_count=900,
    )
    first = ArchitectureMutationRequest(
        ArchitectureAxis.LAYER_WIDTHS,
        root.state_id,
        first_target,
        first_plan,
        ComponentIdentityRemap.retained((component,)),
        AlignmentDecision(AlignmentDecisionOutcome.LEGAL, True, None, "graph and codec aligned"),
    )
    second_plan = _plan(component, MutationKind.REQUANTIZE, MutationDelta(0, 0, -10, -96))
    second_target = _target_state(
        first_target,
        mutation_id=second_plan.request.mutation_id,
        layer_widths=first_target.layer_widths,
        storage_bytes=3_600,
        parameter_count=900,
        quantization=replace(root.quantization, codec="Q4_K_M"),
    )
    second = ArchitectureMutationRequest(
        ArchitectureAxis.QUANTIZATION,
        first_target.state_id,
        second_target,
        second_plan,
        ComponentIdentityRemap.retained((component,)),
    )

    result = compile_multi_axis_sequence(root, (component,), (first, second))
    assert result.outcome is MultiAxisCompilationOutcome.ACCEPTED
    assert result.sequence is not None
    assert result.sequence.current_state.state_id == second_target.state_id
    assert result.sequence.cumulative_delta == MutationDelta(-100, -10, -30, -496)


def test_multi_axis_sequence_rejects_stale_state_and_removed_identity() -> None:
    root = _state()
    component = ComponentId.parse("model.layers.0")
    retained_component = ComponentId.parse("model.layers.1")
    plan = _plan(component, MutationKind.REMOVE, MutationDelta(-100, 0, 0, -400))
    target = _target_state(
        root,
        mutation_id=plan.request.mutation_id,
        layer_widths=(replace(root.layer_widths[0], mlp_width=96), root.layer_widths[1]),
        storage_bytes=3_696,
        parameter_count=900,
    )
    stale = ArchitectureMutationRequest(
        ArchitectureAxis.LAYER_WIDTHS,
        "state_" + "0" * 64,
        target,
        plan,
        ComponentIdentityRemap.retained((component,)),
        AlignmentDecision(AlignmentDecisionOutcome.LEGAL, True, None, "aligned"),
    )
    stale_result = compile_multi_axis_sequence(root, (component,), (stale,))
    assert stale_result.outcome is MultiAxisCompilationOutcome.REJECTED
    assert "stale" in stale_result.reason

    removed_plan = _plan(component, MutationKind.REMOVE, MutationDelta(-100, 0, 0, -400))
    removed_target = _target_state(
        root,
        mutation_id=removed_plan.request.mutation_id,
        layer_widths=(replace(root.layer_widths[0], mlp_width=96), root.layer_widths[1]),
        storage_bytes=3_696,
        parameter_count=900,
    )
    removed = ArchitectureMutationRequest(
        ArchitectureAxis.LAYER_WIDTHS,
        root.state_id,
        removed_target,
        removed_plan,
        ComponentIdentityRemap((ComponentIdentityMapping(component, (), "removed by first step"),)),
        AlignmentDecision(AlignmentDecisionOutcome.LEGAL, True, None, "aligned"),
    )
    next_plan = _plan(component, MutationKind.MASK, MutationDelta())
    next_target = replace(
        removed_target,
        hidden_size=65,
        mutation_order=(*removed_target.mutation_order, next_plan.request.mutation_id),
    )
    next_request = ArchitectureMutationRequest(
        ArchitectureAxis.HIDDEN_SIZE,
        removed_target.state_id,
        next_target,
        next_plan,
        ComponentIdentityRemap.retained((component,)),
    )
    removed_result = compile_multi_axis_sequence(
        root, (component, retained_component), (removed, next_request)
    )
    assert removed_result.outcome is MultiAxisCompilationOutcome.REJECTED
    assert removed_result.rejected_index == 1
    assert "removed" in removed_result.reason

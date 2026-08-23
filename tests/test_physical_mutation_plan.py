"""Tests for allocation-free compilation of physical tensor edit plans."""

from __future__ import annotations

import json

import pytest

from modelsurgeon.graph import (
    ComponentId,
    ComponentIdentityMapping,
    ComponentIdentityRemap,
)
from modelsurgeon.surgery import (
    AxisRemoval,
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationPrecondition,
    MutationRequest,
    PhysicalMetadataUpdate,
    PhysicalPlanError,
    PhysicalTensorDescriptor,
    TensorEditIntent,
    compile_physical_mutation_plan,
)

A = ComponentId.parse("model.a.weight")
B = ComponentId.parse("model.b.weight")
LOGICAL = ComponentId.parse("model.logical")


def _base_plan(storage_delta: int = -300) -> MutationPlan:
    request = MutationRequest(MutationKind.REMOVE, (LOGICAL,), (("channels", 2),))
    return MutationPlan(
        request,
        (A, B, LOGICAL),
        (MutationPrecondition("revision", "abc"),),
        MutationDelta(parameters=-10, storage_bytes=storage_delta),
    )


def _descriptors() -> tuple[PhysicalTensorDescriptor, ...]:
    return (
        PhysicalTensorDescriptor(A, "tensor-a", (4, 5), 400),
        PhysicalTensorDescriptor(B, "tensor-b", (6, 4), 800),
    )


def _intents() -> tuple[TensorEditIntent, ...]:
    return (
        TensorEditIntent(A, (AxisRemoval(1, (1, 3)),), 300),
        TensorEditIntent(B, (AxisRemoval(0, (0,)),), 600),
    )


def _remap() -> ComponentIdentityRemap:
    return ComponentIdentityRemap.build(
        tuple(
            ComponentIdentityMapping(item, (item,), "retained after structural edit")
            for item in (A, B, LOGICAL)
        )
    )


def test_compiles_axes_shapes_metadata_identity_and_storage_delta() -> None:
    compiled = compile_physical_mutation_plan(
        _base_plan(),
        descriptors=_descriptors(),
        edit_intents=_intents(),
        metadata_updates=(
            PhysicalMetadataUpdate("architecture.feed_forward_length", 3),
            PhysicalMetadataUpdate("general.description", "surgically edited"),
        ),
        identity_remap=_remap(),
    )

    assert compiled.expected_storage_delta == -300
    assert compiled.tensor_edits[0].old_shape == (4, 5)
    assert compiled.tensor_edits[0].new_shape == (4, 3)
    assert compiled.tensor_edits[1].new_shape == (5, 4)
    transform = compiled.tensor_edits[0].transforms[0]
    assert transform.map_index(0) == 0
    assert transform.map_index(1) is None
    assert transform.map_index(4) == 2
    assert json.loads(json.dumps(compiled.to_record()))["mutation_id"] == (
        _base_plan().request.mutation_id
    )


def test_missing_coupled_tensor_edit_fails_before_compilation() -> None:
    with pytest.raises(PhysicalPlanError, match=r"model\.b\.weight"):
        compile_physical_mutation_plan(
            _base_plan(),
            descriptors=_descriptors(),
            edit_intents=(_intents()[0],),
            metadata_updates=(),
            identity_remap=_remap(),
        )


@pytest.mark.parametrize(
    "intent",
    [
        TensorEditIntent(A, (AxisRemoval(2, (0,)),), 300),
        TensorEditIntent(A, (AxisRemoval(1, (5,)),), 300),
        TensorEditIntent(A, (AxisRemoval(0, (0, 1, 2, 3)),), 300),
    ],
)
def test_unrepresentable_axis_intents_fail_before_tensor_allocation(
    intent: TensorEditIntent,
) -> None:
    with pytest.raises(PhysicalPlanError, match=r"axis|index"):
        compile_physical_mutation_plan(
            _base_plan(),
            descriptors=_descriptors(),
            edit_intents=(intent, _intents()[1]),
            metadata_updates=(),
            identity_remap=_remap(),
        )


def test_identity_coverage_and_storage_estimate_must_reconcile() -> None:
    incomplete = ComponentIdentityRemap.build(
        (
            ComponentIdentityMapping(A, (A,), "retained"),
            ComponentIdentityMapping(B, (B,), "retained"),
        )
    )
    with pytest.raises(PhysicalPlanError, match="identity mappings"):
        compile_physical_mutation_plan(
            _base_plan(),
            descriptors=_descriptors(),
            edit_intents=_intents(),
            metadata_updates=(),
            identity_remap=incomplete,
        )
    with pytest.raises(PhysicalPlanError, match="storage delta"):
        compile_physical_mutation_plan(
            _base_plan(storage_delta=-299),
            descriptors=_descriptors(),
            edit_intents=_intents(),
            metadata_updates=(),
            identity_remap=_remap(),
        )


def test_duplicate_metadata_and_noncanonical_intents_fail_closed() -> None:
    with pytest.raises(PhysicalPlanError, match="metadata updates"):
        compile_physical_mutation_plan(
            _base_plan(),
            descriptors=_descriptors(),
            edit_intents=_intents(),
            metadata_updates=(
                PhysicalMetadataUpdate("same", 1),
                PhysicalMetadataUpdate("same", 2),
            ),
            identity_remap=_remap(),
        )

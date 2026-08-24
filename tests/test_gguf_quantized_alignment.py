"""Tests for GGUF codec block and physical mutation alignment validation."""

from __future__ import annotations

import pytest

from modelsurgeon.adapters.gguf import GGMLQuantizationType
from modelsurgeon.graph import (
    ComponentId,
    ComponentIdentityMapping,
    ComponentIdentityRemap,
)
from modelsurgeon.surgery import (
    AxisRemoval,
    GGUFAlignmentError,
    GGUFQuantizationBinding,
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationRequest,
    PhysicalTensorDescriptor,
    QuantizedEditStrategy,
    TensorEditIntent,
    compile_physical_mutation_plan,
    propose_aligned_axis_removal,
    validate_gguf_quantized_plan,
)

TENSOR = ComponentId.parse("model.weight")


def _physical(
    removal: AxisRemoval,
    *,
    old_shape: tuple[int, ...] = (512, 8),
    old_bytes: int = 2304,
    new_bytes: int,
):
    delta = new_bytes - old_bytes
    base = MutationPlan(
        MutationRequest(MutationKind.REMOVE, (TENSOR,)),
        (TENSOR,),
        (),
        MutationDelta(storage_bytes=delta),
    )
    return compile_physical_mutation_plan(
        base,
        descriptors=(PhysicalTensorDescriptor(TENSOR, "weight", old_shape, old_bytes),),
        edit_intents=(TensorEditIntent(TENSOR, (removal,), new_bytes),),
        metadata_updates=(),
        identity_remap=ComponentIdentityRemap.build(
            (ComponentIdentityMapping(TENSOR, (TENSOR,), "retained"),)
        ),
    )


def _validate(
    plan,
    quant_type=GGMLQuantizationType.Q4_K,
    destination_quant_type=None,
):
    return validate_gguf_quantized_plan(
        plan,
        (GGUFQuantizationBinding(TENSOR, quant_type, destination_quant_type),),
    )


def test_complete_contiguous_blocks_use_direct_block_copy() -> None:
    plan = _physical(AxisRemoval(0, tuple(range(256))), new_bytes=1152)
    validated = _validate(plan)
    axis = validated.tensor_edits[0].axis_edits[0]
    assert axis.strategy is QuantizedEditStrategy.DIRECT_BLOCK_COPY
    assert [(item.block_offset, item.block_count) for item in axis.affected_block_ranges] == [
        (0, 1)
    ]


def test_unaligned_contiguous_axis_edit_requires_decode_repack() -> None:
    removed = tuple(range(0, 512, 2))
    plan = _physical(AxisRemoval(0, removed), new_bytes=1152)
    axis = _validate(plan).tensor_edits[0].axis_edits[0]
    assert axis.strategy is QuantizedEditStrategy.REPACK_CONTIGUOUS_AXIS
    assert axis.affected_block_ranges[0].block_offset == 0
    assert axis.affected_block_ranges[0].block_count == 2


def test_outer_axis_edits_are_whole_slice_copies() -> None:
    plan = _physical(AxisRemoval(1, (1, 3)), new_bytes=1728)
    axis = _validate(plan).tensor_edits[0].axis_edits[0]
    assert axis.strategy is QuantizedEditStrategy.WHOLE_SLICE_COPY
    assert axis.affected_block_ranges == ()


def test_unrepresentable_new_shape_fails_before_decode_with_explicit_proposal() -> None:
    plan = _physical(AxisRemoval(0, (1,)), new_bytes=2300)
    with pytest.raises(GGUFAlignmentError, match="not representable"):
        _validate(plan)

    proposal = propose_aligned_axis_removal(GGMLQuantizationType.Q4_K, 512, (1,))
    assert proposal.aligned_indices == tuple(range(256))
    assert proposal.added_indices == tuple(index for index in range(256) if index != 1)
    assert proposal.new_size == 256
    assert proposal.to_axis_removal() == AxisRemoval(0, tuple(range(256)))


def test_exact_binding_coverage_and_native_codec_are_required() -> None:
    plan = _physical(AxisRemoval(0, tuple(range(256))), new_bytes=1152)
    with pytest.raises(GGUFAlignmentError, match="exactly cover"):
        validate_gguf_quantized_plan(plan, ())
    repack = _physical(
        AxisRemoval(0, tuple(range(0, 512, 2))),
        old_bytes=4672,
        new_bytes=2336,
    )
    with pytest.raises(GGUFAlignmentError, match="no exact native write codec"):
        _validate(repack, GGMLQuantizationType.Q8_K)


def test_storage_only_legacy_codec_allows_copy_only_edits() -> None:
    outer = _physical(
        AxisRemoval(1, (1, 3)),
        old_shape=(32, 8),
        old_bytes=176,
        new_bytes=132,
    )
    validated = _validate(outer, GGMLQuantizationType.Q5_0)
    assert (
        validated.tensor_edits[0].axis_edits[0].strategy
        is QuantizedEditStrategy.WHOLE_SLICE_COPY
    )

    partial = _physical(
        AxisRemoval(0, tuple(range(0, 64, 2))),
        old_shape=(64, 8),
        old_bytes=352,
        new_bytes=176,
    )
    with pytest.raises(GGUFAlignmentError, match="only unchanged-codec"):
        _validate(partial, GGMLQuantizationType.Q5_0)


def test_encoded_size_is_recomputed_from_exact_layout() -> None:
    plan = _physical(AxisRemoval(0, tuple(range(256))), new_bytes=1153)
    with pytest.raises(GGUFAlignmentError, match="layout size 1152"):
        _validate(plan)


def test_explicit_destination_codec_controls_new_shape_payload_size() -> None:
    plan = _physical(AxisRemoval(0, tuple(range(256))), new_bytes=2176)
    validated = _validate(
        plan,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q8_0,
    )
    assert validated.tensor_edits[0].quant_type is GGMLQuantizationType.Q4_K
    assert (
        validated.tensor_edits[0].destination_quant_type
        is GGMLQuantizationType.Q8_0
    )
    assert validated.tensor_edits[0].encoded_size == 2176

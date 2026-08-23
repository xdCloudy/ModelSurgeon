"""Tests for bounded changed-block GGUF requantization and validation."""

from __future__ import annotations

import pytest

from modelsurgeon.adapters.gguf import (
    Q4_K_CODEC,
    Q8_0_CODEC,
    ByteOrder,
    CodecRegistry,
    GGMLQuantizationType,
)
from modelsurgeon.graph import (
    ComponentId,
    ComponentIdentityMapping,
    ComponentIdentityRemap,
)
from modelsurgeon.surgery import (
    AxisRemoval,
    ChangedGGUFFloatChunk,
    GGUFQuantizationBinding,
    GGUFRequantizationLimits,
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationRequest,
    PhysicalTensorDescriptor,
    SelectiveGGUFRequantizer,
    SelectiveRequantizationError,
    TensorEditIntent,
    compile_physical_mutation_plan,
    validate_gguf_quantized_plan,
)

TENSOR = ComponentId.parse("model.weight")


def _registry() -> CodecRegistry:
    registry = CodecRegistry()
    registry.register(Q4_K_CODEC)
    registry.register(Q8_0_CODEC)
    return registry


def _plan(
    *,
    old_shape: tuple[int, ...] = (512, 1),
    removed: tuple[int, ...] = tuple(range(0, 512, 2)),
    destination: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
):
    old_bytes = old_shape[0] // 256 * 144
    new_size = old_shape[0] - len(removed)
    if destination is GGMLQuantizationType.Q4_K:
        new_bytes = new_size // 256 * 144
    else:
        new_bytes = new_size // 32 * 34
    base = MutationPlan(
        MutationRequest(MutationKind.REMOVE, (TENSOR,)),
        (TENSOR,),
        (),
        MutationDelta(storage_bytes=new_bytes - old_bytes),
    )
    physical = compile_physical_mutation_plan(
        base,
        descriptors=(PhysicalTensorDescriptor(TENSOR, "weight", old_shape, old_bytes),),
        edit_intents=(
            TensorEditIntent(TENSOR, (AxisRemoval(0, removed),), new_bytes),
        ),
        metadata_updates=(),
        identity_remap=ComponentIdentityRemap.build(
            (ComponentIdentityMapping(TENSOR, (TENSOR,), "retained"),)
        ),
    )
    return validate_gguf_quantized_plan(
        physical,
        (GGUFQuantizationBinding(TENSOR, GGMLQuantizationType.Q4_K, destination),),
    )


def _values(count: int) -> list[float]:
    return [((index % 37) - 18) / 7 for index in range(count)]


def test_original_codec_chunk_validates_shape_payload_and_error_summary() -> None:
    plan = _plan()
    requantizer = SelectiveGGUFRequantizer()
    chunks = list(
        requantizer.iter_encoded(
            plan,
            (ChangedGGUFFloatChunk(TENSOR, 0, _values(256)),),
            _registry(),
            byte_order=ByteOrder.LITTLE,
        )
    )
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.destination_quant_type is GGMLQuantizationType.Q4_K
    assert chunk.block_count == 1
    assert len(chunk.payload) == 144
    assert chunk.tensor_byte_offset == 0
    assert chunk.error_summary.error.max_absolute_error < 1.0
    report = requantizer.report()
    assert report.complete is True
    assert report.encoded_blocks == 1
    assert report.encoded_bytes == 144
    assert report.touched_ranges[0].byte_count == 144


def test_explicit_selected_codec_streams_exact_selected_layout() -> None:
    plan = _plan(destination=GGMLQuantizationType.Q8_0)
    requantizer = SelectiveGGUFRequantizer(
        GGUFRequantizationLimits(
            max_encoded_chunk_bytes=34,
            max_validation_values=32,
            max_working_bytes=400,
        )
    )
    chunks = list(
        requantizer.iter_encoded(
            plan,
            (ChangedGGUFFloatChunk(TENSOR, 0, _values(256)),),
            _registry(),
            byte_order=ByteOrder.BIG,
        )
    )
    assert len(chunks) == 8
    assert all(chunk.destination_quant_type is GGMLQuantizationType.Q8_0 for chunk in chunks)
    assert all(len(chunk.payload) == 34 for chunk in chunks)
    assert [chunk.block_offset for chunk in chunks] == list(range(8))
    assert requantizer.report().encoded_bytes == 272
    assert requantizer.report().peak_working_bytes <= 400


def test_large_changed_values_are_split_by_working_memory_budget() -> None:
    plan = _plan(
        old_shape=(768, 1),
        removed=tuple(range(0, 768, 3)),
    )
    requantizer = SelectiveGGUFRequantizer(
        GGUFRequantizationLimits(
            max_encoded_chunk_bytes=144,
            max_validation_values=256,
            max_working_bytes=2400,
        )
    )
    chunks = list(
        requantizer.iter_encoded(
            plan,
            (ChangedGGUFFloatChunk(TENSOR, 0, _values(512)),),
            _registry(),
            byte_order=ByteOrder.LITTLE,
        )
    )
    assert [(chunk.block_offset, chunk.block_count) for chunk in chunks] == [(0, 1), (1, 1)]
    assert requantizer.report().peak_working_bytes == 2 * 144 + 256 * 8


def test_partial_blocks_ranges_and_overlaps_fail_before_emission() -> None:
    plan = _plan()
    requantizer = SelectiveGGUFRequantizer()
    with pytest.raises(SelectiveRequantizationError, match="block multiple"):
        list(
            requantizer.iter_encoded(
                plan,
                (ChangedGGUFFloatChunk(TENSOR, 0, _values(255)),),
                _registry(),
                byte_order=ByteOrder.LITTLE,
            )
        )
    with pytest.raises(SelectiveRequantizationError, match="escapes"):
        list(
            requantizer.iter_encoded(
                plan,
                (ChangedGGUFFloatChunk(TENSOR, 1, _values(256)),),
                _registry(),
                byte_order=ByteOrder.LITTLE,
            )
        )


def test_error_ceiling_rejects_payload_before_it_is_reported() -> None:
    plan = _plan()
    requantizer = SelectiveGGUFRequantizer(
        GGUFRequantizationLimits(max_absolute_error=0.0)
    )
    with pytest.raises(SelectiveRequantizationError, match="max error"):
        list(
            requantizer.iter_encoded(
                plan,
                (ChangedGGUFFloatChunk(TENSOR, 0, _values(256)),),
                _registry(),
                byte_order=ByteOrder.LITTLE,
            )
        )
    assert requantizer.report().touched_ranges == ()
    assert requantizer.report().complete is False

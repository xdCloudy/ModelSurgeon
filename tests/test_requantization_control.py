"""Tests for matched codec-range requantization controls and delta attribution."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from modelsurgeon.adapters.gguf import (
    Q4_K_CODEC,
    Q8_0_CODEC,
    ByteOrder,
    CodecRegistry,
    GGMLQuantizationType,
    GGUFTensorChunk,
    GGUFTensorHandle,
)
from modelsurgeon.evaluation import (
    MatchedGGUFRequantizationControl,
    MatchedRequantizationControlError,
    MatchedRequantizationControlLimits,
    MatchedRequantizationDeltas,
)
from modelsurgeon.graph import (
    ComponentId,
    ComponentIdentityMapping,
    ComponentIdentityRemap,
)
from modelsurgeon.surgery import (
    AxisRemoval,
    GGUFQuantizationBinding,
    GGUFRequantizationLimits,
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationRequest,
    PhysicalTensorDescriptor,
    SelectiveDequantizationLimits,
    TensorEditIntent,
    compile_physical_mutation_plan,
    validate_gguf_quantized_plan,
)

_TENSOR = ComponentId.parse("model.weight")


class _Index:
    def __init__(self, handle: GGUFTensorHandle) -> None:
        self.handle = handle

    def tensor(self, name: str) -> GGUFTensorHandle:
        assert name == "weight"
        return self.handle


class _Reader:
    def __init__(self, shape: tuple[int, ...]) -> None:
        block_count = shape[0] // 256 * shape[1]
        values = [((index % 43) - 21) / 9 for index in range(block_count * 256)]
        encoded = bytearray(block_count * Q4_K_CODEC.layout.type_size)
        Q4_K_CODEC.encode_blocks(values, memoryview(encoded), byte_order=ByteOrder.LITTLE)
        self.payload = bytes(encoded)
        self.handle = GGUFTensorHandle(
            "fake",
            0,
            "weight",
            shape,
            GGMLQuantizationType.Q4_K,
            len(encoded),
            Q4_K_CODEC.layout.type_size,
            Q4_K_CODEC.layout.block_size,
        )
        self.index = _Index(self.handle)
        self.source = SimpleNamespace(
            container=SimpleNamespace(byte_order=ByteOrder.LITTLE)
        )
        self.reads: list[tuple[int, int]] = []

    def read_blocks(
        self,
        handle: GGUFTensorHandle,
        block_offset: int,
        block_count: int,
    ) -> GGUFTensorChunk:
        assert handle == self.handle
        self.reads.append((block_offset, block_count))
        start = block_offset * Q4_K_CODEC.layout.type_size
        size = block_count * Q4_K_CODEC.layout.type_size
        return GGUFTensorChunk(
            handle,
            block_offset,
            block_count,
            start,
            block_offset * Q4_K_CODEC.layout.block_size,
            self.payload[start : start + size],
        )


def _registry() -> CodecRegistry:
    registry = CodecRegistry()
    registry.register(Q4_K_CODEC)
    registry.register(Q8_0_CODEC)
    return registry


def _plan(
    *,
    removed: tuple[int, ...],
    destination: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
):
    shape = (768, 2)
    old_bytes = 6 * Q4_K_CODEC.layout.type_size
    new_width = shape[0] - len(removed)
    destination_layout = (
        Q4_K_CODEC.layout
        if destination is GGMLQuantizationType.Q4_K
        else Q8_0_CODEC.layout
    )
    new_bytes = new_width * shape[1] // destination_layout.block_size
    new_bytes *= destination_layout.type_size
    mutation = MutationPlan(
        MutationRequest(MutationKind.REMOVE, (_TENSOR,)),
        (_TENSOR,),
        (),
        MutationDelta(storage_bytes=new_bytes - old_bytes),
    )
    physical = compile_physical_mutation_plan(
        mutation,
        descriptors=(PhysicalTensorDescriptor(_TENSOR, "weight", shape, old_bytes),),
        edit_intents=(
            TensorEditIntent(_TENSOR, (AxisRemoval(0, removed),), new_bytes),
        ),
        metadata_updates=(),
        identity_remap=ComponentIdentityRemap.build(
            (ComponentIdentityMapping(_TENSOR, (_TENSOR,), "retained"),)
        ),
    )
    return validate_gguf_quantized_plan(
        physical,
        (GGUFQuantizationBinding(_TENSOR, GGMLQuantizationType.Q4_K, destination),),
    )


def test_control_decodes_and_reencodes_exact_same_sparse_codec_ranges() -> None:
    removed = tuple(range(256, 768, 2))
    reader = _Reader((768, 2))
    control = MatchedGGUFRequantizationControl(
        seed=1729,
        limits=MatchedRequantizationControlLimits(
            SelectiveDequantizationLimits(
                max_encoded_chunk_bytes=Q4_K_CODEC.layout.type_size,
                max_decoded_values=256,
                max_working_bytes=2048,
            ),
            GGUFRequantizationLimits(
                max_encoded_chunk_bytes=Q4_K_CODEC.layout.type_size,
                max_validation_values=256,
                max_working_bytes=4096,
            ),
        ),
    )

    chunks = list(control.iter_encoded(_plan(removed=removed), reader, _registry()))
    report = control.report()

    assert report.complete is True
    assert report.seed == 1729
    assert [(item.block_offset, item.block_count) for item in report.matched_ranges] == [
        (1, 2),
        (4, 2),
    ]
    assert reader.reads == [(1, 1), (2, 1), (4, 1), (5, 1)]
    assert [(item.block_offset, item.block_count) for item in chunks] == [
        (1, 1),
        (2, 1),
        (4, 1),
        (5, 1),
    ]
    assert report.dequantization.encoded_bytes_read == 4 * Q4_K_CODEC.layout.type_size
    assert report.requantization.encoded_bytes == 4 * Q4_K_CODEC.layout.type_size


def test_aligned_copy_only_plan_has_an_empty_complete_control() -> None:
    reader = _Reader((768, 2))
    control = MatchedGGUFRequantizationControl()

    assert list(
        control.iter_encoded(
            _plan(removed=tuple(range(256))), reader, _registry()
        )
    ) == []
    assert control.report().complete is True
    assert control.report().matched_ranges == ()
    assert reader.reads == []


def test_storage_only_copy_plan_never_resolves_a_float_codec() -> None:
    reader = _Reader((768, 2))
    plan = _plan(removed=tuple(range(256)))
    edit = plan.tensor_edits[0]
    storage_only = replace(
        plan,
        tensor_edits=(
            replace(
                edit,
                quant_type=GGMLQuantizationType.Q5_0,
                destination_quant_type=GGMLQuantizationType.Q5_0,
            ),
        ),
    )
    control = MatchedGGUFRequantizationControl(seed=1729)

    assert list(control.iter_encoded(storage_only, reader, CodecRegistry())) == []
    assert control.report().complete is True
    assert control.report().matched_ranges == ()
    assert reader.reads == []


def test_codec_substitution_and_invalid_seed_fail_closed() -> None:
    with pytest.raises(MatchedRequantizationControlError, match="unsigned 64-bit"):
        MatchedGGUFRequantizationControl(seed=-1)
    reader = _Reader((768, 2))
    control = MatchedGGUFRequantizationControl()
    with pytest.raises(MatchedRequantizationControlError, match="identical"):
        list(
            control.iter_encoded(
                _plan(
                    removed=tuple(range(256, 768, 2)),
                    destination=GGMLQuantizationType.Q8_0,
                ),
                reader,
                _registry(),
            )
        )
    assert reader.reads == []


def test_metric_deltas_separate_and_reconcile_control_and_surgery_effects() -> None:
    deltas = MatchedRequantizationDeltas("perplexity", 10.0, 10.25, 11.0)

    assert deltas.requantization_delta == 0.25
    assert deltas.surgery_delta == 0.75
    assert deltas.combined_delta == 1.0
    assert deltas.requantization_delta + deltas.surgery_delta == deltas.combined_delta
    assert deltas.to_record()["surgery_delta"] == 0.75
    with pytest.raises(MatchedRequantizationControlError, match="finite"):
        MatchedRequantizationDeltas("perplexity", 10.0, float("nan"), 11.0)

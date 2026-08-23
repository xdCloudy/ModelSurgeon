"""Tests for bounded decoding of only GGUF blocks affected by physical edits."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from modelsurgeon.adapters.gguf import (
    Q4_K_CODEC,
    ByteOrder,
    CodecRegistry,
    GGMLQuantizationType,
    GGUFTensorChunk,
    GGUFTensorHandle,
    UnsupportedCodecError,
)
from modelsurgeon.graph import (
    ComponentId,
    ComponentIdentityMapping,
    ComponentIdentityRemap,
)
from modelsurgeon.surgery import (
    AxisRemoval,
    DequantizedPrecision,
    GGUFQuantizationBinding,
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationRequest,
    PhysicalTensorDescriptor,
    SelectiveDequantizationError,
    SelectiveDequantizationLimits,
    SelectiveGGUFDequantizer,
    TensorEditIntent,
    compile_physical_mutation_plan,
    validate_gguf_quantized_plan,
)

TENSOR = ComponentId.parse("model.weight")


class _Index:
    def __init__(self, handle: GGUFTensorHandle) -> None:
        self.handle = handle

    def tensor(self, name: str) -> GGUFTensorHandle:
        assert name == self.handle.name
        return self.handle


class _Reader:
    def __init__(self, shape: tuple[int, ...]) -> None:
        block_count = shape[0] // 256
        for dimension in shape[1:]:
            block_count *= dimension
        values = [((index % 31) - 15) / 5 for index in range(block_count * 256)]
        encoded = bytearray(block_count * 144)
        Q4_K_CODEC.encode_blocks(values, memoryview(encoded), byte_order=ByteOrder.LITTLE)
        self.payload = bytes(encoded)
        self.handle = GGUFTensorHandle(
            "fake",
            0,
            "weight",
            shape,
            GGMLQuantizationType.Q4_K,
            len(encoded),
            144,
            256,
        )
        self.index = _Index(self.handle)
        self.source = SimpleNamespace(
            container=SimpleNamespace(byte_order=ByteOrder.LITTLE)
        )
        self.reads: list[tuple[int, int]] = []

    def read_blocks(
        self, handle: GGUFTensorHandle, block_offset: int, block_count: int
    ) -> GGUFTensorChunk:
        assert handle == self.handle
        self.reads.append((block_offset, block_count))
        start = block_offset * 144
        return GGUFTensorChunk(
            handle,
            block_offset,
            block_count,
            start,
            block_offset * 256,
            self.payload[start : start + block_count * 144],
        )


def _registry() -> CodecRegistry:
    registry = CodecRegistry()
    registry.register(Q4_K_CODEC)
    return registry


def _plan(
    shape: tuple[int, ...], removals: tuple[AxisRemoval, ...], new_bytes: int
):
    old_blocks = shape[0] // 256
    for dimension in shape[1:]:
        old_blocks *= dimension
    old_bytes = old_blocks * 144
    base = MutationPlan(
        MutationRequest(MutationKind.REMOVE, (TENSOR,)),
        (TENSOR,),
        (),
        MutationDelta(storage_bytes=new_bytes - old_bytes),
    )
    physical = compile_physical_mutation_plan(
        base,
        descriptors=(PhysicalTensorDescriptor(TENSOR, "weight", shape, old_bytes),),
        edit_intents=(TensorEditIntent(TENSOR, removals, new_bytes),),
        metadata_updates=(),
        identity_remap=ComponentIdentityRemap.build(
            (ComponentIdentityMapping(TENSOR, (TENSOR,), "retained"),)
        ),
    )
    return validate_gguf_quantized_plan(
        physical, (GGUFQuantizationBinding(TENSOR, GGMLQuantizationType.Q4_K),)
    )


def test_large_repack_is_chunked_under_hard_peak_memory_limit() -> None:
    plan = _plan((512, 4), (AxisRemoval(0, tuple(range(0, 512, 2))),), 576)
    reader = _Reader((512, 4))
    decoder = SelectiveGGUFDequantizer(
        SelectiveDequantizationLimits(
            max_encoded_chunk_bytes=144,
            max_decoded_values=256,
            max_working_bytes=1200,
            precision=DequantizedPrecision.FP32,
        )
    )

    chunks = list(decoder.iter_chunks(plan, reader, _registry()))
    report = decoder.report()
    assert len(chunks) == 8
    assert all(chunk.values.typecode == "f" for chunk in chunks)
    assert all(chunk.block_count == 1 for chunk in chunks)
    assert report.complete is True
    assert report.encoded_bytes_read == 8 * 144
    assert report.decoded_values == 8 * 256
    assert report.peak_working_bytes == 144 + 256 * 4
    assert report.peak_working_bytes <= 1200
    assert [(item.block_offset, item.block_count) for item in report.touched_ranges] == [
        (index, 1) for index in range(8)
    ]


def test_removed_outer_slices_are_not_read_or_decoded() -> None:
    plan = _plan(
        (512, 4),
        (AxisRemoval(0, tuple(range(0, 512, 2))), AxisRemoval(1, (1, 3))),
        288,
    )
    reader = _Reader((512, 4))
    decoder = SelectiveGGUFDequantizer(
        SelectiveDequantizationLimits(max_working_bytes=2400)
    )
    chunks = list(decoder.iter_chunks(plan, reader, _registry()))
    assert [(chunk.block_offset, chunk.block_count) for chunk in chunks] == [
        (0, 2),
        (4, 2),
    ]
    assert reader.reads == [(0, 2), (4, 2)]


def test_untouched_prefix_blocks_are_excluded_from_sparse_reads() -> None:
    removed = tuple(range(256, 768, 2))
    plan = _plan((768, 2), (AxisRemoval(0, removed),), 576)
    reader = _Reader((768, 2))
    decoder = SelectiveGGUFDequantizer()
    list(decoder.iter_chunks(plan, reader, _registry()))
    assert reader.reads == [(1, 2), (4, 2)]
    assert all(offset not in {0, 3} for offset, _ in reader.reads)


def test_direct_block_and_outer_only_edits_require_no_dequantization() -> None:
    plan = _plan((512, 4), (AxisRemoval(0, tuple(range(256))),), 576)
    reader = _Reader((512, 4))
    decoder = SelectiveGGUFDequantizer()
    assert list(decoder.iter_chunks(plan, reader, _registry())) == []
    assert decoder.report().complete is True
    assert reader.reads == []


def test_limits_and_exact_codec_fail_before_read() -> None:
    plan = _plan((512, 1), (AxisRemoval(0, tuple(range(0, 512, 2))),), 144)
    reader = _Reader((512, 1))
    too_small = SelectiveGGUFDequantizer(
        SelectiveDequantizationLimits(max_working_bytes=100)
    )
    with pytest.raises(SelectiveDequantizationError, match="cannot hold one"):
        list(too_small.iter_chunks(plan, reader, _registry()))
    assert reader.reads == []

    missing = SelectiveGGUFDequantizer()
    with pytest.raises(UnsupportedCodecError, match="no exact codec registered"):
        list(missing.iter_chunks(plan, reader, CodecRegistry()))
    assert reader.reads == []

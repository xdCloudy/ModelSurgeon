"""Tests for exact GGUF block-layout and quantization codec contracts."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence

import pytest

from modelsurgeon.adapters.gguf import (
    QUANT_LAYOUTS,
    AxisEditMode,
    BlockOperation,
    BlockValidation,
    ByteOrder,
    CodecContractError,
    CodecIdentity,
    CodecRegistry,
    GGMLQuantizationType,
    QuantizationError,
    UnsupportedCodecError,
    plan_axis_edit,
    plan_supported_axes,
    validate_tensor_alignment,
)
from modelsurgeon.adapters.gguf.quantization import LEGACY_STORAGE_LAYOUTS


class FakeCodec:
    def __init__(self, quant_type: GGMLQuantizationType) -> None:
        self.identity = CodecIdentity(quant_type, "fake", "1", "upstream")
        self.layout = QUANT_LAYOUTS[quant_type]

    def validate_blocks(self, source: memoryview, *, byte_order: ByteOrder) -> BlockValidation:
        del byte_order
        valid = len(source) % self.layout.type_size == 0
        return BlockValidation(
            valid,
            len(source) // self.layout.type_size,
            "valid blocks" if valid else "partial encoded block",
        )

    def decode_blocks(
        self,
        source: memoryview,
        destination: MutableSequence[float],
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation:
        del destination, byte_order
        validation = self.validate_blocks(source, byte_order=ByteOrder.LITTLE)
        validation.require_valid()
        return BlockOperation(
            validation.block_count,
            validation.block_count * self.layout.block_size,
            len(source),
        )

    def encode_blocks(
        self,
        source: Sequence[float],
        destination: memoryview,
        *,
        byte_order: ByteOrder,
    ) -> BlockOperation:
        del destination, byte_order
        byte_count = self.layout.encoded_size(len(source))
        return BlockOperation(len(source) // self.layout.block_size, len(source), byte_count)

    def estimate_error(
        self,
        reference: Sequence[float],
        candidate: Sequence[float],
    ) -> QuantizationError:
        return QuantizationError.measure(reference, candidate)


def test_all_layout_fields_are_contiguous_and_match_pinned_sizes() -> None:
    for layout in QUANT_LAYOUTS.values():
        assert sum(field.size for field in layout.fields) == layout.type_size
        assert [field.offset for field in layout.fields] == [
            sum(previous.size for previous in layout.fields[:index])
            for index in range(len(layout.fields))
        ]


def test_legacy_layouts_are_index_only_and_not_native_codec_claims() -> None:
    expected = {
        GGMLQuantizationType.Q4_0: (32, 18),
        GGMLQuantizationType.Q4_1: (32, 20),
        GGMLQuantizationType.Q5_0: (32, 22),
        GGMLQuantizationType.Q5_1: (32, 24),
    }
    assert {
        quant_type: (layout.block_size, layout.type_size)
        for quant_type, layout in LEGACY_STORAGE_LAYOUTS.items()
    } == expected
    assert not set(LEGACY_STORAGE_LAYOUTS) & set(QUANT_LAYOUTS)


def test_quantized_contiguous_axes_report_block_granularity() -> None:
    plan = plan_axis_edit(GGMLQuantizationType.Q4_K, (512, 8, 2), 0)

    assert plan.mode is AxisEditMode.BLOCK_VALUES
    assert plan.index_granularity == 256
    assert plan.row_bytes == 288
    assert plan.tensor_bytes == 4_608


def test_outer_axes_report_whole_slice_support() -> None:
    plan = plan_axis_edit(GGMLQuantizationType.Q8_0, (64, 12), 1)

    assert plan.mode is AxisEditMode.WHOLE_SLICES
    assert plan.index_granularity == 1
    assert plan.row_bytes == 68
    assert plan.tensor_bytes == 816


def test_all_supported_axes_and_container_alignment_are_explicit() -> None:
    plans = plan_supported_axes(GGMLQuantizationType.Q8_0, (64, 12, 2))

    assert [plan.axis for plan in plans] == [0, 1, 2]
    assert [plan.index_granularity for plan in plans] == [32, 1, 1]
    assert validate_tensor_alignment(64, 32).container_alignment == 32
    with pytest.raises(CodecContractError, match="power of two"):
        validate_tensor_alignment(0, 24)
    with pytest.raises(CodecContractError, match="not aligned"):
        validate_tensor_alignment(33, 32)


@pytest.mark.parametrize(
    ("shape", "axis", "message"),
    [
        ((255, 4), 0, "divisible"),
        ((256, 0), 0, "positive"),
        ((256, 4), 2, "outside rank"),
    ],
)
def test_invalid_shape_and_axis_constraints_fail_closed(
    shape: tuple[int, ...],
    axis: int,
    message: str,
) -> None:
    with pytest.raises(CodecContractError, match=message):
        plan_axis_edit(GGMLQuantizationType.Q2_K, shape, axis)


def test_registry_never_substitutes_one_k_quant_for_another() -> None:
    registry = CodecRegistry()
    q4_codec = FakeCodec(GGMLQuantizationType.Q4_K)
    registry.register(q4_codec)

    assert registry.resolve(GGMLQuantizationType.Q4_K) is q4_codec
    with pytest.raises(UnsupportedCodecError, match="family substitution is forbidden"):
        registry.resolve(GGMLQuantizationType.Q5_K)


def test_codec_validation_and_operations_are_bounded_in_blocks() -> None:
    codec = FakeCodec(GGMLQuantizationType.Q8_0)
    valid = memoryview(bytes(68))
    invalid = memoryview(bytes(35))

    assert codec.validate_blocks(valid, byte_order=ByteOrder.LITTLE).block_count == 2
    with pytest.raises(CodecContractError, match="partial encoded block"):
        codec.validate_blocks(invalid, byte_order=ByteOrder.LITTLE).require_valid()
    operation = codec.decode_blocks(valid, [], byte_order=ByteOrder.LITTLE)
    assert operation == BlockOperation(block_count=2, element_count=64, byte_count=68)


def test_error_estimation_reports_stable_metrics() -> None:
    error = QuantizationError.measure([0.0, 1.0, -2.0], [0.0, 0.5, -1.0])

    assert error.sample_count == 3
    assert error.mean_absolute_error == pytest.approx(0.5)
    assert error.mean_squared_error == pytest.approx(1.25 / 3)
    assert error.max_absolute_error == 1.0
    assert error.reference_l2 == pytest.approx(5**0.5)


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [([], []), ([1.0], []), ([float("nan")], [0.0])],
)
def test_error_estimation_rejects_invalid_samples(
    reference: list[float],
    candidate: list[float],
) -> None:
    with pytest.raises(CodecContractError):
        QuantizationError.measure(reference, candidate)

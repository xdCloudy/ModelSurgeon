"""GGUF codec block and axis validation for compiled physical mutation plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.adapters.gguf.quantization import (
    QUANT_LAYOUTS,
    CodecContractError,
    GGMLQuantizationType,
    plan_axis_edit,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.physical_plan import (
    AxisRemoval,
    PhysicalMutationPlan,
    PhysicalTensorEdit,
)


class GGUFAlignmentError(CodecContractError):
    """Raised before decode/output when a physical plan is not GGUF-representable."""


class QuantizedEditStrategy(StrEnum):
    DIRECT_BLOCK_COPY = "direct_block_copy"
    REPACK_CONTIGUOUS_AXIS = "repack_contiguous_axis"
    WHOLE_SLICE_COPY = "whole_slice_copy"


@dataclass(frozen=True, slots=True)
class GGUFQuantizationBinding:
    component_id: ComponentId
    quant_type: GGMLQuantizationType
    destination_quant_type: GGMLQuantizationType | None = None

    @property
    def output_quant_type(self) -> GGMLQuantizationType:
        return (
            self.quant_type
            if self.destination_quant_type is None
            else self.destination_quant_type
        )


@dataclass(frozen=True, slots=True)
class EncodedBlockRange:
    block_offset: int
    block_count: int

    def __post_init__(self) -> None:
        if self.block_offset < 0 or self.block_count <= 0:
            raise GGUFAlignmentError("encoded block ranges must be non-negative and non-empty")


@dataclass(frozen=True, slots=True)
class QuantizedAxisEdit:
    axis: int
    strategy: QuantizedEditStrategy
    index_granularity: int
    removed_indices: tuple[int, ...]
    affected_block_ranges: tuple[EncodedBlockRange, ...] = ()

    def to_record(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "strategy": self.strategy.value,
            "index_granularity": self.index_granularity,
            "removed_indices": list(self.removed_indices),
            "affected_block_ranges": [
                {"block_offset": item.block_offset, "block_count": item.block_count}
                for item in self.affected_block_ranges
            ],
        }


@dataclass(frozen=True, slots=True)
class GGUFQuantizedTensorEdit:
    component_id: ComponentId
    quant_type: GGMLQuantizationType
    destination_quant_type: GGMLQuantizationType
    old_shape: tuple[int, ...]
    new_shape: tuple[int, ...]
    axis_edits: tuple[QuantizedAxisEdit, ...]
    encoded_size: int

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "quant_type": self.quant_type.value,
            "destination_quant_type": self.destination_quant_type.value,
            "old_shape": list(self.old_shape),
            "new_shape": list(self.new_shape),
            "axis_edits": [item.to_record() for item in self.axis_edits],
            "encoded_size": self.encoded_size,
        }


@dataclass(frozen=True, slots=True)
class GGUFQuantizedMutationPlan:
    physical_plan: PhysicalMutationPlan
    tensor_edits: tuple[GGUFQuantizedTensorEdit, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "mutation_id": self.physical_plan.mutation_plan.request.mutation_id,
            "tensor_edits": [item.to_record() for item in self.tensor_edits],
        }


@dataclass(frozen=True, slots=True)
class AlignedAxisRemovalProposal:
    quant_type: GGMLQuantizationType
    old_size: int
    requested_indices: tuple[int, ...]
    aligned_indices: tuple[int, ...]
    added_indices: tuple[int, ...]

    @property
    def new_size(self) -> int:
        return self.old_size - len(self.aligned_indices)

    def to_axis_removal(self, axis: int = 0) -> AxisRemoval:
        return AxisRemoval(axis, self.aligned_indices)


_NATIVE_WRITE_TYPES = frozenset(
    {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
        GGMLQuantizationType.Q8_0,
        GGMLQuantizationType.Q2_K,
        GGMLQuantizationType.Q3_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
        GGMLQuantizationType.IQ4_NL,
        GGMLQuantizationType.IQ4_XS,
    }
)


def propose_aligned_axis_removal(
    quant_type: GGMLQuantizationType,
    old_size: int,
    requested_indices: tuple[int, ...],
) -> AlignedAxisRemovalProposal:
    """Expand touched contiguous-axis blocks explicitly; never apply the proposal."""

    layout = QUANT_LAYOUTS[quant_type]
    if old_size <= 0 or old_size % layout.block_size:
        raise GGUFAlignmentError(
            f"{quant_type.value} old axis size must be a positive block multiple"
        )
    if (
        not requested_indices
        or requested_indices != tuple(sorted(set(requested_indices)))
        or requested_indices[0] < 0
        or requested_indices[-1] >= old_size
    ):
        raise GGUFAlignmentError("requested removal indices are invalid or non-canonical")
    blocks = {index // layout.block_size for index in requested_indices}
    aligned = tuple(
        index
        for block in sorted(blocks)
        for index in range(
            block * layout.block_size, (block + 1) * layout.block_size
        )
    )
    if len(aligned) == old_size:
        raise GGUFAlignmentError("aligned proposal would remove the entire tensor axis")
    requested = set(requested_indices)
    return AlignedAxisRemovalProposal(
        quant_type,
        old_size,
        requested_indices,
        aligned,
        tuple(index for index in aligned if index not in requested),
    )


def _block_ranges(blocks: tuple[int, ...]) -> tuple[EncodedBlockRange, ...]:
    ranges: list[EncodedBlockRange] = []
    start = previous = blocks[0]
    for block in blocks[1:]:
        if block == previous + 1:
            previous = block
            continue
        ranges.append(EncodedBlockRange(start, previous - start + 1))
        start = previous = block
    ranges.append(EncodedBlockRange(start, previous - start + 1))
    return tuple(ranges)


def _axis_edit(
    edit: PhysicalTensorEdit,
    quant_type: GGMLQuantizationType,
    axis: int,
    removed: tuple[int, ...],
) -> QuantizedAxisEdit:
    constraint = plan_axis_edit(quant_type, edit.old_shape, axis)
    if axis != 0:
        return QuantizedAxisEdit(
            axis,
            QuantizedEditStrategy.WHOLE_SLICE_COPY,
            constraint.index_granularity,
            removed,
        )
    block_size = constraint.index_granularity
    selected = set(removed)
    touched_blocks = tuple(sorted({index // block_size for index in removed}))
    complete = all(
        all(index in selected for index in range(block * block_size, (block + 1) * block_size))
        for block in touched_blocks
    )
    if complete:
        return QuantizedAxisEdit(
            axis,
            QuantizedEditStrategy.DIRECT_BLOCK_COPY,
            block_size,
            removed,
            _block_ranges(touched_blocks),
        )
    first = touched_blocks[0]
    old_blocks = edit.old_shape[0] // block_size
    return QuantizedAxisEdit(
        axis,
        QuantizedEditStrategy.REPACK_CONTIGUOUS_AXIS,
        block_size,
        removed,
        (EncodedBlockRange(first, old_blocks - first),),
    )


def validate_gguf_quantized_plan(
    physical_plan: PhysicalMutationPlan,
    bindings: tuple[GGUFQuantizationBinding, ...],
) -> GGUFQuantizedMutationPlan:
    """Require exact codecs and representable new shapes before any block decode."""

    components = tuple(item.component_id for item in bindings)
    if components != tuple(sorted(set(components))):
        raise GGUFAlignmentError("quantization bindings must be unique and canonical")
    edits = physical_plan.tensor_edits
    edit_components = tuple(edit.component_id for edit in edits)
    if components != edit_components:
        raise GGUFAlignmentError(
            "quantization bindings must exactly cover physical tensor edits"
        )
    binding_by_component = {item.component_id: item for item in bindings}
    validated: list[GGUFQuantizedTensorEdit] = []
    for edit in edits:
        quant_type = binding_by_component[edit.component_id].quant_type
        destination_quant_type = binding_by_component[
            edit.component_id
        ].output_quant_type
        if quant_type not in _NATIVE_WRITE_TYPES:
            raise GGUFAlignmentError(
                f"{quant_type.value} has no exact native write codec; family substitution "
                "is forbidden"
            )
        if destination_quant_type not in _NATIVE_WRITE_TYPES:
            raise GGUFAlignmentError(
                f"{destination_quant_type.value} has no exact native write codec; "
                "family substitution is forbidden"
            )
        try:
            plan_axis_edit(destination_quant_type, edit.new_shape, 0)
        except CodecContractError as error:
            raise GGUFAlignmentError(
                f"new shape {edit.new_shape} for {edit.locator!r} is not representable; "
                "request an explicit aligned axis-removal proposal if semantic expansion "
                "is acceptable"
            ) from error
        axis_edits = tuple(
            _axis_edit(edit, quant_type, transform.axis, transform.removed_indices)
            for transform in edit.transforms
        )
        expected_size = plan_axis_edit(
            destination_quant_type, edit.new_shape, 0
        ).tensor_bytes
        if expected_size != edit.new_storage_bytes:
            raise GGUFAlignmentError(
                f"{edit.locator!r} encoded size {edit.new_storage_bytes} disagrees with "
                f"{destination_quant_type.value} layout size {expected_size}"
            )
        validated.append(
            GGUFQuantizedTensorEdit(
                edit.component_id,
                quant_type,
                destination_quant_type,
                edit.old_shape,
                edit.new_shape,
                axis_edits,
                expected_size,
            )
        )
    return GGUFQuantizedMutationPlan(physical_plan, tuple(validated))

"""Bounded selective GGUF block reads and dequantization for physical edits."""

from __future__ import annotations

from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from operator import mul
from typing import cast

from modelsurgeon.adapters.gguf.quantization import CodecRegistry
from modelsurgeon.adapters.gguf.tensor_reader import GGUFTensorReader
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.gguf_alignment import (
    EncodedBlockRange,
    GGUFQuantizedMutationPlan,
    GGUFQuantizedTensorEdit,
    QuantizedEditStrategy,
)


class SelectiveDequantizationError(ValueError):
    """Raised before an unsafe or over-budget selective block decode."""


class DequantizedPrecision(StrEnum):
    FP32 = "float32"
    FP64 = "float64"

    @property
    def item_size(self) -> int:
        return 4 if self is DequantizedPrecision.FP32 else 8

    @property
    def array_code(self) -> str:
        return "f" if self is DequantizedPrecision.FP32 else "d"


@dataclass(frozen=True, slots=True)
class SelectiveDequantizationLimits:
    max_encoded_chunk_bytes: int = 4 * 1024 * 1024
    max_decoded_values: int = 1_048_576
    max_working_bytes: int = 16 * 1024 * 1024
    precision: DequantizedPrecision = DequantizedPrecision.FP32

    def __post_init__(self) -> None:
        if min(
            self.max_encoded_chunk_bytes,
            self.max_decoded_values,
            self.max_working_bytes,
        ) <= 0:
            raise SelectiveDequantizationError(
                "selective dequantization limits must be positive"
            )


@dataclass(frozen=True, slots=True)
class TouchedGGUFBlockRange:
    component_id: ComponentId
    tensor_name: str
    block_offset: int
    block_count: int
    tensor_byte_offset: int
    byte_count: int


@dataclass(frozen=True, slots=True)
class DecodedGGUFBlockChunk:
    component_id: ComponentId
    tensor_name: str
    block_offset: int
    block_count: int
    element_offset: int
    precision: DequantizedPrecision
    values: array[float]

    @property
    def decoded_bytes(self) -> int:
        return len(self.values) * self.precision.item_size


@dataclass(frozen=True, slots=True)
class SelectiveDequantizationReport:
    complete: bool
    touched_ranges: tuple[TouchedGGUFBlockRange, ...]
    encoded_bytes_read: int
    decoded_values: int
    peak_working_bytes: int


def _row_removed(
    row: int,
    shape: tuple[int, ...],
    removals: dict[int, set[int]],
) -> bool:
    stride = 1
    for axis in range(1, len(shape)):
        coordinate = (row // stride) % shape[axis]
        if coordinate in removals.get(axis, set()):
            return True
        stride *= shape[axis]
    return False


def _decode_spans(edit: GGUFQuantizedTensorEdit) -> Iterator[EncodedBlockRange]:
    repack = next(
        (
            axis
            for axis in edit.axis_edits
            if axis.axis == 0
            and axis.strategy is QuantizedEditStrategy.REPACK_CONTIGUOUS_AXIS
        ),
        None,
    )
    if repack is None:
        return
    outer_removals = {
        axis.axis: set(axis.removed_indices)
        for axis in edit.axis_edits
        if axis.axis > 0
    }
    blocks_per_row = edit.old_shape[0] // repack.index_granularity
    row_count = reduce(mul, edit.old_shape[1:], 1)
    pending_start: int | None = None
    pending_end = 0
    for row in range(row_count):
        if _row_removed(row, edit.old_shape, outer_removals):
            continue
        for local in repack.affected_block_ranges:
            start = row * blocks_per_row + local.block_offset
            end = start + local.block_count
            if pending_start is not None and start == pending_end:
                pending_end = end
                continue
            if pending_start is not None:
                yield EncodedBlockRange(pending_start, pending_end - pending_start)
            pending_start, pending_end = start, end
    if pending_start is not None:
        yield EncodedBlockRange(pending_start, pending_end - pending_start)


class SelectiveGGUFDequantizer:
    """One-use-at-a-time bounded decoder with auditable touched-range reporting."""

    def __init__(self, limits: SelectiveDequantizationLimits | None = None) -> None:
        self.limits = limits or SelectiveDequantizationLimits()
        self._touched: list[TouchedGGUFBlockRange] = []
        self._encoded_bytes = 0
        self._decoded_values = 0
        self._peak_working_bytes = 0
        self._complete = False

    def report(self) -> SelectiveDequantizationReport:
        return SelectiveDequantizationReport(
            self._complete,
            tuple(self._touched),
            self._encoded_bytes,
            self._decoded_values,
            self._peak_working_bytes,
        )

    def _reset(self) -> None:
        self._touched.clear()
        self._encoded_bytes = 0
        self._decoded_values = 0
        self._peak_working_bytes = 0
        self._complete = False

    def iter_chunks(
        self,
        plan: GGUFQuantizedMutationPlan,
        reader: GGUFTensorReader,
        codecs: CodecRegistry,
    ) -> Iterator[DecodedGGUFBlockChunk]:
        self._reset()
        for edit in plan.tensor_edits:
            spans = tuple(_decode_spans(edit))
            if not spans:
                continue
            codec = codecs.resolve(edit.quant_type)
            handle = reader.index.tensor(
                next(
                    physical.locator
                    for physical in plan.physical_plan.tensor_edits
                    if physical.component_id == edit.component_id
                )
            )
            if handle.dimensions != edit.old_shape or handle.quant_type is not edit.quant_type:
                raise SelectiveDequantizationError(
                    f"tensor handle {handle.name!r} disagrees with validated quantized plan"
                )
            bytes_per_block = codec.layout.type_size
            values_per_block = codec.layout.block_size
            combined_per_block = (
                bytes_per_block + values_per_block * self.limits.precision.item_size
            )
            blocks_per_chunk = min(
                self.limits.max_encoded_chunk_bytes // bytes_per_block,
                self.limits.max_decoded_values // values_per_block,
                self.limits.max_working_bytes // combined_per_block,
            )
            if blocks_per_chunk <= 0:
                raise SelectiveDequantizationError(
                    f"configured limits cannot hold one {edit.quant_type.value} block"
                )
            for span in spans:
                offset = span.block_offset
                remaining = span.block_count
                while remaining:
                    count = min(remaining, blocks_per_chunk)
                    chunk = reader.read_blocks(handle, offset, count)
                    values = cast(
                        array[float], array(self.limits.precision.array_code)
                    )
                    operation = codec.decode_blocks(
                        memoryview(chunk.data),
                        values,
                        byte_order=reader.source.container.byte_order,
                    )
                    if operation.block_count != count or len(values) != count * values_per_block:
                        raise SelectiveDequantizationError(
                            f"{edit.quant_type.value} codec returned inconsistent decode counts"
                        )
                    touched = TouchedGGUFBlockRange(
                        edit.component_id,
                        handle.name,
                        offset,
                        count,
                        chunk.tensor_byte_offset,
                        len(chunk.data),
                    )
                    self._touched.append(touched)
                    self._encoded_bytes += len(chunk.data)
                    self._decoded_values += len(values)
                    working = len(chunk.data) + len(values) * self.limits.precision.item_size
                    self._peak_working_bytes = max(self._peak_working_bytes, working)
                    yield DecodedGGUFBlockChunk(
                        edit.component_id,
                        handle.name,
                        offset,
                        count,
                        chunk.element_offset,
                        self.limits.precision,
                        values,
                    )
                    offset += count
                    remaining -= count
        self._complete = True

"""Bounded encoding and validation of changed GGUF float block chunks."""

from __future__ import annotations

import math
from array import array
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from modelsurgeon.adapters.gguf.quantization import (
    ByteOrder,
    CodecRegistry,
    GGMLQuantizationType,
    QuantizationError,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.gguf_alignment import (
    GGUFQuantizedMutationPlan,
    GGUFQuantizedTensorEdit,
)


class SelectiveRequantizationError(ValueError):
    """Raised before emitting malformed, over-budget, or over-error output."""


@dataclass(frozen=True, slots=True)
class GGUFRequantizationLimits:
    max_encoded_chunk_bytes: int = 4 * 1024 * 1024
    max_validation_values: int = 1_048_576
    max_working_bytes: int = 16 * 1024 * 1024
    max_absolute_error: float | None = None
    max_mean_squared_error: float | None = None

    def __post_init__(self) -> None:
        if (
            min(
                self.max_encoded_chunk_bytes,
                self.max_validation_values,
                self.max_working_bytes,
            )
            <= 0
        ):
            raise SelectiveRequantizationError("requantization limits must be positive")
        for value in (self.max_absolute_error, self.max_mean_squared_error):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise SelectiveRequantizationError(
                    "requantization error ceilings must be finite and non-negative"
                )


@dataclass(frozen=True, slots=True)
class ChangedGGUFFloatChunk:
    component_id: ComponentId
    destination_block_offset: int
    values: Sequence[float]

    def __post_init__(self) -> None:
        if self.destination_block_offset < 0 or not self.values:
            raise SelectiveRequantizationError(
                "changed chunks require a non-negative offset and non-empty values"
            )


@dataclass(frozen=True, slots=True)
class GGUFRequantizationErrorSummary:
    component_id: ComponentId
    destination_quant_type: GGMLQuantizationType
    block_offset: int
    block_count: int
    error: QuantizationError


@dataclass(frozen=True, slots=True)
class EncodedChangedGGUFChunk:
    component_id: ComponentId
    destination_quant_type: GGMLQuantizationType
    block_offset: int
    block_count: int
    tensor_byte_offset: int
    payload: bytes
    error_summary: GGUFRequantizationErrorSummary


@dataclass(frozen=True, slots=True)
class TouchedGGUFOutputRange:
    component_id: ComponentId
    destination_quant_type: GGMLQuantizationType
    block_offset: int
    block_count: int
    tensor_byte_offset: int
    byte_count: int


@dataclass(frozen=True, slots=True)
class GGUFRequantizationReport:
    complete: bool
    touched_ranges: tuple[TouchedGGUFOutputRange, ...]
    error_summaries: tuple[GGUFRequantizationErrorSummary, ...]
    encoded_bytes: int
    encoded_blocks: int
    peak_working_bytes: int


class SelectiveGGUFRequantizer:
    """Stream exact-codec changed blocks with structural and error validation."""

    def __init__(self, limits: GGUFRequantizationLimits | None = None) -> None:
        self.limits = limits or GGUFRequantizationLimits()
        self._touched: list[TouchedGGUFOutputRange] = []
        self._summaries: list[GGUFRequantizationErrorSummary] = []
        self._encoded_bytes = 0
        self._encoded_blocks = 0
        self._peak_working_bytes = 0
        self._complete = False

    def _reset(self) -> None:
        self._touched.clear()
        self._summaries.clear()
        self._encoded_bytes = 0
        self._encoded_blocks = 0
        self._peak_working_bytes = 0
        self._complete = False

    def report(self) -> GGUFRequantizationReport:
        return GGUFRequantizationReport(
            self._complete,
            tuple(self._touched),
            tuple(self._summaries),
            self._encoded_bytes,
            self._encoded_blocks,
            self._peak_working_bytes,
        )

    def _check_error(self, summary: GGUFRequantizationErrorSummary) -> None:
        error = summary.error
        if (
            self.limits.max_absolute_error is not None
            and error.max_absolute_error > self.limits.max_absolute_error
        ):
            raise SelectiveRequantizationError(
                f"requantized max error {error.max_absolute_error} exceeds "
                f"ceiling {self.limits.max_absolute_error}"
            )
        if (
            self.limits.max_mean_squared_error is not None
            and error.mean_squared_error > self.limits.max_mean_squared_error
        ):
            raise SelectiveRequantizationError(
                f"requantized mean squared error {error.mean_squared_error} exceeds "
                f"ceiling {self.limits.max_mean_squared_error}"
            )

    def iter_encoded(
        self,
        plan: GGUFQuantizedMutationPlan,
        changed_chunks: Iterable[ChangedGGUFFloatChunk],
        codecs: CodecRegistry,
        *,
        byte_order: ByteOrder,
    ) -> Iterator[EncodedChangedGGUFChunk]:
        self._reset()
        edits = {edit.component_id: edit for edit in plan.tensor_edits}
        yield from self._iter_encoded_edits(edits, changed_chunks, codecs, byte_order)

    def iter_encoded_tensor(
        self,
        edit: GGUFQuantizedTensorEdit,
        changed_chunks: Iterable[ChangedGGUFFloatChunk],
        codecs: CodecRegistry,
        *,
        byte_order: ByteOrder,
    ) -> Iterator[EncodedChangedGGUFChunk]:
        """Encode one complete selected tensor without requiring an axis-removal plan."""

        self._reset()
        yield from self._iter_encoded_edits(
            {edit.component_id: edit}, changed_chunks, codecs, byte_order
        )

    def _iter_encoded_edits(
        self,
        edits: dict[ComponentId, GGUFQuantizedTensorEdit],
        changed_chunks: Iterable[ChangedGGUFFloatChunk],
        codecs: CodecRegistry,
        byte_order: ByteOrder,
    ) -> Iterator[EncodedChangedGGUFChunk]:
        last_end: dict[ComponentId, int] = {}
        for changed in changed_chunks:
            try:
                edit = edits[changed.component_id]
            except KeyError as error:
                raise SelectiveRequantizationError(
                    f"changed component {changed.component_id} is absent from quantized plan"
                ) from error
            codec = codecs.resolve(edit.destination_quant_type)
            block_values = codec.layout.block_size
            block_bytes = codec.layout.type_size
            if len(changed.values) % block_values:
                raise SelectiveRequantizationError(
                    f"changed value count {len(changed.values)} is not a complete "
                    f"{edit.destination_quant_type.value} block multiple"
                )
            changed_blocks = len(changed.values) // block_values
            total_values = math.prod(edit.new_shape)
            if total_values % block_values:
                raise SelectiveRequantizationError("validated output shape is not block aligned")
            total_blocks = total_values // block_values
            start = changed.destination_block_offset
            if start > total_blocks - changed_blocks:
                raise SelectiveRequantizationError(
                    f"changed destination block range escapes output block count {total_blocks}"
                )
            if start < last_end.get(changed.component_id, 0):
                raise SelectiveRequantizationError(
                    "changed destination block chunks overlap or are out of order"
                )
            max_blocks = min(
                self.limits.max_encoded_chunk_bytes // block_bytes,
                self.limits.max_validation_values // block_values,
                self.limits.max_working_bytes // (2 * block_bytes + 8 * block_values),
            )
            if max_blocks <= 0:
                raise SelectiveRequantizationError(
                    f"configured limits cannot encode and validate one "
                    f"{edit.destination_quant_type.value} block"
                )
            consumed_blocks = 0
            while consumed_blocks < changed_blocks:
                block_count = min(max_blocks, changed_blocks - consumed_blocks)
                value_start = consumed_blocks * block_values
                value_end = value_start + block_count * block_values
                values = changed.values[value_start:value_end]
                encoded = bytearray(block_count * block_bytes)
                operation = codec.encode_blocks(values, memoryview(encoded), byte_order=byte_order)
                validation = codec.validate_blocks(memoryview(encoded), byte_order=byte_order)
                validation.require_valid()
                if (
                    operation.block_count != block_count
                    or operation.element_count != len(values)
                    or operation.byte_count != len(encoded)
                    or validation.block_count != block_count
                ):
                    raise SelectiveRequantizationError(
                        "codec operation counts disagree with changed output chunk"
                    )
                decoded = array("d")
                decoded_operation = codec.decode_blocks(
                    memoryview(encoded), decoded, byte_order=byte_order
                )
                if decoded_operation.block_count != block_count or len(decoded) != len(values):
                    raise SelectiveRequantizationError(
                        "validation decode shape disagrees with encoded output chunk"
                    )
                block_offset = start + consumed_blocks
                summary = GGUFRequantizationErrorSummary(
                    changed.component_id,
                    edit.destination_quant_type,
                    block_offset,
                    block_count,
                    codec.estimate_error(values, decoded),
                )
                self._check_error(summary)
                payload = bytes(encoded)
                expected_bytes = block_count * block_bytes
                if len(payload) != expected_bytes:
                    raise SelectiveRequantizationError(
                        "encoded payload size disagrees with codec block layout"
                    )
                tensor_byte_offset = block_offset * block_bytes
                touched = TouchedGGUFOutputRange(
                    changed.component_id,
                    edit.destination_quant_type,
                    block_offset,
                    block_count,
                    tensor_byte_offset,
                    len(payload),
                )
                self._touched.append(touched)
                self._summaries.append(summary)
                self._encoded_bytes += len(payload)
                self._encoded_blocks += block_count
                working = 2 * len(payload) + len(decoded) * 8
                self._peak_working_bytes = max(self._peak_working_bytes, working)
                yield EncodedChangedGGUFChunk(
                    changed.component_id,
                    edit.destination_quant_type,
                    block_offset,
                    block_count,
                    tensor_byte_offset,
                    payload,
                    summary,
                )
                consumed_blocks += block_count
            last_end[changed.component_id] = start + changed_blocks
        self._complete = True

"""Matched selective requantization controls and evaluation delta attribution."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field

from modelsurgeon.adapters.gguf import (
    CodecRegistry,
    GGUFTensorReader,
    plan_storage_axis_edit,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.gguf_alignment import (
    GGUFQuantizedMutationPlan,
    GGUFQuantizedTensorEdit,
)
from modelsurgeon.surgery.selective_dequant import (
    SelectiveDequantizationLimits,
    SelectiveDequantizationReport,
    SelectiveGGUFDequantizer,
)
from modelsurgeon.surgery.selective_requant import (
    ChangedGGUFFloatChunk,
    EncodedChangedGGUFChunk,
    GGUFRequantizationLimits,
    GGUFRequantizationReport,
    SelectiveGGUFRequantizer,
)


class MatchedRequantizationControlError(ValueError):
    """Raised when a no-surgery control cannot exactly match surgery codec ranges."""


@dataclass(frozen=True, slots=True)
class CodecAlignedControlRange:
    component_id: ComponentId
    block_offset: int
    block_count: int

    def __post_init__(self) -> None:
        if self.block_offset < 0 or self.block_count <= 0:
            raise MatchedRequantizationControlError(
                "matched codec ranges must be non-negative and non-empty"
            )


@dataclass(frozen=True, slots=True)
class MatchedRequantizationControlLimits:
    dequantization: SelectiveDequantizationLimits = field(
        default_factory=SelectiveDequantizationLimits
    )
    requantization: GGUFRequantizationLimits = field(
        default_factory=GGUFRequantizationLimits
    )


@dataclass(frozen=True, slots=True)
class MatchedRequantizationControlReport:
    complete: bool
    seed: int
    matched_ranges: tuple[CodecAlignedControlRange, ...]
    dequantization: SelectiveDequantizationReport
    requantization: GGUFRequantizationReport

    def to_record(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "seed": self.seed,
            "matched_ranges": [
                {
                    "component_id": str(item.component_id),
                    "block_offset": item.block_offset,
                    "block_count": item.block_count,
                }
                for item in self.matched_ranges
            ],
            "encoded_bytes_read": self.dequantization.encoded_bytes_read,
            "decoded_values": self.dequantization.decoded_values,
            "encoded_bytes_written": self.requantization.encoded_bytes,
            "encoded_blocks_written": self.requantization.encoded_blocks,
            "peak_dequantization_working_bytes": (
                self.dequantization.peak_working_bytes
            ),
            "peak_requantization_working_bytes": (
                self.requantization.peak_working_bytes
            ),
        }


@dataclass(frozen=True, slots=True)
class MatchedRequantizationDeltas:
    """Separate codec noise from structural effect for one scalar metric."""

    metric: str
    baseline: float
    requantization_control: float
    surgery: float

    def __post_init__(self) -> None:
        if not self.metric:
            raise MatchedRequantizationControlError("evaluation metric must be named")
        if not all(
            math.isfinite(value)
            for value in (self.baseline, self.requantization_control, self.surgery)
        ):
            raise MatchedRequantizationControlError(
                "matched evaluation measurements must be finite"
            )

    @property
    def requantization_delta(self) -> float:
        return self.requantization_control - self.baseline

    @property
    def surgery_delta(self) -> float:
        return self.surgery - self.requantization_control

    @property
    def combined_delta(self) -> float:
        return self.surgery - self.baseline

    def to_record(self) -> dict[str, str | float]:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "requantization_control": self.requantization_control,
            "surgery": self.surgery,
            "requantization_delta": self.requantization_delta,
            "surgery_delta": self.surgery_delta,
            "combined_delta": self.combined_delta,
        }


def _control_plan(plan: GGUFQuantizedMutationPlan) -> GGUFQuantizedMutationPlan:
    edits: list[GGUFQuantizedTensorEdit] = []
    for edit in plan.tensor_edits:
        if edit.quant_type is not edit.destination_quant_type:
            raise MatchedRequantizationControlError(
                f"matched control for {edit.component_id} requires the surgery source "
                "and destination codec to be identical"
            )
        encoded_size = plan_storage_axis_edit(
            edit.quant_type, edit.old_shape, 0
        ).tensor_bytes
        edits.append(
            GGUFQuantizedTensorEdit(
                edit.component_id,
                edit.quant_type,
                edit.quant_type,
                edit.old_shape,
                edit.old_shape,
                edit.axis_edits,
                encoded_size,
            )
        )
    return GGUFQuantizedMutationPlan(plan.physical_plan, tuple(edits))


def _normalize_ranges(
    ranges: list[tuple[ComponentId, int, int]],
) -> tuple[CodecAlignedControlRange, ...]:
    output: list[CodecAlignedControlRange] = []
    for component_id, offset, count in sorted(ranges):
        current = CodecAlignedControlRange(component_id, offset, count)
        if output and output[-1].component_id == component_id:
            previous = output[-1]
            previous_end = previous.block_offset + previous.block_count
            if offset < previous_end:
                raise MatchedRequantizationControlError(
                    f"matched control ranges overlap for {component_id}"
                )
            if offset == previous_end:
                output[-1] = CodecAlignedControlRange(
                    component_id,
                    previous.block_offset,
                    previous.block_count + count,
                )
                continue
        output.append(current)
    return tuple(output)


class MatchedGGUFRequantizationControl:
    """Re-encode the exact structural-surgery source ranges without surgery."""

    def __init__(
        self,
        *,
        seed: int = 0,
        limits: MatchedRequantizationControlLimits | None = None,
    ) -> None:
        if isinstance(seed, bool) or seed < 0 or seed >= 1 << 64:
            raise MatchedRequantizationControlError(
                "matched requantization seed must be an unsigned 64-bit integer"
            )
        self.seed = seed
        self.limits = limits or MatchedRequantizationControlLimits()
        self._dequantizer = SelectiveGGUFDequantizer(self.limits.dequantization)
        self._requantizer = SelectiveGGUFRequantizer(self.limits.requantization)
        self._matched_ranges: tuple[CodecAlignedControlRange, ...] = ()
        self._complete = False

    def report(self) -> MatchedRequantizationControlReport:
        return MatchedRequantizationControlReport(
            self._complete,
            self.seed,
            self._matched_ranges,
            self._dequantizer.report(),
            self._requantizer.report(),
        )

    def iter_encoded(
        self,
        plan: GGUFQuantizedMutationPlan,
        reader: GGUFTensorReader,
        codecs: CodecRegistry,
    ) -> Iterator[EncodedChangedGGUFChunk]:
        self._matched_ranges = ()
        self._complete = False
        control = _control_plan(plan)
        decoded = self._dequantizer.iter_chunks(control, reader, codecs)
        changed = (
            ChangedGGUFFloatChunk(
                chunk.component_id,
                chunk.block_offset,
                chunk.values,
            )
            for chunk in decoded
        )
        yield from self._requantizer.iter_encoded(
            control,
            changed,
            codecs,
            byte_order=reader.source.container.byte_order,
        )
        decoded_ranges = _normalize_ranges(
            [
                (item.component_id, item.block_offset, item.block_count)
                for item in self._dequantizer.report().touched_ranges
            ]
        )
        encoded_ranges = _normalize_ranges(
            [
                (item.component_id, item.block_offset, item.block_count)
                for item in self._requantizer.report().touched_ranges
            ]
        )
        if decoded_ranges != encoded_ranges:
            raise MatchedRequantizationControlError(
                "requantization control did not encode the exact decoded codec ranges"
            )
        self._matched_ranges = decoded_ranges
        self._complete = True

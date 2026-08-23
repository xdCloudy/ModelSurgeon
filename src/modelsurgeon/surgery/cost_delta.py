"""Attach analytical parameter, FLOP, and activation-memory deltas to mutation plans."""

from __future__ import annotations

from dataclasses import dataclass, replace

from modelsurgeon.features.cost_model import CostModelReport
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.contracts import MutationDelta, MutationPlan

MUTATION_COST_DELTA_VERSION = "1"


class MutationCostDeltaError(ValueError):
    """Raised when cost reports cannot support a comparable mutation delta."""


@dataclass(frozen=True, slots=True)
class AffectedShape:
    component_id: ComponentId
    old_shape: tuple[int, ...]
    new_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.old_shape or not self.new_shape:
            raise MutationCostDeltaError("affected shapes must be non-empty")
        if any(value <= 0 for value in (*self.old_shape, *self.new_shape)):
            raise MutationCostDeltaError("affected shapes must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "old_shape": list(self.old_shape),
            "new_shape": list(self.new_shape),
        }


@dataclass(frozen=True, slots=True)
class MutationCostDeltaReport:
    plan: MutationPlan
    mask_only: bool
    affected_shapes: tuple[AffectedShape, ...]
    parameter_delta: int
    forward_flop_delta: int
    output_activation_delta_bytes: int
    peak_working_activation_delta_bytes: int
    assumptions: tuple[tuple[str, str | int], ...]
    version: str = MUTATION_COST_DELTA_VERSION

    def __post_init__(self) -> None:
        components = tuple(item.component_id for item in self.affected_shapes)
        if components != tuple(sorted(components)) or len(components) != len(set(components)):
            raise MutationCostDeltaError("affected shapes must use unique canonical component IDs")
        if not set(components).issubset(self.plan.affected_components):
            raise MutationCostDeltaError("affected shapes must belong to the mutation closure")
        if self.mask_only and any(
            value != 0
            for value in (
                self.parameter_delta,
                self.forward_flop_delta,
                self.output_activation_delta_bytes,
                self.peak_working_activation_delta_bytes,
                self.plan.expected_delta.storage_bytes,
            )
        ):
            raise MutationCostDeltaError("mask-only mutations cannot claim physical cost reductions")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "mutation_id": self.plan.request.mutation_id,
            "mask_only": self.mask_only,
            "expected_delta": {
                "parameters": self.plan.expected_delta.parameters,
                "flops": self.plan.expected_delta.flops,
                "memory_bytes": self.plan.expected_delta.memory_bytes,
                "storage_bytes": self.plan.expected_delta.storage_bytes,
            },
            "peak_working_activation_delta_bytes": self.peak_working_activation_delta_bytes,
            "affected_shapes": [item.to_record() for item in self.affected_shapes],
            "assumptions": dict(self.assumptions),
        }


def _totals(report: CostModelReport) -> tuple[int, int, int, int]:
    return (
        report.specified_parameter_count,
        sum(item.forward_flops for item in report.estimates),
        sum(item.output_activation_bytes for item in report.estimates),
        sum(item.peak_working_activation_bytes for item in report.estimates),
    )


def _assumption_record(report: CostModelReport) -> tuple[tuple[str, str | int], ...]:
    values: dict[str, str | int] = {
        **report.assumptions.to_record(),
        "batch_size": report.sequence_shape.batch_size,
        "sequence_length": report.sequence_shape.sequence_length,
        "delta_sign": "candidate_minus_baseline",
        "activation_memory_delta": "sum_of_component_output_activation_bytes",
        "peak_working_delta": "sum_of_component_peak_working_activation_bytes",
    }
    return tuple(sorted(values.items()))


def attach_expected_cost_delta(
    plan: MutationPlan,
    baseline: CostModelReport,
    *,
    affected_shapes: tuple[AffectedShape, ...],
    candidate: CostModelReport | None = None,
    mask_only: bool = False,
) -> MutationCostDeltaReport:
    """Return a plan carrying comparable analytical deltas and explicit assumptions."""

    if not affected_shapes:
        raise MutationCostDeltaError("mutation cost deltas require affected shape evidence")
    assumptions = _assumption_record(baseline)
    if mask_only:
        delta = MutationDelta()
        updated = replace(plan, expected_delta=delta)
        return MutationCostDeltaReport(
            updated,
            True,
            affected_shapes,
            0,
            0,
            0,
            0,
            assumptions,
        )

    if candidate is None:
        raise MutationCostDeltaError("physical mutations require a candidate cost report")
    if baseline.sequence_shape != candidate.sequence_shape:
        raise MutationCostDeltaError("baseline and candidate sequence shapes are not comparable")
    if baseline.assumptions != candidate.assumptions:
        raise MutationCostDeltaError("baseline and candidate cost assumptions are not comparable")

    before = _totals(baseline)
    after = _totals(candidate)
    parameter_delta = after[0] - before[0]
    flop_delta = after[1] - before[1]
    output_memory_delta = after[2] - before[2]
    peak_memory_delta = after[3] - before[3]
    delta = MutationDelta(
        parameters=parameter_delta,
        flops=flop_delta,
        memory_bytes=output_memory_delta,
        storage_bytes=plan.expected_delta.storage_bytes,
    )
    updated = replace(plan, expected_delta=delta)
    return MutationCostDeltaReport(
        updated,
        False,
        affected_shapes,
        parameter_delta,
        flop_delta,
        output_memory_delta,
        peak_memory_delta,
        assumptions,
    )

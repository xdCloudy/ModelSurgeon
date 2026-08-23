"""Deterministic analytical parameter, FLOP, and activation-memory estimates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.graph import ComponentGraph, ComponentId

COST_MODEL_VERSION = "1"


class CostModelError(ValueError):
    """Raised when analytical cost estimates would violate their declared assumptions."""


class CostOperation(StrEnum):
    LINEAR = "linear"
    ATTENTION = "attention"
    GATED_MLP = "gated_mlp"
    ELEMENTWISE = "elementwise"
    EMBEDDING = "embedding"


@dataclass(frozen=True, slots=True)
class SequenceShape:
    batch_size: int
    sequence_length: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.sequence_length <= 0:
            raise CostModelError("batch size and sequence length must be positive")


@dataclass(frozen=True, slots=True)
class CostAssumptions:
    multiply_add_flops: int = 2
    activation_bytes_per_element: int = 2
    elementwise_flops_per_element: int = 1

    def __post_init__(self) -> None:
        if (
            self.multiply_add_flops <= 0
            or self.activation_bytes_per_element <= 0
            or self.elementwise_flops_per_element <= 0
        ):
            raise CostModelError("cost-model assumptions must be positive")

    def to_record(self) -> dict[str, int | str]:
        return {
            "multiply_add_flops": self.multiply_add_flops,
            "activation_bytes_per_element": self.activation_bytes_per_element,
            "elementwise_flops_per_element": self.elementwise_flops_per_element,
            "attention_memory_model": "qkv+output+full_score_matrix",
            "mlp_memory_model": "gate+up+output",
        }


@dataclass(frozen=True, slots=True)
class ComponentCostSpec:
    component_id: ComponentId
    operation: CostOperation
    parameter_count: int
    input_width: int
    output_width: int
    intermediate_width: int | None = None
    heads: int | None = None
    head_dim: int | None = None

    def __post_init__(self) -> None:
        if self.parameter_count < 0:
            raise CostModelError("component parameter count cannot be negative")
        if self.input_width <= 0 or self.output_width <= 0:
            raise CostModelError("component input/output widths must be positive")
        if self.operation is CostOperation.GATED_MLP:
            if self.intermediate_width is None or self.intermediate_width <= 0:
                raise CostModelError("gated MLP cost requires a positive intermediate width")
        if self.operation is CostOperation.ATTENTION:
            if self.heads is None or self.heads <= 0 or self.head_dim is None or self.head_dim <= 0:
                raise CostModelError("attention cost requires positive heads and head_dim")


@dataclass(frozen=True, slots=True)
class ComponentCostEstimate:
    component_id: ComponentId
    operation: CostOperation
    parameter_count: int
    forward_flops: int
    output_activation_bytes: int
    peak_working_activation_bytes: int

    def to_record(self) -> dict[str, str | int]:
        return {
            "component_id": str(self.component_id),
            "operation": self.operation.value,
            "parameter_count": self.parameter_count,
            "forward_flops": self.forward_flops,
            "output_activation_bytes": self.output_activation_bytes,
            "peak_working_activation_bytes": self.peak_working_activation_bytes,
        }


@dataclass(frozen=True, slots=True)
class CostModelReport:
    version: str
    sequence_shape: SequenceShape
    assumptions: CostAssumptions
    graph_parameter_count: int
    specified_parameter_count: int
    estimates: tuple[ComponentCostEstimate, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "sequence_shape": {
                "batch_size": self.sequence_shape.batch_size,
                "sequence_length": self.sequence_shape.sequence_length,
            },
            "assumptions": self.assumptions.to_record(),
            "graph_parameter_count": self.graph_parameter_count,
            "specified_parameter_count": self.specified_parameter_count,
            "parameter_counts_reconciled": (
                self.graph_parameter_count == self.specified_parameter_count
            ),
            "estimates": [estimate.to_record() for estimate in self.estimates],
        }


def graph_parameter_count(graph: ComponentGraph) -> int:
    """Return the exact parameter total recorded by canonical parameter nodes."""

    total = 0
    for node in graph.nodes:
        if node.kind != "parameter":
            continue
        value = dict(node.attributes).get("element_count")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CostModelError(
                f"parameter node {node.component_id} requires a non-negative integer element_count"
            )
        total += value
    return total


def _estimate(
    spec: ComponentCostSpec,
    shape: SequenceShape,
    assumptions: CostAssumptions,
) -> ComponentCostEstimate:
    batch = shape.batch_size
    tokens = shape.sequence_length
    input_width = spec.input_width
    output_width = spec.output_width
    bytes_per_element = assumptions.activation_bytes_per_element
    madd = assumptions.multiply_add_flops

    output_elements = batch * tokens * output_width
    output_bytes = output_elements * bytes_per_element

    if spec.operation is CostOperation.LINEAR:
        flops = madd * batch * tokens * input_width * output_width
        peak_bytes = output_bytes
    elif spec.operation is CostOperation.EMBEDDING:
        flops = 0
        peak_bytes = output_bytes
    elif spec.operation is CostOperation.ELEMENTWISE:
        flops = assumptions.elementwise_flops_per_element * output_elements
        peak_bytes = output_bytes
    elif spec.operation is CostOperation.GATED_MLP:
        intermediate = spec.intermediate_width
        if intermediate is None:
            raise CostModelError("gated MLP intermediate width is missing")
        flops = madd * batch * tokens * (2 * input_width * intermediate + intermediate * output_width)
        working_elements = batch * tokens * (2 * intermediate + output_width)
        peak_bytes = working_elements * bytes_per_element
    else:
        heads = spec.heads
        head_dim = spec.head_dim
        if heads is None or head_dim is None:
            raise CostModelError("attention geometry is missing")
        projection_flops = madd * batch * tokens * 4 * input_width * output_width
        attention_flops = madd * batch * heads * tokens * tokens * head_dim * 2
        flops = projection_flops + attention_flops
        working_elements = batch * (4 * tokens * output_width + heads * tokens * tokens)
        peak_bytes = working_elements * bytes_per_element

    return ComponentCostEstimate(
        component_id=spec.component_id,
        operation=spec.operation,
        parameter_count=spec.parameter_count,
        forward_flops=flops,
        output_activation_bytes=output_bytes,
        peak_working_activation_bytes=peak_bytes,
    )


def estimate_component_costs(
    graph: ComponentGraph,
    specs: tuple[ComponentCostSpec, ...],
    sequence_shape: SequenceShape,
    assumptions: CostAssumptions | None = None,
    *,
    require_parameter_reconciliation: bool = True,
) -> CostModelReport:
    """Estimate supported component costs and optionally require exact parameter reconciliation."""

    if not specs:
        raise CostModelError("cost-model estimation requires at least one component spec")
    ids = [spec.component_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise CostModelError("component cost specs must have unique component IDs")
    known = {node.component_id for node in graph.nodes}
    unknown = sorted(component_id for component_id in ids if component_id not in known)
    if unknown:
        raise CostModelError(f"component cost spec references unknown graph node: {unknown[0]}")

    resolved = assumptions or CostAssumptions()
    exact_parameter_count = graph_parameter_count(graph)
    specified_parameter_count = sum(spec.parameter_count for spec in specs)
    if require_parameter_reconciliation and specified_parameter_count != exact_parameter_count:
        raise CostModelError(
            "component parameter counts do not reconcile with graph: "
            f"specified={specified_parameter_count}, graph={exact_parameter_count}"
        )

    estimates = tuple(
        _estimate(spec, sequence_shape, resolved)
        for spec in sorted(specs, key=lambda item: item.component_id)
    )
    return CostModelReport(
        COST_MODEL_VERSION,
        sequence_shape,
        resolved,
        exact_parameter_count,
        specified_parameter_count,
        estimates,
    )

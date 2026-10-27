"""Bounded Kronecker-factored local costs for structured linear-weight removal.

This is a research scoring primitive, not an optimizer or artifact surgery path.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from typing import Any

KFAC_SENSITIVITY_VERSION = "1"


class KroneckerSensitivityError(ValueError):
    """Raised when KFAC factors or structured-removal candidates are invalid."""


class LinearRemovalAxis(StrEnum):
    OUTPUT_ROWS = "output_rows"
    INPUT_COLUMNS = "input_columns"


@dataclass(frozen=True, slots=True)
class StructuredRemovalCost:
    candidate_index: int
    indices: tuple[int, ...]
    estimated_loss_increase: float
    removed_parameter_count: int


@dataclass(frozen=True, slots=True)
class KroneckerSensitivityReport:
    axis: LinearRemovalAxis
    method: str
    input_width: int
    output_width: int
    candidate_costs: tuple[StructuredRemovalCost, ...]
    estimated_peak_workspace_bytes: int
    workspace_budget_bytes: int
    compensation_applied: bool = False

    def to_record(self) -> dict[str, object]:
        return {
            "version": KFAC_SENSITIVITY_VERSION,
            "method": self.method,
            "axis": self.axis.value,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "candidate_costs": [
                {
                    "candidate_index": item.candidate_index,
                    "indices": list(item.indices),
                    "estimated_loss_increase": item.estimated_loss_increase,
                    "removed_parameter_count": item.removed_parameter_count,
                }
                for item in self.candidate_costs
            ],
            "estimated_peak_workspace_bytes": self.estimated_peak_workspace_bytes,
            "workspace_budget_bytes": self.workspace_budget_bytes,
            "compensation_applied": self.compensation_applied,
        }


def _numpy() -> Any:
    try:
        return import_module("numpy")
    except ModuleNotFoundError as error:
        raise KroneckerSensitivityError(
            "KFAC sensitivity requires NumPy; install the ModelSurgeon runtime dependencies"
        ) from error


def _matrix_shape(values: object, name: str) -> tuple[int, int]:
    shape = getattr(values, "shape", None)
    if shape is not None:
        try:
            dimensions = tuple(int(value) for value in shape)
        except (TypeError, ValueError, OverflowError) as error:
            raise KroneckerSensitivityError(f"{name} shape is invalid") from error
        if len(dimensions) != 2 or min(dimensions) <= 0:
            raise KroneckerSensitivityError(f"{name} must be a non-empty rank-two matrix")
        return dimensions[0], dimensions[1]
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise KroneckerSensitivityError(f"{name} must be a non-empty rank-two matrix")
    first_row = values[0]
    if not isinstance(first_row, Sequence) or isinstance(first_row, (str, bytes)):
        raise KroneckerSensitivityError(f"{name} must be a rectangular numeric matrix")
    columns = len(first_row)
    if columns <= 0 or any(
        not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != columns
        for row in values
    ):
        raise KroneckerSensitivityError(f"{name} must be a rectangular numeric matrix")
    return len(values), columns


def _as_symmetric_matrix(np: Any, values: object, name: str) -> Any:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise KroneckerSensitivityError(f"{name} must be a rectangular numeric matrix") from error
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise KroneckerSensitivityError(f"{name} must be a non-empty square matrix")
    if any(not bool(np.isfinite(row).all()) for row in matrix):
        raise KroneckerSensitivityError(f"{name} must contain only finite values")
    for row_index in range(matrix.shape[0]):
        if not bool(
            np.allclose(
                matrix[row_index, :],
                matrix[:, row_index],
                rtol=1e-8,
                atol=1e-10,
            )
        ):
            raise KroneckerSensitivityError(f"{name} must be symmetric")
    return matrix


def _candidate_groups(
    groups: Sequence[Sequence[int]], dimension: int
) -> tuple[tuple[int, ...], ...]:
    if not groups:
        raise KroneckerSensitivityError("at least one candidate group is required")
    resolved: list[tuple[int, ...]] = []
    for candidate_index, group in enumerate(groups):
        indices = tuple(group)
        if not indices:
            raise KroneckerSensitivityError(f"candidate group {candidate_index} is empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= dimension
            for index in indices
        ):
            raise KroneckerSensitivityError(
                f"candidate group {candidate_index} contains an out-of-range index"
            )
        if indices != tuple(sorted(set(indices))):
            raise KroneckerSensitivityError(
                f"candidate group {candidate_index} must be sorted and duplicate-free"
            )
        resolved.append(indices)
    return tuple(resolved)


def score_structured_linear_removals(
    weights: Sequence[Sequence[float]],
    activation_second_moment: Sequence[Sequence[float]],
    output_gradient_second_moment: Sequence[Sequence[float]],
    candidate_groups: Sequence[Sequence[int]],
    *,
    axis: LinearRemovalAxis,
    max_workspace_bytes: int = 256 * 1024 * 1024,
) -> KroneckerSensitivityReport:
    """Score row or column deletion under a Kronecker-factored quadratic loss.

    For a linear weight matrix ``W`` and removal perturbation ``D`` this computes
    ``0.5 * tr(G @ D @ A @ D.T)``, where ``A=E[x x.T]`` and
    ``G=E[g g.T]`` are caller-collected second moments. This is the uncorrected
    local KFAC cost; it does not perform LLM-Surgeon-style compensation, mutate
    weights, collect moments, or claim held-out quality prediction.

    The implementation is CPU-only and bounds its NumPy working arrays before
    allocation. Callers must provide raw second moments (not mean-centered
    covariances) from independent calibration observations.
    """

    if not isinstance(axis, LinearRemovalAxis):
        raise KroneckerSensitivityError("axis must be a LinearRemovalAxis value")
    if (
        isinstance(max_workspace_bytes, bool)
        or not isinstance(max_workspace_bytes, int)
        or max_workspace_bytes <= 0
    ):
        raise KroneckerSensitivityError("workspace budget must be a positive integer")
    output_width, input_width = _matrix_shape(weights, "weights")
    activation_shape = _matrix_shape(activation_second_moment, "activation second moment")
    output_gradient_shape = _matrix_shape(
        output_gradient_second_moment, "output-gradient second moment"
    )
    if activation_shape != (input_width, input_width):
        raise KroneckerSensitivityError("activation second moment shape must match input width")
    if output_gradient_shape != (output_width, output_width):
        raise KroneckerSensitivityError(
            "output-gradient second moment shape must match output width"
        )
    dimension = output_width if axis is LinearRemovalAxis.OUTPUT_ROWS else input_width
    groups = _candidate_groups(candidate_groups, dimension)
    max_group = max(len(group) for group in groups)

    singleton_groups = max_group == 1
    if singleton_groups:
        # A singleton removal uses only the diagonal of W A W.T or W.T G W.
        # Materializing the full pairwise matrix is unnecessary and quadratic
        # in the pruned width. Include converted inputs and the one projection.
        estimated_workspace = 8 * (
            2 * output_width * input_width
            + input_width * input_width
            + output_width * output_width
            + output_width
            + input_width
        )
    else:
        # Covers float64 copies of the three inputs, both matrix products, the
        # selected group blocks, and reduction scratch; deliberately conservative.
        estimated_workspace = 8 * (
            2 * output_width * input_width
            + 3 * input_width * input_width
            + 3 * output_width * output_width
            + 3 * max_group * max_group
        )
    if estimated_workspace > max_workspace_bytes:
        raise KroneckerSensitivityError(
            f"planned KFAC workspace {estimated_workspace} exceeds budget "
            f"{max_workspace_bytes} bytes"
        )

    np = _numpy()
    try:
        matrix = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise KroneckerSensitivityError("weights must be a rectangular numeric matrix") from error
    if any(not bool(np.isfinite(row).all()) for row in matrix):
        raise KroneckerSensitivityError("weights must contain only finite values")
    activation = _as_symmetric_matrix(np, activation_second_moment, "activation second moment")
    output_gradient = _as_symmetric_matrix(
        np, output_gradient_second_moment, "output-gradient second moment"
    )
    if singleton_groups and axis is LinearRemovalAxis.OUTPUT_ROWS:
        projected_weights = matrix @ activation
        diagonal_cost = np.sum(projected_weights * matrix, axis=1) * np.diag(output_gradient)
        parameter_count = input_width
    elif singleton_groups:
        projected_weights = output_gradient @ matrix
        diagonal_cost = np.sum(projected_weights * matrix, axis=0) * np.diag(activation)
        parameter_count = output_width
    elif axis is LinearRemovalAxis.OUTPUT_ROWS:
        pairwise_cost = (matrix @ activation) @ matrix.T
        pairwise_cost = output_gradient * pairwise_cost.T
        parameter_count = input_width
    else:
        pairwise_cost = (matrix.T @ output_gradient) @ matrix
        pairwise_cost = activation * pairwise_cost.T
        parameter_count = output_width

    candidate_costs: list[StructuredRemovalCost] = []
    for candidate_index, indices in enumerate(groups):
        if singleton_groups:
            selected_cost = float(diagonal_cost[indices[0]])
            loss_increase = 0.5 * selected_cost
            scale = max(1.0, abs(selected_cost))
        else:
            selected = pairwise_cost[np.ix_(indices, indices)]
            loss_increase = 0.5 * float(np.sum(selected, dtype=np.float64))
            scale = max(1.0, float(np.max(np.abs(selected))))
        if not math.isfinite(loss_increase) or loss_increase < -1e-10 * scale:
            raise KroneckerSensitivityError(
                "KFAC removal cost is negative or non-finite; supplied factors may not be PSD"
            )
        candidate_costs.append(
            StructuredRemovalCost(
                candidate_index,
                indices,
                max(0.0, loss_increase),
                len(indices) * parameter_count,
            )
        )

    return KroneckerSensitivityReport(
        axis=axis,
        method="uncorrected_kfac_quadratic_removal_cost",
        input_width=input_width,
        output_width=output_width,
        candidate_costs=tuple(candidate_costs),
        estimated_peak_workspace_bytes=estimated_workspace,
        workspace_budget_bytes=max_workspace_bytes,
    )

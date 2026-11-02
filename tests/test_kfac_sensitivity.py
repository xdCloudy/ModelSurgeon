from __future__ import annotations

import numpy as np
import pytest

from modelsurgeon.features.kfac_sensitivity import (
    KroneckerSensitivityError,
    LinearRemovalAxis,
    score_structured_linear_removals,
)


def _dense_cost(
    weights: np.ndarray,
    activation: np.ndarray,
    output_gradient: np.ndarray,
    *,
    axis: LinearRemovalAxis,
    indices: tuple[int, ...],
) -> float:
    delta = np.zeros_like(weights)
    if axis is LinearRemovalAxis.OUTPUT_ROWS:
        delta[list(indices), :] = -weights[list(indices), :]
    else:
        delta[:, list(indices)] = -weights[:, list(indices)]
    hessian = np.kron(output_gradient, activation)
    flattened = delta.reshape(-1)
    return 0.5 * float(flattened @ hessian @ flattened)


@pytest.mark.parametrize(
    ("axis", "groups"),
    [
        (LinearRemovalAxis.OUTPUT_ROWS, ((0,), (0, 2))),
        (LinearRemovalAxis.INPUT_COLUMNS, ((1,), (0, 1))),
    ],
)
def test_structured_cost_matches_dense_kronecker_quadratic(
    axis: LinearRemovalAxis,
    groups: tuple[tuple[int, ...], ...],
) -> None:
    weights = np.asarray(
        [[0.3, -0.7, 0.2], [0.6, 0.1, -0.4], [-0.2, 0.8, 0.5]],
        dtype=np.float64,
    )
    activation = np.asarray(
        [[2.0, 0.2, 0.1], [0.2, 1.5, -0.1], [0.1, -0.1, 1.2]],
        dtype=np.float64,
    )
    output_gradient = np.asarray(
        [[1.4, 0.1, -0.1], [0.1, 1.1, 0.2], [-0.1, 0.2, 1.8]],
        dtype=np.float64,
    )

    report = score_structured_linear_removals(
        weights,
        activation,
        output_gradient,
        groups,
        axis=axis,
    )

    assert report.compensation_applied is False
    assert report.estimated_peak_workspace_bytes <= report.workspace_budget_bytes
    for result, group in zip(report.candidate_costs, groups, strict=True):
        assert result.estimated_loss_increase == pytest.approx(
            _dense_cost(
                weights,
                activation,
                output_gradient,
                axis=axis,
                indices=group,
            ),
            rel=1e-12,
            abs=1e-12,
        )


@pytest.mark.parametrize("axis", [LinearRemovalAxis.OUTPUT_ROWS, LinearRemovalAxis.INPUT_COLUMNS])
def test_singleton_groups_match_dense_quadratic_without_pairwise_workspace(
    axis: LinearRemovalAxis,
) -> None:
    weights = np.asarray(
        [[0.3, -0.7, 0.2], [0.6, 0.1, -0.4], [-0.2, 0.8, 0.5]],
        dtype=np.float64,
    )
    activation = np.asarray(
        [[2.0, 0.2, 0.1], [0.2, 1.5, -0.1], [0.1, -0.1, 1.2]],
        dtype=np.float64,
    )
    output_gradient = np.asarray(
        [[1.4, 0.1, -0.1], [0.1, 1.1, 0.2], [-0.1, 0.2, 1.8]],
        dtype=np.float64,
    )
    groups = ((0,), (1,), (2,))

    report = score_structured_linear_removals(
        weights,
        activation,
        output_gradient,
        groups,
        axis=axis,
    )

    full_pairwise_workspace = 8 * (
        2 * weights.shape[0] * weights.shape[1]
        + 3 * weights.shape[1] * weights.shape[1]
        + 3 * weights.shape[0] * weights.shape[0]
        + 3
    )
    assert report.estimated_peak_workspace_bytes < full_pairwise_workspace
    for result, group in zip(report.candidate_costs, groups, strict=True):
        assert result.estimated_loss_increase == pytest.approx(
            _dense_cost(
                weights,
                activation,
                output_gradient,
                axis=axis,
                indices=group,
            ),
            rel=1e-12,
            abs=1e-12,
        )


def test_cost_record_names_uncompensated_method_and_counts_removed_parameters() -> None:
    report = score_structured_linear_removals(
        [[1.0, 0.0], [0.0, 2.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        ((0, 1),),
        axis=LinearRemovalAxis.OUTPUT_ROWS,
    )

    record = report.to_record()
    assert record["method"] == "uncorrected_kfac_quadratic_removal_cost"
    assert record["compensation_applied"] is False
    assert record["candidate_costs"][0]["removed_parameter_count"] == 4
    assert record["candidate_costs"][0]["estimated_loss_increase"] == pytest.approx(2.5)


def test_rejects_bad_geometry_groups_nonfinite_and_negative_costs() -> None:
    with pytest.raises(KroneckerSensitivityError, match="activation second moment shape"):
        score_structured_linear_removals(
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            ((0,),),
            axis=LinearRemovalAxis.OUTPUT_ROWS,
        )
    with pytest.raises(KroneckerSensitivityError, match="sorted and duplicate-free"):
        score_structured_linear_removals(
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            ((1, 0),),
            axis=LinearRemovalAxis.OUTPUT_ROWS,
        )
    with pytest.raises(KroneckerSensitivityError, match="finite"):
        score_structured_linear_removals(
            [[float("nan")]],
            [[1.0]],
            [[1.0]],
            ((0,),),
            axis=LinearRemovalAxis.OUTPUT_ROWS,
        )
    with pytest.raises(KroneckerSensitivityError, match="may not be PSD"):
        score_structured_linear_removals(
            [[1.0]],
            [[1.0]],
            [[-1.0]],
            ((0,),),
            axis=LinearRemovalAxis.OUTPUT_ROWS,
        )


def test_rejects_workspace_over_budget_before_matrix_products() -> None:
    with pytest.raises(KroneckerSensitivityError, match="exceeds budget"):
        score_structured_linear_removals(
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            ((0,),),
            axis=LinearRemovalAxis.OUTPUT_ROWS,
            max_workspace_bytes=1,
        )


class _ShapeOnlyMatrix:
    shape = (10_000, 10_000)

    def __array__(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("workspace must be preflighted before allocating matrices")


def test_workspace_budget_is_preflighted_before_array_conversion() -> None:
    oversized = _ShapeOnlyMatrix()
    with pytest.raises(KroneckerSensitivityError, match="exceeds budget"):
        score_structured_linear_removals(
            oversized,
            oversized,
            oversized,
            ((0,),),
            axis=LinearRemovalAxis.OUTPUT_ROWS,
            max_workspace_bytes=1024,
        )

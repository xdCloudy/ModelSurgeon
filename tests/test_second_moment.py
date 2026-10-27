from __future__ import annotations

import hashlib

import numpy as np
import pytest

from modelsurgeon.features.kfac_sensitivity import (
    LinearRemovalAxis,
    score_structured_linear_removals,
)
from modelsurgeon.features.second_moment import (
    RawSecondMomentCollector,
    RawSecondMomentConfig,
    RawSecondMomentError,
    RawSecondMomentKind,
)


def test_collects_raw_uncentered_second_moment_with_padding_mask() -> None:
    collector = RawSecondMomentCollector(
        RawSecondMomentConfig(
            kind=RawSecondMomentKind.INPUT_ACTIVATION,
            dimension=2,
            max_batch_rows=1,
        )
    )
    collector.update(((1.0, 2.0), (3.0, 4.0), (100.0, 100.0)), (True, True, False))

    summary = collector.summary()

    np.testing.assert_allclose(summary.matrix, [[5.0, 7.0], [7.0, 10.0]])
    assert summary.observation_count == 2
    assert summary.kind is RawSecondMomentKind.INPUT_ACTIVATION
    assert summary.matrix.flags.writeable is False
    assert summary.to_record()["matrix_sha256"] == hashlib.sha256(
        memoryview(summary.matrix).cast("B")
    ).hexdigest()


def test_collects_output_gradient_second_moment_and_respects_chunking() -> None:
    collector = RawSecondMomentCollector(
        RawSecondMomentConfig(
            kind=RawSecondMomentKind.OUTPUT_GRADIENT,
            dimension=2,
            max_batch_rows=2,
        )
    )
    collector.update(((1.0, 0.0), (0.0, 2.0), (2.0, 2.0)))

    summary = collector.summary()

    np.testing.assert_allclose(summary.matrix, [[5.0 / 3.0, 4.0 / 3.0], [4.0 / 3.0, 8.0 / 3.0]])
    assert summary.observation_count == 3
    assert summary.kind is RawSecondMomentKind.OUTPUT_GRADIENT
    assert summary.planned_peak_workspace_bytes <= summary.workspace_budget_bytes


def test_invalid_later_chunk_does_not_partially_commit_update() -> None:
    collector = RawSecondMomentCollector(
        RawSecondMomentConfig(
            kind=RawSecondMomentKind.INPUT_ACTIVATION,
            dimension=2,
            max_batch_rows=1,
        )
    )
    collector.update(((1.0, 0.0),))

    with pytest.raises(RawSecondMomentError, match="must be finite"):
        collector.update(((2.0, 0.0), (float("nan"), 1.0)))

    summary = collector.summary()
    assert summary.observation_count == 1
    np.testing.assert_allclose(summary.matrix, [[1.0, 0.0], [0.0, 0.0]])


def test_rejects_shape_mask_observation_and_workspace_violations() -> None:
    collector = RawSecondMomentCollector(
        RawSecondMomentConfig(
            kind=RawSecondMomentKind.INPUT_ACTIVATION,
            dimension=2,
            max_observations=2,
            max_batch_rows=2,
        )
    )
    with pytest.raises(RawSecondMomentError, match="shape"):
        collector.update(((1.0,),))
    with pytest.raises(RawSecondMomentError, match="align"):
        collector.update(((1.0, 2.0),), ())
    with pytest.raises(RawSecondMomentError, match="boolean"):
        collector.update(((1.0, 2.0),), (1,))  # type: ignore[arg-type]
    with pytest.raises(RawSecondMomentError, match="max_observations"):
        collector.update(((1.0, 2.0),) * 3)
    with pytest.raises(RawSecondMomentError, match="exceeds budget"):
        RawSecondMomentConfig(
            kind=RawSecondMomentKind.INPUT_ACTIVATION,
            dimension=1000,
            max_workspace_bytes=1024,
        )


def test_summary_seals_collector_and_empty_calibration_fails_closed() -> None:
    empty = RawSecondMomentCollector(
        RawSecondMomentConfig(kind=RawSecondMomentKind.OUTPUT_GRADIENT, dimension=2)
    )
    empty.update(((1.0, 2.0),), (False,))
    with pytest.raises(RawSecondMomentError, match="at least one"):
        empty.summary()

    collector = RawSecondMomentCollector(
        RawSecondMomentConfig(kind=RawSecondMomentKind.INPUT_ACTIVATION, dimension=1)
    )
    collector.update(((2.0,),))
    collector.summary()
    with pytest.raises(RawSecondMomentError, match="sealed"):
        collector.update(((3.0,),))
    with pytest.raises(RawSecondMomentError, match="already finalized"):
        collector.summary()


def test_accepts_numpy_batches_and_ignores_masked_nonfinite_rows() -> None:
    collector = RawSecondMomentCollector(
        RawSecondMomentConfig(
            kind=RawSecondMomentKind.INPUT_ACTIVATION,
            dimension=2,
            max_batch_rows=2,
        )
    )
    collector.update(
        np.asarray([[2.0, 1.0], [float("nan"), float("inf")]]),
        np.asarray([True, False], dtype=np.bool_),
    )

    summary = collector.summary()

    assert summary.observation_count == 1
    np.testing.assert_allclose(summary.matrix, [[4.0, 2.0], [2.0, 1.0]])



def test_collected_factors_feed_structured_kfac_cost_without_copying() -> None:
    activations = RawSecondMomentCollector(
        RawSecondMomentConfig(kind=RawSecondMomentKind.INPUT_ACTIVATION, dimension=2)
    )
    gradients = RawSecondMomentCollector(
        RawSecondMomentConfig(kind=RawSecondMomentKind.OUTPUT_GRADIENT, dimension=2)
    )
    activations.update(((1.0, 0.0), (0.0, 1.0)))
    gradients.update(((1.0, 0.0), (0.0, 1.0)))

    report = score_structured_linear_removals(
        [[1.0, 0.0], [0.0, 2.0]],
        activations.summary().matrix,
        gradients.summary().matrix,
        ((0,), (1,)),
        axis=LinearRemovalAxis.OUTPUT_ROWS,
    )

    assert tuple(item.estimated_loss_increase for item in report.candidate_costs) == (
        0.125,
        0.5,
    )


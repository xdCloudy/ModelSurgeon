"""Tests for fixed-memory diagonal and Nyström activation covariance estimates."""

from __future__ import annotations

import numpy as np
import pytest

from modelsurgeon.features.activation_covariance import (
    ActivationCovarianceCollector,
    ActivationCovarianceConfig,
    ActivationCovarianceError,
    evaluate_covariance_accuracy,
)
from modelsurgeon.graph import ComponentId


def _channel_ids(count: int) -> tuple[ComponentId, ...]:
    return tuple(
        ComponentId.parse(f"model.layers.0.mlp.up_proj.channel.{index}")
        for index in range(count)
    )


def _low_rank_samples() -> tuple[tuple[float, ...], ...]:
    latent = (
        (-2.0, 1.0),
        (-1.0, -2.0),
        (0.0, 0.5),
        (1.0, 2.0),
        (2.0, -1.0),
        (3.0, 1.5),
        (-3.0, -0.5),
        (0.5, -1.5),
    )
    return tuple(
        (first, second, first + 2.0 * second, 3.0 * first - second)
        for first, second in latent
    )


def test_rank_two_sketch_matches_exact_rank_two_covariance() -> None:
    samples = _low_rank_samples()
    channels = _channel_ids(4)
    config = ActivationCovarianceConfig(
        component_id=ComponentId.parse("model.layers.0.mlp.up_proj.weight"),
        channel_ids=channels,
        sketch_rank=2,
        seed=17,
        max_workspace_bytes=1_000_000,
        covariance_ddof=1,
        eigenvalue_tolerance=1e-12,
    )
    collector = ActivationCovarianceCollector(config)
    collector.update(samples, (True,) * len(samples))

    summary = collector.summary()
    exact_array = np.cov(np.asarray(samples, dtype=np.float64), rowvar=False, ddof=1)
    exact = tuple(tuple(float(value) for value in row) for row in exact_array)
    accuracy = evaluate_covariance_accuracy(summary, exact)

    assert summary.observation_count == len(samples)
    assert summary.factor_rank == 2
    assert summary.diagonal == pytest.approx(np.diag(exact_array).tolist(), abs=1e-12)
    assert accuracy.diagonal_max_abs_error < 1e-12
    assert accuracy.sketch_relative_frobenius_error < 1e-9
    assert summary.planned_peak_workspace_bytes <= config.max_workspace_bytes
    assert collector.planned_peak_workspace_bytes == config.planned_peak_workspace_bytes

    records = {record.name: record for record in summary.feature_records()}
    assert set(records) == {
        "activation_covariance_diagonal",
        "activation_covariance_sketch_factor",
    }
    metadata = dict(records["activation_covariance_diagonal"].metadata)
    assert metadata["configured_sketch_rank"] == 2
    assert metadata["factor_rank"] == 2
    assert metadata["workspace_peak_bytes"] == config.planned_peak_workspace_bytes


def test_masked_samples_are_excluded_and_seed_is_deterministic() -> None:
    samples = _low_rank_samples()
    mask = (True, False, True, True, False, True, True, True)
    config = ActivationCovarianceConfig(
        component_id=ComponentId.parse("model.layers.0.self_attn.q_proj.weight"),
        channel_ids=_channel_ids(4),
        sketch_rank=3,
        seed=99,
        max_workspace_bytes=1_000_000,
    )

    first = ActivationCovarianceCollector(config)
    first.update(samples, mask)
    second = ActivationCovarianceCollector(config)
    second.update(samples, mask)
    first_summary = first.summary()
    second_summary = second.summary()

    included = np.asarray(
        [sample for sample, keep in zip(samples, mask, strict=True) if keep],
        dtype=np.float64,
    )
    expected = np.cov(included, rowvar=False, ddof=1)

    assert first_summary.to_record() == second_summary.to_record()
    assert first_summary.observation_count == int(np.sum(mask))
    assert first_summary.diagonal == pytest.approx(np.diag(expected).tolist(), abs=1e-12)


def test_workspace_budget_is_rejected_before_collector_allocation() -> None:
    with pytest.raises(ActivationCovarianceError, match="exceeds budget"):
        ActivationCovarianceConfig(
            component_id=ComponentId.parse("model.layers.0.mlp.down_proj.weight"),
            channel_ids=_channel_ids(128),
            sketch_rank=32,
            max_workspace_bytes=1,
        )


def test_insufficient_or_non_finite_observations_fail_explicitly() -> None:
    config = ActivationCovarianceConfig(
        component_id=ComponentId.parse("model.layers.0.mlp.gate_proj.weight"),
        channel_ids=_channel_ids(2),
        sketch_rank=2,
        max_workspace_bytes=1_000_000,
    )
    collector = ActivationCovarianceCollector(config)
    collector.update(((1.0, 2.0),), (True,))
    with pytest.raises(ActivationCovarianceError, match="more observations than ddof"):
        collector.summary()

    broken = ActivationCovarianceCollector(config)
    with pytest.raises(ActivationCovarianceError, match="must be finite"):
        broken.update(((1.0, float("nan")),), (True,))


def test_width_and_mask_alignment_are_validated() -> None:
    config = ActivationCovarianceConfig(
        component_id=ComponentId.parse("model.layers.0.self_attn.o_proj.weight"),
        channel_ids=_channel_ids(3),
        sketch_rank=2,
        max_workspace_bytes=1_000_000,
    )
    collector = ActivationCovarianceCollector(config)

    with pytest.raises(ActivationCovarianceError, match="tokens and mask must align"):
        collector.update(((1.0, 2.0, 3.0),), ())
    with pytest.raises(ActivationCovarianceError, match="activation width"):
        collector.update(((1.0, 2.0),), (True,))

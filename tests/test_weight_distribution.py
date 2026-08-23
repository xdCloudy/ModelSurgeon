"""Tests for percentile, histogram, skewness, and kurtosis weight features."""

from __future__ import annotations

import math

import numpy as np
import pytest

from modelsurgeon.features.weight_distribution import (
    WeightDistributionConfig,
    WeightDistributionError,
    extract_weight_distribution,
)
from modelsurgeon.graph import ComponentId


class FakeTensor:
    def __init__(
        self,
        values: list[float],
        shape: tuple[int, ...],
        *,
        dtype: str = "torch.float16",
        device: str = "cpu",
    ) -> None:
        self.values = values
        self.shape = shape
        self.dtype = dtype
        self.device = device

    def numel(self) -> int:
        return len(self.values)

    def detach(self) -> FakeTensor:
        return FakeTensor(list(self.values), self.shape, dtype=self.dtype, device=self.device)

    def cpu(self) -> FakeTensor:
        return FakeTensor(list(self.values), self.shape, dtype=self.dtype, device="cpu")

    def double(self) -> FakeTensor:
        return FakeTensor(list(self.values), self.shape, dtype="torch.float64", device=self.device)

    def reshape(self, *shape: int) -> FakeTensor:
        assert shape == (-1,)
        return FakeTensor(
            list(self.values),
            (len(self.values),),
            dtype=self.dtype,
            device=self.device,
        )

    def tolist(self) -> object:
        return list(self.values)


def test_distribution_matches_numpy_reference() -> None:
    values = [-2.0, -1.0, 0.0, 1.0, 4.0]
    array = np.asarray(values, dtype=np.float64)
    config = WeightDistributionConfig(quantiles=(0.25, 0.5, 0.75), histogram_bins=4)

    result = extract_weight_distribution(
        ComponentId.parse("model.layers.0.mlp.up_proj.weight"),
        FakeTensor(values, (1, 5), dtype="torch.float16", device="cuda:0"),
        config,
    )

    expected_quantiles = np.quantile(array, config.quantiles, method="linear")
    assert [value for _, value in result.percentiles] == pytest.approx(expected_quantiles.tolist())

    expected_counts, expected_edges = np.histogram(
        array, bins=config.histogram_bins, range=(float(array.min()), float(array.max()))
    )
    assert result.histogram_counts == tuple(int(value) for value in expected_counts)
    assert result.histogram_edges == pytest.approx(expected_edges.tolist())

    centered = array - array.mean()
    variance = float(np.mean(centered**2))
    expected_skewness = float(np.mean(centered**3) / variance**1.5)
    expected_kurtosis = float(np.mean(centered**4) / variance**2 - 3.0)
    assert result.skewness == pytest.approx(expected_skewness)
    assert result.excess_kurtosis == pytest.approx(expected_kurtosis)
    assert result.percentile_interpolation == "linear"
    assert result.histogram_method == "equal_width"
    assert result.storage_dtype == "float16"
    assert result.source_device == "cuda:0"

    records = {record.name: record for record in result.feature_records()}
    assert set(records) == {
        "weight_percentiles",
        "weight_histogram_edges",
        "weight_histogram_counts",
        "weight_skewness",
        "weight_excess_kurtosis",
    }
    assert dict(records["weight_percentiles"].metadata)["percentile_interpolation"] == "linear"
    assert dict(records["weight_percentiles"].metadata)["histogram_bins"] == 4


def test_constant_low_precision_tensor_never_emits_nan() -> None:
    result = extract_weight_distribution(
        ComponentId.parse("model.layers.0.self_attn.q_proj.weight"),
        FakeTensor([7.0] * 8, (2, 4), dtype="torch.bfloat16"),
        WeightDistributionConfig(quantiles=(0.0, 0.5, 1.0), histogram_bins=8),
    )

    assert result.percentiles == ((0.0, 7.0), (0.5, 7.0), (1.0, 7.0))
    assert result.skewness == 0.0
    assert result.excess_kurtosis == 0.0
    assert sum(result.histogram_counts) == 8
    assert len(result.histogram_counts) == 8
    assert len(result.histogram_edges) == 9
    assert all(math.isfinite(value) for value in result.histogram_edges)
    assert all(math.isfinite(value) for _, value in result.percentiles)


def test_invalid_distribution_configuration_is_rejected() -> None:
    with pytest.raises(WeightDistributionError, match="strictly increasing"):
        WeightDistributionConfig(quantiles=(0.5, 0.25))
    with pytest.raises(WeightDistributionError, match=r"within \[0, 1\]"):
        WeightDistributionConfig(quantiles=(-0.1,))
    with pytest.raises(WeightDistributionError, match="bin count"):
        WeightDistributionConfig(histogram_bins=1)

"""Tests for deterministic bounded exact singular-value features."""

from __future__ import annotations

import math

import numpy as np
import pytest

from modelsurgeon.features.spectral_exact import (
    ExactSpectralConfig,
    ExactSpectralError,
    extract_exact_spectral_features,
)
from modelsurgeon.graph import ComponentId


class FakeTensor:
    def __init__(
        self,
        values: list[float],
        shape: tuple[int, ...],
        *,
        dtype: str = "torch.float32",
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


def test_exact_spectral_features_match_numpy_svd() -> None:
    matrix = np.asarray(
        [[3.0, 1.0, 1.0], [-1.0, 3.0, 1.0], [0.0, 2.0, 4.0], [1.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    result = extract_exact_spectral_features(
        ComponentId.parse("model.layers.0.mlp.up_proj.weight"),
        FakeTensor(matrix.reshape(-1).tolist(), matrix.shape, device="cuda:0"),
        ExactSpectralConfig(
            max_elements=64,
            max_minor_dimension=8,
            convergence_tolerance=1e-13,
            max_sweeps=100,
            energy_thresholds=(0.5, 0.9, 0.99),
        ),
    )

    assert result.accepted is True
    assert result.features is not None
    features = result.features
    expected = np.linalg.svd(matrix, compute_uv=False)
    assert features.singular_values == pytest.approx(expected.tolist(), rel=1e-10, abs=1e-10)
    assert features.spectral_norm == pytest.approx(float(expected[0]), rel=1e-10)
    assert features.singular_value_decay == pytest.approx((expected / expected[0]).tolist())
    expected_stable = float(np.sum(expected**2) / expected[0] ** 2)
    assert features.stable_rank == pytest.approx(expected_stable)

    probabilities = expected / expected.sum()
    expected_effective = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    assert features.effective_rank == pytest.approx(expected_effective)

    energy = expected**2
    cumulative = np.cumsum(energy) / np.sum(energy)
    expected_ranks = tuple(
        (threshold, int(np.searchsorted(cumulative, threshold, side="left")) + 1)
        for threshold in (0.5, 0.9, 0.99)
    )
    assert features.energy_ranks == expected_ranks
    assert features.compute_dtype == "float64"
    assert features.convergence_tolerance == 1e-13
    assert features.source_device == "cuda:0"
    assert features.sweeps >= 0

    records = {record.name: record for record in features.feature_records()}
    assert set(records) == {
        "singular_values",
        "singular_value_decay",
        "spectral_norm",
        "effective_rank",
        "stable_rank",
        "energy_ranks",
    }
    metadata = dict(records["singular_values"].metadata)
    assert metadata["compute_dtype"] == "float64"
    assert metadata["convergence_tolerance"] == 1e-13


def test_zero_matrix_has_defined_rank_features() -> None:
    result = extract_exact_spectral_features(
        ComponentId.parse("model.layers.0.self_attn.q_proj.weight"),
        FakeTensor([0.0] * 12, (3, 4), dtype="torch.float16"),
    )

    assert result.features is not None
    assert result.features.singular_values == (0.0, 0.0, 0.0)
    assert result.features.singular_value_decay == (0.0, 0.0, 0.0)
    assert result.features.spectral_norm == 0.0
    assert result.features.effective_rank == 0.0
    assert result.features.stable_rank == 0.0
    assert result.features.energy_ranks == ((0.9, 0), (0.95, 0), (0.99, 0))
    assert all(
        math.isfinite(value)
        for value in (
            *result.features.singular_values,
            result.features.effective_rank,
            result.features.stable_rank,
        )
    )


def test_oversized_and_non_matrix_inputs_decline_with_reason() -> None:
    oversized = extract_exact_spectral_features(
        ComponentId.parse("model.layers.0.mlp.down_proj.weight"),
        FakeTensor([1.0] * 16, (4, 4)),
        ExactSpectralConfig(max_elements=8),
    )
    assert oversized.accepted is False
    assert oversized.features is None
    assert oversized.decline_reason is not None
    assert "exceeding limit 8" in oversized.decline_reason

    vector = extract_exact_spectral_features(
        ComponentId.parse("model.embed_tokens.weight"),
        FakeTensor([1.0, 2.0, 3.0], (3,)),
    )
    assert vector.accepted is False
    assert vector.decline_reason == "expected a matrix, received shape (3,)"


def test_invalid_exact_spectral_configuration_is_rejected() -> None:
    with pytest.raises(ExactSpectralError, match="float64"):
        ExactSpectralConfig(compute_dtype="float32")
    with pytest.raises(ExactSpectralError, match="strictly increasing"):
        ExactSpectralConfig(energy_thresholds=(0.99, 0.9))

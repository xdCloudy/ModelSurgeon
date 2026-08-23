"""Tests for seeded workspace-bounded randomized spectral extraction."""

from __future__ import annotations

import numpy as np
import pytest

from modelsurgeon.features.spectral_randomized import (
    RandomizedSpectralConfig,
    RandomizedSpectralError,
    extract_randomized_spectral_features,
    plan_randomized_spectral_workspace,
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
        calls: list[str] | None = None,
    ) -> None:
        self.values = values
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.calls = calls if calls is not None else []

    def numel(self) -> int:
        return len(self.values)

    def detach(self) -> FakeTensor:
        self.calls.append("detach")
        return FakeTensor(
            list(self.values), self.shape, dtype=self.dtype, device=self.device, calls=self.calls
        )

    def cpu(self) -> FakeTensor:
        self.calls.append("cpu")
        return FakeTensor(
            list(self.values), self.shape, dtype=self.dtype, device="cpu", calls=self.calls
        )

    def double(self) -> FakeTensor:
        self.calls.append("double")
        return FakeTensor(
            list(self.values), self.shape, dtype="torch.float64", device=self.device, calls=self.calls
        )

    def reshape(self, *shape: int) -> FakeTensor:
        self.calls.append("reshape")
        assert shape == (-1,)
        return FakeTensor(
            list(self.values),
            (len(self.values),),
            dtype=self.dtype,
            device=self.device,
            calls=self.calls,
        )

    def tolist(self) -> object:
        self.calls.append("tolist")
        return list(self.values)


def test_randomized_svd_is_seeded_and_close_to_exact_small_case() -> None:
    rng = np.random.Generator(np.random.PCG64(41))
    left = rng.standard_normal((40, 5))
    right = rng.standard_normal((5, 24))
    matrix = left @ right + 0.01 * rng.standard_normal((40, 24))
    tensor = FakeTensor(matrix.reshape(-1).tolist(), matrix.shape, device="cuda:0")
    config = RandomizedSpectralConfig(
        target_rank=5,
        oversampling=4,
        power_iterations=1,
        seed=1234,
        max_workspace_bytes=10_000_000,
        reconstruction_ranks=(1, 3, 5),
    )

    first = extract_randomized_spectral_features(
        ComponentId.parse("model.layers.0.mlp.up_proj.weight"), tensor, config
    )
    second = extract_randomized_spectral_features(
        ComponentId.parse("model.layers.0.mlp.up_proj.weight"),
        FakeTensor(matrix.reshape(-1).tolist(), matrix.shape, device="cuda:0"),
        config,
    )

    assert first.features is not None
    assert second.features is not None
    assert first.features.singular_values == second.features.singular_values
    exact = np.linalg.svd(matrix, compute_uv=False)[:5]
    assert first.features.singular_values == pytest.approx(exact.tolist(), rel=0.03, abs=1e-8)
    assert first.features.workspace.estimated_peak_bytes <= config.max_workspace_bytes
    assert first.features.seed == 1234
    assert first.features.algorithm == "gaussian_range_finder_qr_svd_v1"
    assert first.features.source_device == "cuda:0"

    total_energy = float(np.sum(matrix * matrix))
    for rank, error in first.features.reconstruction_errors:
        expected = np.sqrt(max(0.0, total_energy - np.sum(exact[:rank] ** 2)) / total_energy)
        assert error == pytest.approx(float(expected), abs=0.02)


def test_workspace_preflight_declines_before_cpu_snapshot() -> None:
    tensor = FakeTensor([1.0] * 400, (20, 20), device="cuda:0")
    config = RandomizedSpectralConfig(
        target_rank=4,
        oversampling=2,
        power_iterations=0,
        max_workspace_bytes=1,
        reconstruction_ranks=(1, 4),
    )

    result = extract_randomized_spectral_features(
        ComponentId.parse("model.layers.0.self_attn.q_proj.weight"), tensor, config
    )

    assert result.accepted is False
    assert result.workspace is not None
    assert result.workspace.fits is False
    assert result.decline_reason is not None
    assert "exceeds budget" in result.decline_reason
    assert tensor.calls == ["detach"]


def test_workspace_planner_chooses_minimum_orientation_and_is_deterministic() -> None:
    config = RandomizedSpectralConfig(
        target_rank=8,
        oversampling=4,
        power_iterations=1,
        max_workspace_bytes=1 << 30,
        reconstruction_ranks=(1, 4, 8),
    )
    plan = plan_randomized_spectral_workspace((32, 2048), config)
    repeat = plan_randomized_spectral_workspace((32, 2048), config)

    assert plan == repeat
    assert plan.estimated_peak_bytes <= config.max_workspace_bytes
    assert plan.target_rank == 8
    assert plan.sketch_rank == 12
    assert sorted(plan.oriented_shape) == [32, 2048]


def test_zero_matrix_and_small_minor_dimension_have_finite_outputs() -> None:
    result = extract_randomized_spectral_features(
        ComponentId.parse("model.layers.0.mlp.down_proj.weight"),
        FakeTensor([0.0] * 12, (3, 4)),
        RandomizedSpectralConfig(
            target_rank=8,
            oversampling=2,
            power_iterations=0,
            max_workspace_bytes=1_000_000,
            reconstruction_ranks=(1, 2, 3, 4, 8),
        ),
    )

    assert result.features is not None
    assert result.features.singular_values == (0.0, 0.0, 0.0)
    assert result.features.reconstruction_errors == ((1, 0.0), (2, 0.0), (3, 0.0))


def test_invalid_randomized_configuration_is_rejected() -> None:
    with pytest.raises(RandomizedSpectralError, match="target rank"):
        RandomizedSpectralConfig(target_rank=0)
    with pytest.raises(RandomizedSpectralError, match="reconstruction ranks"):
        RandomizedSpectralConfig(target_rank=4, reconstruction_ranks=(1, 8))

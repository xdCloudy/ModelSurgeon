"""Tests for deterministic per-tensor weight statistics."""

from __future__ import annotations

import math

import pytest

from modelsurgeon.features.weight_statistics import (
    WeightStatisticsError,
    extract_weight_statistics,
)
from modelsurgeon.graph import ComponentId


class FakeTensor:
    def __init__(
        self,
        values: list[float],
        shape: tuple[int, ...],
        *,
        dtype: str = "torch.float16",
        device: str = "cuda:0",
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
            list(self.values),
            self.shape,
            dtype=self.dtype,
            device=self.device,
            calls=self.calls,
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
            list(self.values), (len(self.values),), dtype=self.dtype, device=self.device, calls=self.calls
        )

    def tolist(self) -> object:
        self.calls.append("tolist")
        return list(self.values)


def test_statistics_match_reference_values_and_release_cuda_source() -> None:
    tensor = FakeTensor([-2.0, 0.0, 1.0, 3.0], (2, 2))
    component_id = ComponentId.parse("model.layers.0.mlp.up_proj.weight")

    result = extract_weight_statistics(component_id, tensor)

    assert result.count == 4
    assert result.shape == (2, 2)
    assert result.storage_dtype == "float16"
    assert result.source_device == "cuda:0"
    assert result.minimum == -2.0
    assert result.maximum == 3.0
    assert result.mean == 0.5
    assert result.variance == pytest.approx(3.25)
    assert result.standard_deviation == pytest.approx(math.sqrt(3.25))
    assert result.l1_norm == 6.0
    assert result.l2_norm == pytest.approx(math.sqrt(14.0))
    assert result.frobenius_norm == result.l2_norm
    assert result.max_magnitude == 3.0
    assert result.sparsity == 0.25
    assert tensor.calls == ["detach", "cpu", "double", "reshape", "tolist"]

    records = result.feature_records()
    assert {record.name for record in records} >= {
        "weight_count",
        "weight_mean",
        "weight_variance",
        "weight_frobenius_norm",
        "weight_shape",
    }
    assert all(record.component_id == component_id for record in records)
    assert all(record.precision.compute_dtype == "float64" for record in records)
    assert result.to_record()["shape"] == [2, 2]


def test_constant_tensor_has_zero_dispersion_without_nan() -> None:
    result = extract_weight_statistics(
        ComponentId.parse("model.layers.0.self_attn.q_proj.weight"),
        FakeTensor([4.0, 4.0, 4.0, 4.0], (2, 2), dtype="torch.bfloat16", device="cpu"),
    )

    assert result.mean == 4.0
    assert result.variance == 0.0
    assert result.standard_deviation == 0.0
    assert result.sparsity == 0.0
    assert all(math.isfinite(float(value)) for key, value in result.to_record().items() if key in {
        "minimum",
        "maximum",
        "mean",
        "variance",
        "standard_deviation",
        "l1_norm",
        "l2_norm",
        "frobenius_norm",
        "max_magnitude",
        "sparsity",
    })


def test_empty_tensor_declines_explicitly() -> None:
    with pytest.raises(WeightStatisticsError, match="empty weight tensors"):
        extract_weight_statistics(
            ComponentId.parse("model.layers.0.mlp.down_proj.weight"),
            FakeTensor([], (0, 4), device="cpu"),
        )


def test_non_finite_tensor_declines_explicitly() -> None:
    with pytest.raises(WeightStatisticsError, match="non-finite"):
        extract_weight_statistics(
            ComponentId.parse("model.layers.0.mlp.gate_proj.weight"),
            FakeTensor([1.0, float("nan")], (1, 2), device="cpu"),
        )


def test_shape_numel_mismatch_is_rejected() -> None:
    with pytest.raises(WeightStatisticsError, match="shape does not match numel"):
        extract_weight_statistics(
            ComponentId.parse("model.layers.0.self_attn.o_proj.weight"),
            FakeTensor([1.0, 2.0], (2, 2), device="cpu"),
        )

"""Analytic tests for gradient norms and first-order sensitivity."""

from __future__ import annotations

import math

import pytest

from modelsurgeon.features.gradient_features import (
    GradientFeatureError,
    extract_gradient_features,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.instrumentation import GradientSnapshot


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
        return FakeTensor(
            list(self.values),
            self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def cpu(self) -> FakeTensor:
        return FakeTensor(
            list(self.values),
            self.shape,
            dtype=self.dtype,
            device="cpu",
        )

    def double(self) -> FakeTensor:
        return FakeTensor(
            list(self.values),
            self.shape,
            dtype="torch.float64",
            device=self.device,
        )

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


def _component() -> ComponentId:
    return ComponentId.parse("model.layers.0.mlp.up_proj.weight")


def test_gradient_features_match_analytic_tiny_reference() -> None:
    component_id = _component()
    weight = FakeTensor([2.0, -3.0], (2,), dtype="torch.float16", device="cuda:0")
    snapshot = GradientSnapshot(
        component_id,
        0,
        (0.5, -1.0),
        (2,),
        "float32",
        "cuda:0",
    )

    outcome = extract_gradient_features(component_id, weight, (snapshot,))

    assert outcome.features is not None
    features = outcome.features
    assert features.gradient_l1_norm == 1.5
    assert features.gradient_l2_norm == pytest.approx(math.sqrt(1.25))
    assert features.gradient_max_magnitude == 1.0
    assert features.weight_gradient_sum == 4.0
    assert features.weight_gradient_abs_sum == 4.0
    assert features.weight_gradient_l2_norm == pytest.approx(math.sqrt(10.0))
    assert features.first_order_removal_estimate == -4.0
    assert features.first_order_removal_magnitude == 4.0
    assert features.weight_storage_dtype == "float16"
    assert features.weight_source_device == "cuda:0"
    assert features.gradient_storage_dtypes == ("float32",)
    assert features.gradient_source_devices == ("cuda:0",)

    records = {record.name: record for record in features.feature_records()}
    assert records["first_order_removal_estimate"].value == -4.0
    assert dict(records["gradient_l2_norm"].metadata)["batch_count"] == 1
    assert dict(records["gradient_l2_norm"].metadata)["aggregation"] == "mean_over_batches"


def test_multiple_batches_are_mean_aggregated_not_summed() -> None:
    component_id = _component()
    weight = FakeTensor([1.0, 1.0], (2,))
    snapshots = (
        GradientSnapshot(component_id, 0, (1.0, 0.0), (2,), "float32", "cpu"),
        GradientSnapshot(component_id, 1, (0.0, 2.0), (2,), "float32", "cpu"),
    )

    outcome = extract_gradient_features(component_id, weight, snapshots)

    assert outcome.features is not None
    assert outcome.features.batch_count == 2
    assert outcome.features.gradient_l1_norm == 1.5
    assert outcome.features.gradient_l2_norm == 1.5
    assert outcome.features.weight_gradient_sum == 1.5
    assert outcome.features.first_order_removal_estimate == -1.5
    assert outcome.features.first_order_removal_magnitude == 1.5


def test_missing_gradient_is_explicit_and_not_zero_filled() -> None:
    outcome = extract_gradient_features(
        _component(),
        FakeTensor([1.0, 2.0], (2,)),
        (),
    )

    assert outcome.available is False
    assert outcome.features is None
    assert outcome.missing_reason == "no gradients observed for selected component"
    assert outcome.to_record()["features"] is None


def test_mismatched_component_and_shape_fail_closed() -> None:
    component_id = _component()
    other = ComponentId.parse("model.layers.0.mlp.down_proj.weight")
    weight = FakeTensor([1.0, 2.0], (2,))

    with pytest.raises(GradientFeatureError, match="identity"):
        extract_gradient_features(
            component_id,
            weight,
            (GradientSnapshot(other, 0, (1.0, 2.0), (2,), "float32", "cpu"),),
        )

    with pytest.raises(GradientFeatureError, match="shape"):
        extract_gradient_features(
            component_id,
            weight,
            (GradientSnapshot(component_id, 0, (1.0, 2.0), (1, 2), "float32", "cpu"),),
        )

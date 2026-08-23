"""Tests for bounded selected-parameter gradient collection."""

from __future__ import annotations

import pytest

from modelsurgeon.graph import ComponentId
from modelsurgeon.instrumentation.gradients import (
    GradientCollector,
    GradientCollectorConfig,
    GradientCollectorError,
    GradientSnapshot,
)


class FakeTensor:
    def __init__(
        self,
        values: list[float],
        shape: tuple[int, ...],
        *,
        dtype: str = "torch.float32",
        device: str = "cuda:0",
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


class FakeParameter:
    def __init__(self) -> None:
        self.grad: FakeTensor | None = None


class FakeModel:
    def __init__(self, parameters: tuple[FakeParameter, ...]) -> None:
        self.parameters = parameters
        self.zero_grad_calls = 0
        self.live_gradient_counts: list[int] = []

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        assert set_to_none is True
        self.live_gradient_counts.append(
            sum(parameter.grad is not None for parameter in self.parameters)
        )
        for parameter in self.parameters:
            parameter.grad = None
        self.zero_grad_calls += 1


class FakeLoss:
    def __init__(self, callback: object) -> None:
        self.callback = callback

    def backward(self) -> None:
        callback = self.callback
        assert callable(callback)
        callback()


def _ids() -> tuple[ComponentId, ComponentId]:
    return (
        ComponentId.parse("model.layers.0.mlp.up_proj.weight"),
        ComponentId.parse("model.layers.0.mlp.down_proj.weight"),
    )


def test_collector_snapshots_selected_gradients_and_clears_each_batch() -> None:
    first_id, second_id = _ids()
    first = FakeParameter()
    second = FakeParameter()
    model = FakeModel((first, second))
    captured: list[GradientSnapshot] = []
    collector = GradientCollector(
        model,
        {first_id: first, second_id: second},
        (first_id, second_id),
        GradientCollectorConfig(enabled=True, max_batches=2),
    )

    def step(batch: int) -> FakeLoss:
        def backward() -> None:
            first.grad = FakeTensor([float(batch), float(batch + 1)], (2,))
            second.grad = FakeTensor([float(batch * 2)], (1,))

        return FakeLoss(backward)

    report = collector.collect((1, 3, 5), step, on_gradient=captured.append)

    assert report.enabled is True
    assert report.batches_processed == 2
    assert report.observations == 4
    assert report.peak_snapshot_elements == 2
    assert tuple(item.observed_batches for item in report.targets) == (2, 2)
    assert tuple(item.missing_batches for item in report.targets) == (0, 0)
    assert model.zero_grad_calls == 4
    assert max(model.live_gradient_counts) == 2
    assert first.grad is None
    assert second.grad is None
    assert [snapshot.source_device for snapshot in captured] == ["cuda:0"] * 4
    assert [snapshot.storage_dtype for snapshot in captured] == ["float32"] * 4


def test_disabled_collection_does_not_iterate_or_run_backward() -> None:
    first_id, _ = _ids()
    parameter = FakeParameter()
    model = FakeModel((parameter,))
    collector = GradientCollector(
        model,
        {first_id: parameter},
        (first_id,),
        GradientCollectorConfig(enabled=False),
    )
    calls = 0

    def step(batch: int) -> FakeLoss:
        nonlocal calls
        calls += batch
        return FakeLoss(lambda: None)

    report = collector.collect((1, 2, 3), step)

    assert report.enabled is False
    assert report.batches_processed == 0
    assert calls == 0
    assert model.zero_grad_calls == 0


def test_missing_gradient_is_reported_explicitly() -> None:
    first_id, second_id = _ids()
    first = FakeParameter()
    second = FakeParameter()
    model = FakeModel((first, second))
    collector = GradientCollector(
        model,
        {first_id: first, second_id: second},
        (first_id, second_id),
        GradientCollectorConfig(enabled=True, max_batches=1),
    )

    def step(batch: int) -> FakeLoss:
        del batch

        def backward() -> None:
            first.grad = FakeTensor([1.0], (1,))

        return FakeLoss(backward)

    report = collector.collect((0,), step)

    by_id = {item.component_id: item for item in report.targets}
    assert by_id[first_id].observed_batches == 1
    assert by_id[second_id].missing_batches == 1
    assert second.grad is None


def test_oversized_gradient_fails_closed_and_still_cleans_model() -> None:
    first_id, _ = _ids()
    parameter = FakeParameter()
    model = FakeModel((parameter,))
    collector = GradientCollector(
        model,
        {first_id: parameter},
        (first_id,),
        GradientCollectorConfig(enabled=True, max_elements_per_gradient=2),
    )

    def step(batch: int) -> FakeLoss:
        del batch

        def backward() -> None:
            parameter.grad = FakeTensor([1.0, 2.0, 3.0], (3,))

        return FakeLoss(backward)

    with pytest.raises(GradientCollectorError, match="exceeding limit"):
        collector.collect((0,), step)

    assert parameter.grad is None
    assert model.zero_grad_calls == 2


def test_backward_exception_still_clears_existing_gradients() -> None:
    first_id, _ = _ids()
    parameter = FakeParameter()
    model = FakeModel((parameter,))
    collector = GradientCollector(
        model,
        {first_id: parameter},
        (first_id,),
        GradientCollectorConfig(enabled=True),
    )

    def step(batch: int) -> FakeLoss:
        del batch

        def backward() -> None:
            parameter.grad = FakeTensor([1.0], (1,))
            raise RuntimeError("boom")

        return FakeLoss(backward)

    with pytest.raises(RuntimeError, match="boom"):
        collector.collect((0,), step)

    assert parameter.grad is None
    assert model.zero_grad_calls == 2

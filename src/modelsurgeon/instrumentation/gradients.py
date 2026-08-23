"""Bounded selected-parameter gradient collection with strict cleanup."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar

from modelsurgeon.graph import ComponentId

GRADIENT_COLLECTOR_VERSION = "1"

BatchT = TypeVar("BatchT")


class GradientCollectorError(ValueError):
    """Raised when safe gradient collection cannot preserve its contract."""


class GradientTensor(Protocol):
    shape: object
    dtype: object
    device: object

    def numel(self) -> int: ...

    def detach(self) -> GradientTensor: ...

    def cpu(self) -> GradientTensor: ...

    def double(self) -> GradientTensor: ...

    def reshape(self, *shape: int) -> GradientTensor: ...

    def tolist(self) -> object: ...


class GradientParameter(Protocol):
    grad: GradientTensor | None


class GradientModel(Protocol):
    def zero_grad(self, *, set_to_none: bool = True) -> None: ...


class BackwardLoss(Protocol):
    def backward(self) -> None: ...


@dataclass(frozen=True, slots=True)
class GradientCollectorConfig:
    """Safety limits for calibration backward passes."""

    enabled: bool = False
    max_batches: int = 32
    max_elements_per_gradient: int = 4_000_000

    def __post_init__(self) -> None:
        if self.max_batches <= 0:
            raise GradientCollectorError("gradient max_batches must be positive")
        if self.max_elements_per_gradient <= 0:
            raise GradientCollectorError(
                "gradient max_elements_per_gradient must be positive"
            )


@dataclass(frozen=True, slots=True)
class GradientSnapshot:
    """Detached CPU float64 snapshot for one selected parameter and batch."""

    component_id: ComponentId
    batch_index: int
    values: tuple[float, ...]
    shape: tuple[int, ...]
    storage_dtype: str
    source_device: str

    def __post_init__(self) -> None:
        if self.batch_index < 0 or not self.values:
            raise GradientCollectorError("gradient snapshots require observations")
        if math.prod(self.shape) != len(self.values):
            raise GradientCollectorError("gradient snapshot shape does not match values")
        if any(not math.isfinite(value) for value in self.values):
            raise GradientCollectorError("gradient snapshots must be finite")


@dataclass(frozen=True, slots=True)
class GradientTargetReport:
    component_id: ComponentId
    observed_batches: int
    missing_batches: int


@dataclass(frozen=True, slots=True)
class GradientCollectionReport:
    enabled: bool
    batches_processed: int
    observations: int
    peak_snapshot_elements: int
    targets: tuple[GradientTargetReport, ...]
    collector_version: str = GRADIENT_COLLECTOR_VERSION


class GradientCollector:
    """Run selected backward passes without retaining parameter gradients."""

    def __init__(
        self,
        model: GradientModel,
        parameters: Mapping[ComponentId, GradientParameter],
        targets: Iterable[ComponentId],
        config: GradientCollectorConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or GradientCollectorConfig()
        ordered = tuple(targets)
        if not ordered:
            raise GradientCollectorError("at least one gradient target is required")
        if len(ordered) != len(set(ordered)):
            raise GradientCollectorError("gradient targets must be unique")
        missing = tuple(target for target in ordered if target not in parameters)
        if missing:
            names = ", ".join(str(target) for target in missing)
            raise GradientCollectorError(f"unknown gradient targets: {names}")
        self.parameters = parameters
        self.targets = tuple(sorted(ordered))

    @staticmethod
    def _shape(value: object) -> tuple[int, ...]:
        if not isinstance(value, Iterable):
            raise GradientCollectorError("gradient shape is not iterable")
        try:
            shape = tuple(int(item) for item in value)
        except (TypeError, ValueError, OverflowError) as error:
            raise GradientCollectorError("gradient shape is invalid") from error
        if any(dimension < 0 for dimension in shape):
            raise GradientCollectorError("gradient shape contains a negative dimension")
        return shape

    def _snapshot(
        self,
        component_id: ComponentId,
        batch_index: int,
        gradient: GradientTensor,
    ) -> GradientSnapshot:
        try:
            detached = gradient.detach()
            count = int(detached.numel())
            shape = self._shape(detached.shape)
            storage_dtype = str(detached.dtype).removeprefix("torch.")
            source_device = str(detached.device)
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise GradientCollectorError("gradient tensor surface is invalid") from error
        if count <= 0 or math.prod(shape) != count:
            raise GradientCollectorError("gradient tensor shape does not match numel")
        if count > self.config.max_elements_per_gradient:
            raise GradientCollectorError(
                f"gradient for {component_id} has {count} elements, exceeding limit "
                f"{self.config.max_elements_per_gradient}"
            )
        try:
            raw = detached.cpu().double().reshape(-1).tolist()
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise GradientCollectorError("gradient could not be copied to CPU") from error
        if not isinstance(raw, list):
            raise GradientCollectorError("flattened gradient did not produce a list")
        try:
            values = tuple(float(item) for item in raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise GradientCollectorError("gradient contains non-numeric values") from error
        if len(values) != count:
            raise GradientCollectorError("gradient snapshot element count changed")
        return GradientSnapshot(
            component_id,
            batch_index,
            values,
            shape,
            storage_dtype,
            source_device,
        )

    def collect(
        self,
        batches: Iterable[BatchT],
        step: Callable[[BatchT], BackwardLoss],
        *,
        on_gradient: Callable[[GradientSnapshot], None] | None = None,
    ) -> GradientCollectionReport:
        """Collect selected gradients and clear all model gradients every batch."""

        observed = {target: 0 for target in self.targets}
        missing = {target: 0 for target in self.targets}
        if not self.config.enabled:
            return GradientCollectionReport(
                False,
                0,
                0,
                0,
                tuple(GradientTargetReport(target, 0, 0) for target in self.targets),
            )

        batches_processed = 0
        observations = 0
        peak_snapshot_elements = 0
        callback = on_gradient or (lambda snapshot: None)
        for batch_index, batch in enumerate(batches):
            if batch_index >= self.config.max_batches:
                break
            self.model.zero_grad(set_to_none=True)
            loss: BackwardLoss | None = None
            try:
                loss = step(batch)
                loss.backward()
                for target in self.targets:
                    gradient = self.parameters[target].grad
                    if gradient is None:
                        missing[target] += 1
                        continue
                    snapshot = self._snapshot(target, batch_index, gradient)
                    callback(snapshot)
                    observed[target] += 1
                    observations += 1
                    peak_snapshot_elements = max(
                        peak_snapshot_elements,
                        len(snapshot.values),
                    )
                batches_processed += 1
            finally:
                loss = None
                self.model.zero_grad(set_to_none=True)

        target_reports = tuple(
            GradientTargetReport(target, observed[target], missing[target])
            for target in self.targets
        )
        return GradientCollectionReport(
            True,
            batches_processed,
            observations,
            peak_snapshot_elements,
            target_reports,
        )

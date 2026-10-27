"""Streaming, bounded collection of raw KFAC second-moment factors."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import import_module
from typing import Any

SECOND_MOMENT_COLLECTOR_VERSION = "1"
_BLAS_WORKSPACE_RESERVE_BYTES = 16 * 1024 * 1024


class RawSecondMomentError(ValueError):
    """Raised when second-moment inputs or their resource budget are invalid."""


class RawSecondMomentKind(StrEnum):
    INPUT_ACTIVATION = "input_activation"
    OUTPUT_GRADIENT = "output_gradient"


@dataclass(frozen=True, slots=True)
class RawSecondMomentConfig:
    kind: RawSecondMomentKind
    dimension: int
    max_observations: int = 8192
    max_batch_rows: int = 32
    max_workspace_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in (
            ("dimension", self.dimension),
            ("max_observations", self.max_observations),
            ("max_batch_rows", self.max_batch_rows),
            ("max_workspace_bytes", self.max_workspace_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RawSecondMomentError(f"{name} must be a positive integer")
        if self.max_batch_rows > self.max_observations:
            raise RawSecondMomentError("max_batch_rows cannot exceed max_observations")
        if not isinstance(self.kind, RawSecondMomentKind):
            raise RawSecondMomentError("kind must identify activation or output-gradient rows")
        if self.planned_peak_workspace_bytes > self.max_workspace_bytes:
            raise RawSecondMomentError(
                f"planned second-moment workspace {self.planned_peak_workspace_bytes} exceeds "
                f"budget {self.max_workspace_bytes} bytes"
            )

    @property
    def planned_peak_workspace_bytes(self) -> int:
        dimension = self.dimension
        rows = self.max_batch_rows
        # Accumulator, reusable matmul scratch, per-update transactional sum,
        # two bounded row batches, mask bytes, and a BLAS implementation reserve.
        return (
            3 * dimension * dimension * 8
            + 2 * rows * dimension * 8
            + rows
            + self.max_observations
            + _BLAS_WORKSPACE_RESERVE_BYTES
        )


@dataclass(frozen=True, slots=True)
class RawSecondMomentSummary:
    kind: RawSecondMomentKind
    observation_count: int
    dimension: int
    matrix_sha256: str
    planned_peak_workspace_bytes: int
    workspace_budget_bytes: int
    _matrix: Any = field(repr=False, compare=False)

    @property
    def matrix(self) -> Any:
        """Read-only float64 NumPy view of ``E[x x.T]`` or ``E[g g.T]``."""

        view = self._matrix.view()
        view.flags.writeable = False
        return view

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "kind": self.kind.value,
            "observation_count": self.observation_count,
            "shape": [self.dimension, self.dimension],
            "dtype": "float64",
            "matrix_sha256": self.matrix_sha256,
            "planned_peak_workspace_bytes": self.planned_peak_workspace_bytes,
            "workspace_budget_bytes": self.workspace_budget_bytes,
        }


class RawSecondMomentCollector:
    """Stream raw outer products with a fixed dimension, batch, and workspace cap.

    Updates are transactional across a whole call: malformed or overflowing rows
    do not partially change the accumulated moment. The summary transfers the
    accumulator without copying and seals the collector. Only NumPy-owned arrays
    and a conservative BLAS reserve are included in the workspace estimate; the
    caller-owned input sequences are bounded by ``max_observations``.
    """

    def __init__(self, config: RawSecondMomentConfig) -> None:
        self.config = config
        self._np = self._numpy()
        shape = (config.dimension, config.dimension)
        try:
            self._sum = self._np.zeros(shape, dtype=self._np.float64)
            self._scratch = self._np.empty(shape, dtype=self._np.float64)
        except MemoryError as error:
            raise RawSecondMomentError(
                "second-moment arrays could not be allocated within the configured budget"
            ) from error
        self.observation_count = 0
        self._sealed = False

    @staticmethod
    def _numpy() -> Any:
        try:
            return import_module("numpy")
        except ModuleNotFoundError as error:
            raise RawSecondMomentError(
                "raw second moments require NumPy; install ModelSurgeon's runtime dependencies"
            ) from error

    def update(
        self,
        rows: object,
        valid_mask: object | None = None,
    ) -> None:
        """Accumulate valid vectors as raw, uncentered outer-product moments."""

        if self._sealed:
            raise RawSecondMomentError("a summarized second-moment collector is sealed")
        shape = getattr(rows, "shape", None)
        if shape is not None:
            try:
                row_shape = tuple(int(value) for value in shape)
            except (TypeError, ValueError, OverflowError) as error:
                raise RawSecondMomentError("second-moment row shape is invalid") from error
            if len(row_shape) != 2 or row_shape[1] != self.config.dimension:
                raise RawSecondMomentError(
                    f"row batches must have shape [tokens, {self.config.dimension}]"
                )
            row_count = row_shape[0]
        else:
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                raise RawSecondMomentError("second-moment rows must be a bounded sequence")
            row_count = len(rows)
        rows_data: Any = rows
        if row_count == 0:
            return
        if row_count > self.config.max_observations:
            raise RawSecondMomentError("batch rows exceed max_observations")
        if valid_mask is not None:
            try:
                mask_array = self._np.asarray(valid_mask)
            except (MemoryError, TypeError, ValueError, OverflowError) as error:
                raise RawSecondMomentError("valid mask could not be read") from error
            if mask_array.shape != (row_count,) or mask_array.dtype != self._np.bool_:
                raise RawSecondMomentError(
                    "valid mask must be a boolean vector aligned with input rows"
                )
            added_count = int(self._np.count_nonzero(mask_array))
        else:
            mask_array = None
            added_count = row_count
        if self.observation_count + added_count > self.config.max_observations:
            raise RawSecondMomentError("observations exceed max_observations")
        try:
            pending = self._np.zeros_like(self._sum)
        except MemoryError as error:
            raise RawSecondMomentError(
                "transactional second-moment buffer allocation failed"
            ) from error
        dimension = self.config.dimension
        for start in range(0, row_count, self.config.max_batch_rows):
            stop = min(start + self.config.max_batch_rows, row_count)
            raw_rows = rows_data[start:stop]
            try:
                values = self._np.asarray(raw_rows, dtype=self._np.float64)
            except (MemoryError, TypeError, ValueError, OverflowError) as error:
                raise RawSecondMomentError("rows must form a numeric rank-two matrix") from error
            if values.ndim != 2 or values.shape != (stop - start, dimension):
                raise RawSecondMomentError(
                    f"row batches must have shape [tokens, {dimension}]"
                )
            selected = (
                values if mask_array is None else values[mask_array[start:stop]]
            )
            if selected.shape[0] == 0:
                continue
            if any(not bool(self._np.isfinite(row).all()) for row in selected):
                raise RawSecondMomentError("valid second-moment rows must be finite")
            try:
                self._np.matmul(selected.T, selected, out=self._scratch)
                self._np.add(pending, self._scratch, out=pending)
            except (FloatingPointError, ValueError) as error:
                raise RawSecondMomentError("second-moment accumulation failed") from error
            if any(not bool(self._np.isfinite(row).all()) for row in self._scratch):
                raise RawSecondMomentError("second-moment products overflowed")
            if any(not bool(self._np.isfinite(row).all()) for row in pending):
                raise RawSecondMomentError("second-moment batch sum overflowed")

        # Commit atomically only after validating the complete input batch.
        self._np.add(self._sum, pending, out=self._scratch)
        if any(not bool(self._np.isfinite(row).all()) for row in self._scratch):
            raise RawSecondMomentError("second-moment total overflowed")
        self._sum, self._scratch = self._scratch, self._sum
        self.observation_count += added_count

    def summary(self) -> RawSecondMomentSummary:
        """Return the mean raw second moment and seal this collector."""

        if self._sealed:
            raise RawSecondMomentError("second-moment summary is already finalized")
        if self.observation_count <= 0:
            raise RawSecondMomentError("second moment requires at least one valid observation")
        self._np.divide(self._sum, float(self.observation_count), out=self._sum)
        for row in self._sum:
            if not bool(self._np.isfinite(row).all()):
                raise RawSecondMomentError("second-moment mean is non-finite")
        self._sum.flags.writeable = False
        matrix_hash = hashlib.sha256(memoryview(self._sum).cast("B")).hexdigest()
        self._sealed = True
        return RawSecondMomentSummary(
            kind=self.config.kind,
            observation_count=self.observation_count,
            dimension=self.config.dimension,
            matrix_sha256=matrix_hash,
            planned_peak_workspace_bytes=self.config.planned_peak_workspace_bytes,
            workspace_budget_bytes=self.config.max_workspace_bytes,
            _matrix=self._sum,
        )

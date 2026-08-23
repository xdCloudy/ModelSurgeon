"""Reversible graph-addressed component output masking without physical resizing."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol, Self

from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.contracts import MutationTransaction, require_safe_transaction


class ComponentMaskError(ValueError):
    """Raised when a component output cannot be masked without changing its contract."""


class MaskHookHandle(Protocol):
    def remove(self) -> None: ...


class MaskHookModule(Protocol):
    def register_forward_hook(self, hook: Callable[..., object]) -> MaskHookHandle: ...


@dataclass(frozen=True, slots=True)
class OutputSignature:
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ComponentMaskError("masked outputs require a positive rectangular shape")
        if not self.dtype:
            raise ComponentMaskError("masked outputs require a dtype identity")


class OutputMaskBackend(Protocol):
    def signature(self, output: object) -> OutputSignature: ...

    def mask_last_axis(
        self,
        output: object,
        indices: tuple[int, ...] | None,
    ) -> object: ...


def _sequence_shape(value: object) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ComponentMaskError("output contains an unsupported non-numeric leaf")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ComponentMaskError("output contains a non-finite numeric leaf")
        return ()
    if not value:
        raise ComponentMaskError("empty output sequences are unsupported")
    child_shapes = tuple(_sequence_shape(item) for item in value)
    if len(set(child_shapes)) != 1:
        raise ComponentMaskError("ragged output sequences are unsupported")
    return (len(value), *child_shapes[0])


def _sequence_dtype(value: object) -> str:
    leaves: list[str] = []

    def visit(item: object) -> None:
        if isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
            return
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ComponentMaskError("output contains an unsupported non-numeric leaf")
        leaves.append("float" if isinstance(item, float) else "int")

    visit(value)
    if not leaves:
        raise ComponentMaskError("output contains no numeric leaves")
    return "python.float" if "float" in leaves else "python.int"


def _typed_zero(value: object) -> int | float:
    if isinstance(value, float):
        return 0.0
    if isinstance(value, int) and not isinstance(value, bool):
        return 0
    raise ComponentMaskError("output contains an unsupported non-numeric leaf")


def _mask_sequence_last_axis(
    value: object,
    selected: frozenset[int],
    width: int,
) -> object:
    if not isinstance(value, (tuple, list)):
        raise ComponentMaskError("mask traversal reached a scalar before the final axis")
    if len(value) == width and all(not isinstance(item, (tuple, list)) for item in value):
        masked = [
            _typed_zero(item) if index in selected else item
            for index, item in enumerate(value)
        ]
        return tuple(masked) if isinstance(value, tuple) else masked
    masked_children = [
        _mask_sequence_last_axis(item, selected, width) for item in value
    ]
    return tuple(masked_children) if isinstance(value, tuple) else masked_children


class SequenceOutputMaskBackend:
    """Dependency-free rectangular numeric sequence backend used by tiny fixtures."""

    def signature(self, output: object) -> OutputSignature:
        shape = _sequence_shape(output)
        if not shape:
            raise ComponentMaskError("scalar component outputs cannot be channel-masked")
        return OutputSignature(shape, _sequence_dtype(output))

    def mask_last_axis(
        self,
        output: object,
        indices: tuple[int, ...] | None,
    ) -> object:
        signature = self.signature(output)
        width = signature.shape[-1]
        resolved = tuple(range(width)) if indices is None else indices
        if any(index < 0 or index >= width for index in resolved):
            raise ComponentMaskError(
                f"mask index is outside final-axis width {width}"
            )
        return _mask_sequence_last_axis(output, frozenset(resolved), width)


class ComponentOutputMask(AbstractContextManager["ComponentOutputMask"]):
    """Install one reversible output-transforming hook under transaction ownership."""

    def __init__(
        self,
        component_id: ComponentId,
        module: MaskHookModule,
        transaction: MutationTransaction,
        *,
        indices: tuple[int, ...] | None = None,
        backend: OutputMaskBackend | None = None,
    ) -> None:
        if indices is not None:
            if tuple(sorted(indices)) != indices or len(indices) != len(set(indices)):
                raise ComponentMaskError("mask indices must be unique and canonical")
            if any(index < 0 for index in indices):
                raise ComponentMaskError("mask indices cannot be negative")
        self.component_id = component_id
        self._module = module
        self._transaction = transaction
        self.indices = indices
        self._backend = backend or SequenceOutputMaskBackend()
        self._handle: MaskHookHandle | None = None

    @property
    def active(self) -> bool:
        return self._handle is not None

    def _hook(self, module: object, inputs: object, output: object) -> object:
        del module, inputs
        try:
            before = self._backend.signature(output)
            masked = self._backend.mask_last_axis(output, self.indices)
            after = self._backend.signature(masked)
        except ComponentMaskError as error:
            raise ComponentMaskError(f"{self.component_id}: {error}") from error
        if after != before:
            raise ComponentMaskError(
                f"{self.component_id}: masking changed output shape or dtype"
            )
        return masked

    def __enter__(self) -> Self:
        require_safe_transaction(self._transaction)
        if self._handle is not None:
            raise ComponentMaskError(f"{self.component_id}: mask hook already active")
        try:
            self._handle = self._module.register_forward_hook(self._hook)
        except Exception as error:
            raise ComponentMaskError(
                f"{self.component_id}: failed to register output mask hook"
            ) from error
        return self

    def close(self) -> None:
        if self._handle is None:
            return
        handle = self._handle
        self._handle = None
        try:
            handle.remove()
        except Exception as error:
            raise ComponentMaskError(
                f"{self.component_id}: failed to remove output mask hook"
            ) from error

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

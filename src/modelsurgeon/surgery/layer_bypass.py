"""Reversible transformer-block bypass at adapter-defined execution points."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol, Self

from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.component_mask import SequenceOutputMaskBackend
from modelsurgeon.surgery.contracts import MutationTransaction, require_safe_transaction


class LayerBypassError(ValueError):
    """Raised when a transformer residual layout cannot be bypassed safely."""


class LayerBypassHandle(Protocol):
    def remove(self) -> None: ...


class BypassableLayer(Protocol):
    def install_bypass(
        self,
        replacement: Callable[[tuple[object, ...], Mapping[str, object]], object],
    ) -> LayerBypassHandle: ...


class LayerBypassBackend(Protocol):
    def bypass(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> object: ...


class DirectResidualBypassBackend:
    """Bypass layers whose first positional input is also the complete output structure."""

    def __init__(self) -> None:
        self._signature = SequenceOutputMaskBackend()

    def bypass(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> object:
        del kwargs
        if not args:
            raise LayerBypassError("direct residual bypass requires a positional residual input")
        residual = args[0]
        try:
            before = self._signature.signature(residual)
        except ValueError as error:
            raise LayerBypassError("unsupported direct residual layout") from error
        output = residual
        after = self._signature.signature(output)
        if after != before:
            raise LayerBypassError("bypass changed residual output shape or dtype")
        return output


class TransformerLayerBypass(AbstractContextManager["TransformerLayerBypass"]):
    """Install a true block bypass while a prepared mutation transaction owns the model."""

    def __init__(
        self,
        component_id: ComponentId,
        layer: BypassableLayer,
        transaction: MutationTransaction,
        *,
        backend: LayerBypassBackend | None = None,
    ) -> None:
        self.component_id = component_id
        self._layer = layer
        self._transaction = transaction
        self._backend = backend or DirectResidualBypassBackend()
        self._handle: LayerBypassHandle | None = None

    @property
    def active(self) -> bool:
        return self._handle is not None

    def _replacement(
        self,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> object:
        try:
            return self._backend.bypass(args, kwargs)
        except LayerBypassError as error:
            raise LayerBypassError(f"{self.component_id}: {error}") from error

    def __enter__(self) -> Self:
        require_safe_transaction(self._transaction)
        if self._handle is not None:
            raise LayerBypassError(f"{self.component_id}: bypass already active")
        try:
            self._handle = self._layer.install_bypass(self._replacement)
        except Exception as error:
            raise LayerBypassError(
                f"{self.component_id}: failed to install layer bypass"
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
            raise LayerBypassError(
                f"{self.component_id}: failed to remove layer bypass"
            ) from error

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

"""Individual and grouped MLP intermediate-channel masking across coupled path points."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Self

from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.component_mask import (
    ComponentOutputMask,
    MaskHookModule,
    OutputMaskBackend,
)
from modelsurgeon.surgery.contracts import MutationTransaction, require_safe_transaction


class MLPChannelMaskError(ValueError):
    """Raised when a coupled MLP channel path is incomplete or ambiguous."""


class MLPVariant(StrEnum):
    GATED = "gated"
    UNGATED = "ungated"


class MLPPathRole(StrEnum):
    GATE = "gate"
    UP = "up"
    DOWN_INPUT = "down_input"


@dataclass(frozen=True, slots=True)
class MLPChannelMaskSpec:
    channels: tuple[int, ...]
    intermediate_width: int
    variant: MLPVariant = MLPVariant.GATED

    def __post_init__(self) -> None:
        if self.intermediate_width <= 0:
            raise MLPChannelMaskError("MLP intermediate width must be positive")
        if (
            not self.channels
            or self.channels != tuple(sorted(set(self.channels)))
            or self.channels[0] < 0
        ):
            raise MLPChannelMaskError(
                "MLP channels must be non-empty, unique, non-negative, and canonical"
            )
        if self.channels[-1] >= self.intermediate_width:
            raise MLPChannelMaskError("MLP channel index exceeds intermediate width")

    @property
    def required_roles(self) -> tuple[MLPPathRole, ...]:
        if self.variant is MLPVariant.GATED:
            return (MLPPathRole.GATE, MLPPathRole.UP, MLPPathRole.DOWN_INPUT)
        return (MLPPathRole.UP, MLPPathRole.DOWN_INPUT)


@dataclass(frozen=True, slots=True)
class MLPPathPoint:
    component_id: ComponentId
    module: MaskHookModule


class MLPChannelMask(AbstractContextManager["MLPChannelMask"]):
    """Apply one channel/group mask to every adapter-defined coupled MLP point."""

    def __init__(
        self,
        spec: MLPChannelMaskSpec,
        points: Mapping[MLPPathRole, MLPPathPoint],
        transaction: MutationTransaction,
        *,
        backend: OutputMaskBackend | None = None,
    ) -> None:
        missing = tuple(role for role in spec.required_roles if role not in points)
        if missing:
            raise MLPChannelMaskError(
                "MLP mask path is missing required points: "
                + ", ".join(role.value for role in missing)
            )
        extras = tuple(sorted(set(points) - set(spec.required_roles), key=lambda role: role.value))
        if extras:
            raise MLPChannelMaskError(
                "MLP mask path contains unsupported points for variant: "
                + ", ".join(role.value for role in extras)
            )
        self.spec = spec
        self._transaction = transaction
        self._masks = tuple(
            ComponentOutputMask(
                points[role].component_id,
                points[role].module,
                transaction,
                indices=spec.channels,
                backend=backend,
            )
            for role in spec.required_roles
        )
        self._stack: ExitStack | None = None

    @property
    def active(self) -> bool:
        return self._stack is not None and all(mask.active for mask in self._masks)

    def __enter__(self) -> Self:
        require_safe_transaction(self._transaction)
        if self._stack is not None:
            raise MLPChannelMaskError("MLP channel mask is already active")
        stack = ExitStack()
        try:
            for mask in self._masks:
                stack.enter_context(mask)
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self

    def close(self) -> None:
        if self._stack is None:
            return
        stack = self._stack
        self._stack = None
        stack.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

"""Logical attention-head masking at adapter-defined output points."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Self

from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.component_mask import (
    ComponentOutputMask,
    MaskHookModule,
    OutputMaskBackend,
    OutputSignature,
    SequenceOutputMaskBackend,
)
from modelsurgeon.surgery.contracts import MutationTransaction


class AttentionHeadMaskError(ValueError):
    """Raised when one logical head cannot be isolated safely at the requested point."""


class AttentionHeadMaskPoint(StrEnum):
    CONCATENATED_QUERY_OUTPUT = "concatenated_query_output"
    SHARED_KV_OUTPUT = "shared_kv_output"


@dataclass(frozen=True, slots=True)
class AttentionHeadMaskSpec:
    component_id: ComponentId
    head_index: int
    query_heads: int
    key_value_heads: int
    head_width: int
    point: AttentionHeadMaskPoint = AttentionHeadMaskPoint.CONCATENATED_QUERY_OUTPUT

    def __post_init__(self) -> None:
        if self.query_heads <= 0 or self.key_value_heads <= 0 or self.head_width <= 0:
            raise AttentionHeadMaskError("attention head geometry must be positive")
        if self.query_heads % self.key_value_heads != 0:
            raise AttentionHeadMaskError("query heads must be divisible by key/value heads")
        if self.head_index < 0 or self.head_index >= self.query_heads:
            raise AttentionHeadMaskError("attention head index is outside query-head geometry")
        if (
            self.point is AttentionHeadMaskPoint.SHARED_KV_OUTPUT
            and self.query_heads != self.key_value_heads
        ):
            mode = "MQA" if self.key_value_heads == 1 else "GQA"
            raise AttentionHeadMaskError(
                f"{mode} shared key/value output cannot isolate one logical query head"
            )

    @property
    def mask_indices(self) -> tuple[int, ...]:
        start = self.head_index * self.head_width
        return tuple(range(start, start + self.head_width))

    @property
    def expected_output_width(self) -> int:
        heads = (
            self.query_heads
            if self.point is AttentionHeadMaskPoint.CONCATENATED_QUERY_OUTPUT
            else self.key_value_heads
        )
        return heads * self.head_width


class _WidthCheckedBackend:
    def __init__(self, backend: OutputMaskBackend, expected_width: int) -> None:
        self._backend = backend
        self._expected_width = expected_width

    def signature(self, output: object) -> OutputSignature:
        signature = self._backend.signature(output)
        if signature.shape[-1] != self._expected_width:
            raise AttentionHeadMaskError(
                f"attention mask point width {signature.shape[-1]} does not match "
                f"adapter geometry {self._expected_width}"
            )
        return signature

    def mask_last_axis(
        self,
        output: object,
        indices: tuple[int, ...] | None,
    ) -> object:
        return self._backend.mask_last_axis(output, indices)


class AttentionHeadMask(AbstractContextManager["AttentionHeadMask"]):
    """Compose generic output masking into one logical attention-head experiment."""

    def __init__(
        self,
        spec: AttentionHeadMaskSpec,
        module: MaskHookModule,
        transaction: MutationTransaction,
        *,
        backend: OutputMaskBackend | None = None,
    ) -> None:
        self.spec = spec
        resolved_backend = _WidthCheckedBackend(
            backend or SequenceOutputMaskBackend(), spec.expected_output_width
        )
        self._mask = ComponentOutputMask(
            spec.component_id,
            module,
            transaction,
            indices=spec.mask_indices,
            backend=resolved_backend,
        )

    @property
    def active(self) -> bool:
        return self._mask.active

    def __enter__(self) -> Self:
        self._mask.__enter__()
        return self

    def close(self) -> None:
        self._mask.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._mask.__exit__(exc_type, exc_value, traceback)

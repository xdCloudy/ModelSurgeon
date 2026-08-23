"""Framework-neutral model adapter contracts and persisted records."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from modelsurgeon.graph import ComponentId

type Primitive = str | int | float | bool | None
type Record = dict[str, Primitive | list[Primitive] | dict[str, Primitive]]


class ModelFormat(StrEnum):
    """Model container formats understood at the adapter boundary."""

    HUGGING_FACE = "huggingface"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"


class AdapterCapability(StrEnum):
    """Independently advertised adapter operations."""

    LOAD = "load"
    DISCOVER = "discover"
    TENSOR_METADATA = "tensor_metadata"
    TENSOR_READ = "tensor_read"
    ACTIVATION_INSTRUMENTATION = "activation_instrumentation"
    MASK_MUTATION = "mask_mutation"
    PHYSICAL_MUTATION = "physical_mutation"
    CHECKPOINT_WRITE = "checkpoint_write"
    NATIVE_QUANTIZED_SURGERY = "native_quantized_surgery"


@dataclass(frozen=True, slots=True)
class AdapterIdentity:
    """Versioned adapter implementation identity safe for persistence."""

    name: str
    version: str

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("adapter name and version must be non-empty")

    def to_record(self) -> Record:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, slots=True)
class ModelSource:
    """Framework-neutral source identity supplied to an adapter."""

    format: ModelFormat
    locator: str
    revision: str | None = None
    content_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.locator:
            raise ValueError("model source locator must be non-empty")

    def to_record(self) -> Record:
        return {
            "format": self.format.value,
            "locator": self.locator,
            "revision": self.revision,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True, slots=True)
class OpenOptions:
    """Resource and trust controls shared by adapter open operations."""

    max_ram_bytes: int | None = None
    max_vram_bytes: int | None = None
    cpu_only: bool = False
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("max_ram_bytes", self.max_ram_bytes),
            ("max_vram_bytes", self.max_vram_bytes),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    """Persistable logical component metadata produced by discovery."""

    component_id: ComponentId
    kind: str
    parent: ComponentId | None = None
    attributes: tuple[tuple[str, Primitive], ...] = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("component kind must be non-empty")
        if self.parent is not None and self.component_id.parent != self.parent:
            raise ValueError("component parent must match the canonical ID parent")
        keys = [key for key, _ in self.attributes]
        if len(keys) != len(set(keys)):
            raise ValueError("component attribute keys must be unique")

    def to_record(self) -> Record:
        return {
            "component_id": str(self.component_id),
            "kind": self.kind,
            "parent": None if self.parent is None else str(self.parent),
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    """Persistable tensor metadata without a live framework tensor."""

    component_id: ComponentId
    tensor_name: str
    shape: tuple[int, ...]
    dtype: str
    storage_bytes: int
    quantization: str | None = None

    def __post_init__(self) -> None:
        if not self.tensor_name or not self.dtype:
            raise ValueError("tensor name and dtype must be non-empty")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("tensor dimensions must be non-negative")
        if self.storage_bytes < 0:
            raise ValueError("tensor storage bytes must be non-negative")

    def to_record(self) -> Record:
        return {
            "component_id": str(self.component_id),
            "tensor_name": self.tensor_name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "storage_bytes": self.storage_bytes,
            "quantization": self.quantization,
        }


@dataclass(frozen=True, slots=True)
class MutationSupport:
    """Adapter decision for one mutation kind and target component."""

    supported: bool
    reason: str
    constraints: tuple[tuple[str, Primitive], ...] = ()

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("mutation support decisions require a reason")
        keys = [key for key, _ in self.constraints]
        if len(keys) != len(set(keys)):
            raise ValueError("mutation constraint keys must be unique")

    def to_record(self) -> Record:
        return {
            "supported": self.supported,
            "reason": self.reason,
            "constraints": dict(self.constraints),
        }


@dataclass(frozen=True, slots=True)
class TensorChunk:
    """Ephemeral framework-neutral tensor bytes returned by a live session."""

    component_id: ComponentId
    offset: int
    data: memoryview

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("tensor chunk offset must be non-negative")


class UnsupportedCapabilityError(RuntimeError):
    """Raised before work when an adapter lacks a required capability."""

    def __init__(
        self,
        adapter: AdapterIdentity,
        capability: AdapterCapability,
        *,
        reason: str | None = None,
    ) -> None:
        detail = f": {reason}" if reason else ""
        super().__init__(
            f"adapter {adapter.name}@{adapter.version} does not support {capability.value}{detail}"
        )
        self.adapter = adapter
        self.capability = capability
        self.reason = reason


def require_capability(
    adapter: AdapterIdentity,
    capabilities: frozenset[AdapterCapability],
    capability: AdapterCapability,
    *,
    reason: str | None = None,
) -> None:
    """Fail explicitly before an unsupported operation starts."""
    if capability not in capabilities:
        raise UnsupportedCapabilityError(adapter, capability, reason=reason)


@runtime_checkable
class ModelSession(Protocol):
    """Opaque live model session; adapters retain framework-specific objects."""

    @property
    def source(self) -> ModelSource: ...

    @property
    def adapter_identity(self) -> AdapterIdentity: ...

    @property
    def capabilities(self) -> frozenset[AdapterCapability]: ...

    def components(self) -> Iterator[ComponentDescriptor]: ...

    def tensors(self) -> Iterator[TensorDescriptor]: ...

    def resolve_component(self, component_id: ComponentId) -> ComponentDescriptor: ...

    def read_tensor(
        self,
        component_id: ComponentId,
        *,
        byte_range: tuple[int, int] | None = None,
    ) -> TensorChunk: ...

    def mutation_support(
        self,
        operation: str,
        component_id: ComponentId,
        parameters: Mapping[str, Primitive],
    ) -> MutationSupport: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


@runtime_checkable
class ModelAdapter(Protocol):
    """Capability-oriented entry point for one or more model formats."""

    @property
    def identity(self) -> AdapterIdentity: ...

    @property
    def formats(self) -> frozenset[ModelFormat]: ...

    @property
    def capabilities(self) -> frozenset[AdapterCapability]: ...

    def can_open(self, source: ModelSource) -> bool: ...

    def open(
        self,
        source: ModelSource,
        options: OpenOptions,
    ) -> AbstractContextManager[ModelSession]: ...


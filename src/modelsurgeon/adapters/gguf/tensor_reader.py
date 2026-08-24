"""Stable GGUF tensor handles and bounded lazy payload reads."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from os.path import normcase
from pathlib import Path
from types import MappingProxyType

from modelsurgeon.adapters.gguf.container import (
    GGUFContainer,
    GGUFTensorDescriptor,
    MemoryMappedGGUF,
)
from modelsurgeon.adapters.gguf.quantization import GGUF_STORAGE_LAYOUTS, GGMLQuantizationType


class GGUFTensorReadError(ValueError):
    """Base error for invalid tensor handles or payload read requests."""


class UnknownGGUFTensorError(GGUFTensorReadError):
    """Raised when no exact tensor name exists in an index."""


class StaleGGUFTensorHandleError(GGUFTensorReadError):
    """Raised when a handle does not belong to the active source index."""


class GGUFTensorBoundsError(GGUFTensorReadError):
    """Raised when a request could escape its tensor byte range."""


class GGUFTensorReadLimitError(GGUFTensorReadError):
    """Raised before a read can allocate more than the configured chunk bound."""


@dataclass(frozen=True, slots=True)
class GGUFTensorReadLimits:
    max_chunk_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_chunk_bytes <= 0:
            raise ValueError("GGUF maximum chunk bytes must be positive")


@dataclass(frozen=True, slots=True)
class GGUFTensorHandle:
    """Immutable reference to one validated descriptor, never to its payload."""

    source_key: str
    ordinal: int
    name: str
    dimensions: tuple[int, ...]
    quant_type: GGMLQuantizationType
    byte_size: int
    encoded_block_bytes: int
    logical_block_values: int

    @property
    def block_count(self) -> int:
        return self.byte_size // self.encoded_block_bytes


@dataclass(frozen=True, slots=True)
class GGUFTensorIndex:
    """Deterministically ordered tensor handles with exact-name lookup."""

    source_key: str
    tensors: tuple[GGUFTensorHandle, ...]
    _by_name: Mapping[str, GGUFTensorHandle] = field(repr=False, compare=False)

    def tensor(self, name: str) -> GGUFTensorHandle:
        try:
            return self._by_name[name]
        except KeyError as error:
            raise UnknownGGUFTensorError(f"GGUF tensor {name!r} is not present") from error


@dataclass(frozen=True, slots=True)
class GGUFTensorChunk:
    """One bounded copy of complete encoded blocks."""

    handle: GGUFTensorHandle
    block_offset: int
    block_count: int
    tensor_byte_offset: int
    element_offset: int
    data: bytes


def _source_key(container: GGUFContainer) -> str:
    normalized_path = normcase(str(Path(container.path).resolve()))
    return (
        f"gguf:{normalized_path}:{container.file_size}:{container.version}:"
        f"{container.data_offset}"
    )


def build_tensor_index(container: GGUFContainer) -> GGUFTensorIndex:
    """Build handles from descriptors without accessing any tensor payload byte."""
    source_key = _source_key(container)
    handles = tuple(
        GGUFTensorHandle(
            source_key=source_key,
            ordinal=ordinal,
            name=tensor.name,
            dimensions=tensor.dimensions,
            quant_type=tensor.quant_type,
            byte_size=tensor.byte_size,
            encoded_block_bytes=GGUF_STORAGE_LAYOUTS[tensor.quant_type].type_size,
            logical_block_values=GGUF_STORAGE_LAYOUTS[tensor.quant_type].block_size,
        )
        for ordinal, tensor in enumerate(container.tensors)
    )
    return GGUFTensorIndex(
        source_key,
        handles,
        MappingProxyType({handle.name: handle for handle in handles}),
    )


class GGUFTensorReader:
    """Lazily copy bounded tensor bytes from an active read-only GGUF mapping."""

    def __init__(
        self,
        source: MemoryMappedGGUF,
        *,
        limits: GGUFTensorReadLimits | None = None,
    ) -> None:
        self.source = source
        self.limits = limits or GGUFTensorReadLimits()
        self.index = build_tensor_index(source.container)

    def descriptor(self, handle: GGUFTensorHandle) -> GGUFTensorDescriptor:
        """Resolve a handle to its immutable descriptor after source validation."""

        if handle.source_key != self.index.source_key:
            raise StaleGGUFTensorHandleError("tensor handle belongs to a different GGUF source")
        if handle.ordinal < 0 or handle.ordinal >= len(self.index.tensors):
            raise StaleGGUFTensorHandleError("tensor handle ordinal is outside the source index")
        expected = self.index.tensors[handle.ordinal]
        if handle != expected:
            raise StaleGGUFTensorHandleError(
                "tensor handle does not match the indexed source descriptor"
            )
        return self.source.container.tensors[handle.ordinal]

    def read_bytes(
        self,
        handle: GGUFTensorHandle,
        tensor_byte_offset: int,
        size: int,
    ) -> bytes:
        """Copy an arbitrary bounded byte span wholly inside one tensor."""
        descriptor = self.descriptor(handle)
        if self.source.closed:
            raise GGUFTensorReadError("GGUF source is closed")
        if size < 0 or size > self.limits.max_chunk_bytes:
            raise GGUFTensorReadLimitError(
                f"requested {size} bytes exceeds configured chunk limit "
                f"{self.limits.max_chunk_bytes}"
            )
        if (
            tensor_byte_offset < 0
            or tensor_byte_offset > descriptor.byte_size - size
        ):
            raise GGUFTensorBoundsError(
                f"tensor byte range [{tensor_byte_offset}, {tensor_byte_offset + size}) "
                f"escapes {handle.name!r} size {descriptor.byte_size}"
            )
        return self.source.raw_bytes(
            descriptor.data_offset + tensor_byte_offset,
            size,
            max_bytes=self.limits.max_chunk_bytes,
        )

    def read_blocks(
        self,
        handle: GGUFTensorHandle,
        block_offset: int,
        block_count: int,
    ) -> GGUFTensorChunk:
        """Copy a bounded span of complete encoded codec blocks."""
        if block_offset < 0 or block_count < 0:
            raise GGUFTensorBoundsError("GGUF block offsets and counts must be non-negative")
        if block_offset > handle.block_count - block_count:
            raise GGUFTensorBoundsError(
                f"block range [{block_offset}, {block_offset + block_count}) escapes "
                f"{handle.name!r} block count {handle.block_count}"
            )
        tensor_byte_offset = block_offset * handle.encoded_block_bytes
        byte_count = block_count * handle.encoded_block_bytes
        data = self.read_bytes(handle, tensor_byte_offset, byte_count)
        return GGUFTensorChunk(
            handle,
            block_offset,
            block_count,
            tensor_byte_offset,
            block_offset * handle.logical_block_values,
            data,
        )

    def iter_chunks(
        self,
        handle: GGUFTensorHandle,
        *,
        max_chunk_bytes: int | None = None,
    ) -> Iterator[GGUFTensorChunk]:
        """Yield complete-block chunks no larger than the requested read bound."""
        chunk_limit = (
            self.limits.max_chunk_bytes
            if max_chunk_bytes is None
            else max_chunk_bytes
        )
        if chunk_limit <= 0 or chunk_limit > self.limits.max_chunk_bytes:
            raise GGUFTensorReadLimitError(
                f"chunk limit {chunk_limit} is outside 1..{self.limits.max_chunk_bytes}"
            )
        blocks_per_chunk = chunk_limit // handle.encoded_block_bytes
        if blocks_per_chunk == 0:
            raise GGUFTensorReadLimitError(
                f"chunk limit {chunk_limit} cannot hold one {handle.quant_type.value} "
                f"block of {handle.encoded_block_bytes} bytes"
            )
        block_offset = 0
        while block_offset < handle.block_count:
            block_count = min(blocks_per_chunk, handle.block_count - block_offset)
            yield self.read_blocks(handle, block_offset, block_count)
            block_offset += block_count

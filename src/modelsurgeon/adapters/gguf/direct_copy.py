"""Bounded byte-for-byte GGUF tensor copy sources."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from modelsurgeon.adapters.gguf.tensor_reader import (
    GGUFTensorHandle,
    GGUFTensorReader,
    GGUFTensorReadLimitError,
)
from modelsurgeon.adapters.gguf.writer import GGUFWriteTensor


@dataclass(frozen=True, slots=True)
class GGUFUnchangedTensorCopy:
    """Lazily bridge one indexed source tensor into the streaming writer."""

    reader: GGUFTensorReader
    handle: GGUFTensorHandle
    max_chunk_bytes: int

    def __post_init__(self) -> None:
        self.reader.descriptor(self.handle)
        if (
            self.max_chunk_bytes < self.handle.encoded_block_bytes
            or self.max_chunk_bytes > self.reader.limits.max_chunk_bytes
        ):
            raise GGUFTensorReadLimitError(
                f"copy chunk limit {self.max_chunk_bytes} must fit one encoded block and "
                f"remain within reader limit {self.reader.limits.max_chunk_bytes}"
            )

    @property
    def peak_payload_buffer_bytes(self) -> int:
        """Maximum encoded payload allocation retained by the copy iterator."""

        blocks = self.max_chunk_bytes // self.handle.encoded_block_bytes
        return blocks * self.handle.encoded_block_bytes

    def chunks(self) -> Iterator[bytes]:
        """Yield source bytes exactly once in complete, bounded codec blocks."""

        for chunk in self.reader.iter_chunks(
            self.handle, max_chunk_bytes=self.max_chunk_bytes
        ):
            yield chunk.data

    def as_write_tensor(self) -> GGUFWriteTensor:
        descriptor = self.reader.descriptor(self.handle)
        return GGUFWriteTensor(
            self.handle.name,
            self.handle.dimensions,
            descriptor.ggml_type_id,
            self.chunks(),
        )


def copy_unchanged_gguf_tensor(
    reader: GGUFTensorReader,
    handle: GGUFTensorHandle,
    *,
    max_chunk_bytes: int,
) -> GGUFUnchangedTensorCopy:
    """Create a validated bounded source for one unmodified physical tensor."""

    return GGUFUnchangedTensorCopy(reader, handle, max_chunk_bytes)

"""Transactional, bounded-memory GGUF v2/v3 output construction."""

from __future__ import annotations

import hashlib
import os
import secrets
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from modelsurgeon.adapters.gguf.container import (
    GGML_TYPE_IDS,
    GGUFScalar,
    GGUFValue,
    GGUFValueType,
    gguf_tensor_byte_size,
    open_gguf,
)
from modelsurgeon.adapters.gguf.disk import GGUFDiskPlan, monitor_gguf_disk
from modelsurgeon.adapters.gguf.quantization import ByteOrder


class GGUFWriteError(ValueError):
    """Raised when output cannot be encoded, validated, or safely published."""


@dataclass(frozen=True, slots=True)
class GGUFWriteMetadata:
    key: str
    value_type: GGUFValueType
    value: GGUFValue
    element_type: GGUFValueType | None = None


@dataclass(frozen=True, slots=True)
class GGUFWriteTensor:
    name: str
    dimensions: tuple[int, ...]
    ggml_type_id: int
    chunks: Iterable[bytes | bytearray | memoryview]


@dataclass(frozen=True, slots=True)
class GGUFPlannedTensor:
    name: str
    relative_offset: int
    data_offset: int
    byte_size: int


@dataclass(frozen=True, slots=True)
class GGUFOutputLayout:
    version: int
    byte_order: ByteOrder
    alignment: int
    data_offset: int
    total_bytes: int
    header: bytes
    tensors: tuple[GGUFPlannedTensor, ...]


@dataclass(frozen=True, slots=True)
class GGUFWriteResult:
    path: Path
    file_size: int
    sha256: str
    tensors: tuple[GGUFPlannedTensor, ...]
    tensor_sha256: tuple[tuple[str, str], ...]


_SCALAR_CODES: dict[GGUFValueType, str] = {
    GGUFValueType.UINT8: "B",
    GGUFValueType.INT8: "b",
    GGUFValueType.UINT16: "H",
    GGUFValueType.INT16: "h",
    GGUFValueType.UINT32: "I",
    GGUFValueType.INT32: "i",
    GGUFValueType.FLOAT32: "f",
    GGUFValueType.BOOL: "B",
    GGUFValueType.UINT64: "Q",
    GGUFValueType.INT64: "q",
    GGUFValueType.FLOAT64: "d",
}


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _prefix(byte_order: ByteOrder) -> str:
    return "<" if byte_order is ByteOrder.LITTLE else ">"


def _string(value: str, prefix: str) -> bytes:
    if not value:
        raise GGUFWriteError("GGUF names and metadata keys cannot be empty")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise GGUFWriteError("GGUF strings must be valid UTF-8") from error
    return struct.pack(prefix + "Q", len(encoded)) + encoded


def _scalar(value_type: GGUFValueType, value: GGUFScalar, prefix: str) -> bytes:
    if value_type is GGUFValueType.STRING:
        if not isinstance(value, str):
            raise GGUFWriteError("STRING metadata requires a string")
        return _string(value, prefix)
    code = _SCALAR_CODES.get(value_type)
    if code is None:
        raise GGUFWriteError(f"unsupported scalar metadata type {value_type.name}")
    if value_type is GGUFValueType.BOOL:
        if not isinstance(value, bool):
            raise GGUFWriteError("BOOL metadata requires a bool")
        value = int(value)
    elif isinstance(value, (bool, str)):
        raise GGUFWriteError(f"{value_type.name} metadata requires a numeric value")
    try:
        return struct.pack(prefix + code, value)
    except (struct.error, TypeError, OverflowError) as error:
        raise GGUFWriteError(f"value is outside {value_type.name} representation") from error


def _metadata(entry: GGUFWriteMetadata, prefix: str) -> bytes:
    encoded = bytearray(_string(entry.key, prefix))
    encoded.extend(struct.pack(prefix + "I", int(entry.value_type)))
    if entry.value_type is GGUFValueType.ARRAY:
        element_type = entry.element_type
        if element_type in (None, GGUFValueType.ARRAY):
            raise GGUFWriteError("ARRAY metadata requires a non-array element type")
        if not isinstance(entry.value, tuple):
            raise GGUFWriteError("ARRAY metadata requires an immutable tuple")
        assert element_type is not None
        encoded.extend(struct.pack(prefix + "I", int(element_type)))
        encoded.extend(struct.pack(prefix + "Q", len(entry.value)))
        for value in entry.value:
            encoded.extend(_scalar(element_type, value, prefix))
    else:
        if entry.element_type is not None or isinstance(entry.value, tuple):
            raise GGUFWriteError("scalar metadata cannot declare array elements")
        encoded.extend(_scalar(entry.value_type, entry.value, prefix))
    return bytes(encoded)


def _validated_metadata(
    metadata: tuple[GGUFWriteMetadata, ...], alignment: int
) -> tuple[GGUFWriteMetadata, ...]:
    keys = [entry.key for entry in metadata]
    if len(keys) != len(set(keys)):
        raise GGUFWriteError("GGUF metadata keys must be unique")
    alignment_entry = next(
        (entry for entry in metadata if entry.key == "general.alignment"), None
    )
    if alignment_entry is None:
        return (*metadata, GGUFWriteMetadata("general.alignment", GGUFValueType.UINT32, alignment))
    if (
        alignment_entry.value_type is not GGUFValueType.UINT32
        or alignment_entry.value != alignment
    ):
        raise GGUFWriteError("general.alignment must match the writer alignment")
    return metadata


def plan_gguf_output(
    metadata: tuple[GGUFWriteMetadata, ...],
    tensors: tuple[GGUFWriteTensor, ...],
    *,
    version: int = 3,
    byte_order: ByteOrder = ByteOrder.LITTLE,
    alignment: int = 32,
) -> GGUFOutputLayout:
    """Compute descriptors, offsets, and total size without consuming tensor chunks."""

    if version not in (2, 3) or (version == 2 and byte_order is ByteOrder.BIG):
        raise GGUFWriteError("GGUF v2 is little-endian; v3 supports either byte order")
    if alignment < 8 or alignment > 1 << 20 or alignment & (alignment - 1):
        raise GGUFWriteError("alignment must be a power of two from 8 through 1048576")
    names = [tensor.name for tensor in tensors]
    if len(names) != len(set(names)):
        raise GGUFWriteError("GGUF tensor names must be unique")
    entries = _validated_metadata(metadata, alignment)
    prefix = _prefix(byte_order)
    header = bytearray(b"GGUF")
    header.extend(struct.pack(prefix + "IQQ", version, len(tensors), len(entries)))
    for entry in entries:
        header.extend(_metadata(entry, prefix))

    relative = 0
    raw_tensors: list[tuple[GGUFWriteTensor, int, int]] = []
    for tensor in tensors:
        quant_type = GGML_TYPE_IDS.get(tensor.ggml_type_id)
        if quant_type is None:
            raise GGUFWriteError(f"unsupported ggml type id {tensor.ggml_type_id}")
        try:
            byte_size = gguf_tensor_byte_size(quant_type, tensor.dimensions)
        except ValueError as error:
            raise GGUFWriteError(str(error)) from error
        relative = _align(relative, alignment)
        raw_tensors.append((tensor, relative, byte_size))
        relative += byte_size
    for tensor, tensor_offset, _ in raw_tensors:
        header.extend(_string(tensor.name, prefix))
        header.extend(struct.pack(prefix + "I", len(tensor.dimensions)))
        for dimension in tensor.dimensions:
            header.extend(struct.pack(prefix + "Q", dimension))
        header.extend(struct.pack(prefix + "IQ", tensor.ggml_type_id, tensor_offset))
    data_offset = _align(len(header), alignment)
    planned = tuple(
        GGUFPlannedTensor(tensor.name, relative_offset, data_offset + relative_offset, size)
        for tensor, relative_offset, size in raw_tensors
    )
    total = data_offset + relative if tensors else data_offset
    return GGUFOutputLayout(
        version, byte_order, alignment, data_offset, total, bytes(header), planned
    )


def _write(stream: object, digest: object, data: bytes | memoryview) -> None:
    stream.write(data)  # type: ignore[attr-defined]
    digest.update(data)  # type: ignore[attr-defined]


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_gguf_transactionally(
    destination: str | Path,
    metadata: tuple[GGUFWriteMetadata, ...],
    tensors: tuple[GGUFWriteTensor, ...],
    disk_plan: GGUFDiskPlan,
    *,
    version: int = 3,
    byte_order: ByteOrder = ByteOrder.LITTLE,
    alignment: int = 32,
) -> GGUFWriteResult:
    """Stream a validated staging file, fsync it, then atomically publish it."""

    output = Path(destination).resolve()
    if output != disk_plan.output_path:
        raise GGUFWriteError("disk plan output does not match writer destination")
    if output.exists():
        raise GGUFWriteError("GGUF destination already exists")
    layout = plan_gguf_output(
        metadata, tensors, version=version, byte_order=byte_order, alignment=alignment
    )
    if layout.total_bytes > disk_plan.estimate.aligned_output_bytes:
        raise GGUFWriteError("planned GGUF size exceeds the disk preflight estimate")
    monitor_gguf_disk(
        disk_plan,
        output_remaining_bytes=layout.total_bytes,
        scratch_remaining_bytes=disk_plan.estimate.scratch_bytes,
    )
    staging = output.with_name(f".{output.name}.modelsurgeon-{secrets.token_hex(8)}.tmp")
    digest = hashlib.sha256()
    tensor_digests: list[tuple[str, str]] = []
    try:
        with staging.open("xb") as stream:
            _write(stream, digest, layout.header)
            _write(stream, digest, bytes(layout.data_offset - len(layout.header)))
            cursor = layout.data_offset
            for tensor, planned in zip(tensors, layout.tensors, strict=True):
                padding = planned.data_offset - cursor
                if padding:
                    _write(stream, digest, bytes(padding))
                    cursor += padding
                written = 0
                tensor_digest = hashlib.sha256()
                for chunk in tensor.chunks:
                    view = memoryview(chunk).cast("B")
                    if written > planned.byte_size - len(view):
                        raise GGUFWriteError(f"tensor {tensor.name!r} supplied too many bytes")
                    _write(stream, digest, view)
                    tensor_digest.update(view)
                    written += len(view)
                if written != planned.byte_size:
                    raise GGUFWriteError(
                        f"tensor {tensor.name!r} supplied {written} bytes; "
                        f"expected {planned.byte_size}"
                    )
                cursor += written
                tensor_digests.append((tensor.name, tensor_digest.hexdigest()))
            if cursor != layout.total_bytes:
                raise GGUFWriteError("written GGUF size does not match the planned layout")
            stream.flush()
            os.fsync(stream.fileno())
        with open_gguf(staging) as validation:
            actual = validation.container
            if actual.file_size != layout.total_bytes or tuple(
                (item.name, item.data_offset, item.byte_size) for item in actual.tensors
            ) != tuple(
                (item.name, item.data_offset, item.byte_size) for item in layout.tensors
            ):
                raise GGUFWriteError("staged GGUF offsets or sizes failed validation")
        try:
            os.link(staging, output)
        except FileExistsError as error:
            raise GGUFWriteError("GGUF destination appeared during publication") from error
        staging.unlink()
        _fsync_directory(output.parent)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return GGUFWriteResult(
        output,
        layout.total_bytes,
        digest.hexdigest(),
        layout.tensors,
        tuple(tensor_digests),
    )

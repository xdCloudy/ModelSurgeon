"""Bounded, read-only memory-mapped GGUF container parsing."""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from itertools import pairwise
from operator import mul
from pathlib import Path
from types import TracebackType
from typing import cast

from modelsurgeon.adapters.gguf.quantization import (
    QUANT_LAYOUTS,
    ByteOrder,
    CodecContractError,
    GGMLQuantizationType,
    QuantizationFamily,
)

_MAX_SIGNED_OFFSET = (1 << 63) - 1
_DEFAULT_ALIGNMENT = 32
_GGML_TYPE_IDS = {
    0: GGMLQuantizationType.F32,
    1: GGMLQuantizationType.F16,
    8: GGMLQuantizationType.Q8_0,
    10: GGMLQuantizationType.Q2_K,
    11: GGMLQuantizationType.Q3_K,
    12: GGMLQuantizationType.Q4_K,
    13: GGMLQuantizationType.Q5_K,
    14: GGMLQuantizationType.Q6_K,
    15: GGMLQuantizationType.Q8_K,
    16: GGMLQuantizationType.IQ2_XXS,
    17: GGMLQuantizationType.IQ2_XS,
    18: GGMLQuantizationType.IQ3_XXS,
    19: GGMLQuantizationType.IQ1_S,
    20: GGMLQuantizationType.IQ4_NL,
    21: GGMLQuantizationType.IQ3_S,
    22: GGMLQuantizationType.IQ2_S,
    23: GGMLQuantizationType.IQ4_XS,
    29: GGMLQuantizationType.IQ1_M,
    30: GGMLQuantizationType.BF16,
}


class GGUFParseError(ValueError):
    """Base error for unsupported or malformed GGUF containers."""


class CorruptGGUFError(GGUFParseError):
    """Raised when a GGUF byte range or structural invariant is invalid."""


class UnsupportedGGUFVersionError(GGUFParseError):
    """Raised when the container revision is not in the v2/v3 contract."""


class UnsupportedGGUFTypeError(GGUFParseError):
    """Raised for metadata or tensor types without a pinned layout."""


class GGUFResourceLimitError(GGUFParseError):
    """Raised before an input can force an excessive index allocation."""


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


GGUFScalar = int | float | bool | str
GGUFValue = GGUFScalar | tuple[GGUFScalar, ...]


@dataclass(frozen=True, slots=True)
class GGUFParserLimits:
    """Allocation limits applied before decoding attacker-controlled counts."""

    max_metadata_entries: int = 1_000_000
    max_tensors: int = 1_000_000
    max_dimensions: int = 8
    max_key_bytes: int = 65_536
    max_string_bytes: int = 16 * 1024 * 1024
    max_array_elements: int = 1_000_000

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.max_metadata_entries,
                self.max_tensors,
                self.max_dimensions,
                self.max_key_bytes,
                self.max_string_bytes,
                self.max_array_elements,
            )
        ):
            raise ValueError("GGUF parser limits must be positive")


@dataclass(frozen=True, slots=True)
class GGUFMetadataEntry:
    key: str
    value_type: GGUFValueType
    value: GGUFValue
    raw_offset: int
    raw_size: int
    element_type: GGUFValueType | None = None


@dataclass(frozen=True, slots=True)
class GGUFTensorDescriptor:
    name: str
    dimensions: tuple[int, ...]
    ggml_type_id: int
    quant_type: GGMLQuantizationType
    relative_offset: int
    data_offset: int
    byte_size: int
    descriptor_offset: int
    descriptor_size: int

    @property
    def data_end(self) -> int:
        return self.data_offset + self.byte_size


@dataclass(frozen=True, slots=True)
class GGUFContainer:
    path: Path
    file_size: int
    version: int
    byte_order: ByteOrder
    alignment: int
    data_offset: int
    metadata: tuple[GGUFMetadataEntry, ...]
    tensors: tuple[GGUFTensorDescriptor, ...]

    def metadata_entry(self, key: str) -> GGUFMetadataEntry | None:
        """Return one metadata entry without exposing a mutable index."""
        return next((entry for entry in self.metadata if entry.key == key), None)

    def tensor(self, name: str) -> GGUFTensorDescriptor | None:
        """Return one tensor descriptor by exact physical name."""
        return next((tensor for tensor in self.tensors if tensor.name == name), None)


class _Cursor:
    def __init__(self, mapping: mmap.mmap, byte_order: ByteOrder, limits: GGUFParserLimits):
        self.mapping = mapping
        self.byte_order = "<" if byte_order is ByteOrder.LITTLE else ">"
        self.limits = limits
        self.offset = 0

    def require(self, size: int, label: str) -> None:
        if size < 0 or self.offset > len(self.mapping) - size:
            raise CorruptGGUFError(
                f"{label} at byte {self.offset} exceeds file size {len(self.mapping)}"
            )

    def unpack(self, code: str, label: str) -> int | float:
        size = struct.calcsize(code)
        self.require(size, label)
        (value,) = struct.unpack_from(self.byte_order + code, self.mapping, self.offset)
        self.offset += size
        return cast(int | float, value)

    def string(self, label: str, maximum: int) -> str:
        length = int(self.unpack("Q", f"{label} length"))
        if length > maximum:
            raise GGUFResourceLimitError(
                f"{label} length {length} exceeds configured limit {maximum}"
            )
        self.require(length, label)
        start = self.offset
        self.offset += length
        try:
            return self.mapping[start : self.offset].decode("utf-8")
        except UnicodeDecodeError as error:
            raise CorruptGGUFError(f"{label} is not valid UTF-8") from error


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


def _value_type(raw: int, label: str) -> GGUFValueType:
    try:
        return GGUFValueType(raw)
    except ValueError as error:
        raise UnsupportedGGUFTypeError(f"unsupported {label} type id {raw}") from error


def _read_scalar(cursor: _Cursor, value_type: GGUFValueType, label: str) -> GGUFScalar:
    if value_type is GGUFValueType.STRING:
        return cursor.string(label, cursor.limits.max_string_bytes)
    code = _SCALAR_CODES.get(value_type)
    if code is None:
        raise UnsupportedGGUFTypeError(f"unsupported scalar {label} type {value_type.name}")
    value = cursor.unpack(code, label)
    if value_type is GGUFValueType.BOOL:
        if value not in (0, 1):
            raise CorruptGGUFError(f"{label} bool value must be encoded as 0 or 1")
        return bool(value)
    return value


def _read_value(
    cursor: _Cursor, value_type: GGUFValueType, label: str
) -> tuple[GGUFValue, GGUFValueType | None]:
    if value_type is not GGUFValueType.ARRAY:
        return _read_scalar(cursor, value_type, label), None
    element_type = _value_type(int(cursor.unpack("I", f"{label} element type")), "array")
    if element_type is GGUFValueType.ARRAY:
        raise UnsupportedGGUFTypeError("nested GGUF metadata arrays are not supported")
    length = int(cursor.unpack("Q", f"{label} array length"))
    if length > cursor.limits.max_array_elements:
        raise GGUFResourceLimitError(
            f"{label} array length {length} exceeds configured limit "
            f"{cursor.limits.max_array_elements}"
        )
    values = tuple(
        _read_scalar(cursor, element_type, f"{label}[{index}]")
        for index in range(length)
    )
    return values, element_type


def _align(offset: int, alignment: int) -> int:
    if offset < 0 or alignment <= 0 or offset > _MAX_SIGNED_OFFSET - (alignment - 1):
        raise CorruptGGUFError("GGUF alignment arithmetic exceeds signed 64-bit range")
    return (offset + alignment - 1) & -alignment


def _tensor_size(quant_type: GGMLQuantizationType, dimensions: tuple[int, ...]) -> int:
    layout = QUANT_LAYOUTS[quant_type]
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise CorruptGGUFError("GGUF tensor dimensions must be positive")
    if dimensions[0] % layout.block_size:
        raise CorruptGGUFError(
            f"{quant_type.value} contiguous dimension {dimensions[0]} is not divisible "
            f"by block size {layout.block_size}"
        )
    elements = reduce(mul, dimensions, 1)
    if elements > _MAX_SIGNED_OFFSET:
        raise CorruptGGUFError("GGUF tensor element count exceeds signed 64-bit range")
    try:
        size = layout.encoded_size(elements)
    except CodecContractError as error:
        raise CorruptGGUFError(str(error)) from error
    if size > _MAX_SIGNED_OFFSET:
        raise CorruptGGUFError("GGUF tensor byte size exceeds signed 64-bit range")
    return size


class MemoryMappedGGUF:
    """Own a read-only mapping and its immutable, payload-free descriptor index."""

    def __init__(self, path: str | Path, *, limits: GGUFParserLimits | None = None):
        self.path = Path(path)
        self.limits = limits or GGUFParserLimits()
        self._stream = self.path.open("rb")
        try:
            self._mapping = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_READ)
            self.container = self._parse()
        except Exception:
            mapping = getattr(self, "_mapping", None)
            if mapping is not None:
                mapping.close()
            self._stream.close()
            raise

    def __enter__(self) -> MemoryMappedGGUF:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the virtual mapping and source handle."""
        if not self._mapping.closed:
            self._mapping.close()
        if not self._stream.closed:
            self._stream.close()

    def raw_bytes(self, offset: int, size: int, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
        """Copy one explicitly bounded source span; never return a live mmap view."""
        if max_bytes <= 0 or size < 0 or size > max_bytes:
            raise GGUFResourceLimitError(
                f"requested byte span {size} exceeds configured call limit {max_bytes}"
            )
        if offset < 0 or offset > len(self._mapping) - size:
            raise CorruptGGUFError("requested byte span is outside the GGUF file")
        return self._mapping[offset : offset + size]

    def _parse(self) -> GGUFContainer:
        mapping = self._mapping
        if len(mapping) < 24:
            raise CorruptGGUFError("GGUF file is shorter than the v2/v3 header")
        if mapping[:4] != b"GGUF":
            raise CorruptGGUFError("GGUF magic is missing")
        raw_version = mapping[4:8]
        if raw_version == struct.pack("<I", 2):
            version, byte_order = 2, ByteOrder.LITTLE
        elif raw_version == struct.pack("<I", 3):
            version, byte_order = 3, ByteOrder.LITTLE
        elif raw_version == struct.pack(">I", 3):
            version, byte_order = 3, ByteOrder.BIG
        else:
            little = struct.unpack("<I", raw_version)[0]
            big = struct.unpack(">I", raw_version)[0]
            raise UnsupportedGGUFVersionError(
                f"unsupported GGUF version word (little={little}, big={big}); expected v2 or v3"
            )

        cursor = _Cursor(mapping, byte_order, self.limits)
        cursor.offset = 8
        tensor_count = int(cursor.unpack("Q", "tensor count"))
        metadata_count = int(cursor.unpack("Q", "metadata count"))
        if tensor_count > self.limits.max_tensors:
            raise GGUFResourceLimitError(
                f"tensor count {tensor_count} exceeds configured limit {self.limits.max_tensors}"
            )
        if metadata_count > self.limits.max_metadata_entries:
            raise GGUFResourceLimitError(
                f"metadata count {metadata_count} exceeds configured limit "
                f"{self.limits.max_metadata_entries}"
            )

        metadata: list[GGUFMetadataEntry] = []
        metadata_keys: set[str] = set()
        for index in range(metadata_count):
            entry_start = cursor.offset
            key = cursor.string(f"metadata key {index}", self.limits.max_key_bytes)
            if key in metadata_keys:
                raise CorruptGGUFError(f"duplicate GGUF metadata key {key!r}")
            metadata_keys.add(key)
            value_type = _value_type(int(cursor.unpack("I", f"metadata {key!r} type")), "metadata")
            value, element_type = _read_value(cursor, value_type, f"metadata {key!r}")
            metadata.append(
                GGUFMetadataEntry(
                    key,
                    value_type,
                    value,
                    entry_start,
                    cursor.offset - entry_start,
                    element_type,
                )
            )

        raw_descriptors: list[tuple[str, tuple[int, ...], int, int, int, int]] = []
        tensor_names: set[str] = set()
        for index in range(tensor_count):
            descriptor_start = cursor.offset
            name = cursor.string(f"tensor name {index}", self.limits.max_key_bytes)
            if name in tensor_names:
                raise CorruptGGUFError(f"duplicate GGUF tensor name {name!r}")
            tensor_names.add(name)
            dimension_count = int(cursor.unpack("I", f"tensor {name!r} dimension count"))
            if not 1 <= dimension_count <= self.limits.max_dimensions:
                raise GGUFResourceLimitError(
                    f"tensor {name!r} dimension count {dimension_count} is outside 1.."
                    f"{self.limits.max_dimensions}"
                )
            dimensions = tuple(
                int(cursor.unpack("Q", f"tensor {name!r} dimension {axis}"))
                for axis in range(dimension_count)
            )
            ggml_type_id = int(cursor.unpack("I", f"tensor {name!r} ggml type"))
            relative_offset = int(cursor.unpack("Q", f"tensor {name!r} offset"))
            raw_descriptors.append(
                (
                    name,
                    dimensions,
                    ggml_type_id,
                    relative_offset,
                    descriptor_start,
                    cursor.offset - descriptor_start,
                )
            )

        alignment = _DEFAULT_ALIGNMENT
        alignment_entry = next(
            (entry for entry in metadata if entry.key == "general.alignment"), None
        )
        if alignment_entry is not None:
            if alignment_entry.value_type is not GGUFValueType.UINT32:
                raise CorruptGGUFError("general.alignment must have GGUF UINT32 type")
            if not isinstance(alignment_entry.value, int) or isinstance(
                alignment_entry.value, bool
            ):
                raise CorruptGGUFError("general.alignment must contain an integer")
            alignment = alignment_entry.value
        if alignment < 8 or alignment > 1 << 20 or alignment & (alignment - 1):
            raise CorruptGGUFError(
                "general.alignment must be a power of two from 8 through 1048576"
            )
        data_offset = _align(cursor.offset, alignment)
        if data_offset > len(mapping):
            raise CorruptGGUFError("aligned tensor-data base is outside the GGUF file")

        tensors: list[GGUFTensorDescriptor] = []
        for name, dimensions, ggml_type_id, relative_offset, start, size in raw_descriptors:
            quant_type = _GGML_TYPE_IDS.get(ggml_type_id)
            if quant_type is None:
                raise UnsupportedGGUFTypeError(
                    f"tensor {name!r} uses unsupported ggml type id {ggml_type_id}"
                )
            if relative_offset > _MAX_SIGNED_OFFSET - data_offset:
                raise CorruptGGUFError(f"tensor {name!r} offset overflows signed 64-bit range")
            if relative_offset % alignment:
                raise CorruptGGUFError(
                    f"tensor {name!r} relative offset {relative_offset} is not aligned "
                    f"to {alignment}"
                )
            byte_size = _tensor_size(quant_type, dimensions)
            absolute_offset = data_offset + relative_offset
            if absolute_offset > len(mapping) - byte_size:
                raise CorruptGGUFError(
                    f"tensor {name!r} byte range [{absolute_offset}, "
                    f"{absolute_offset + byte_size}) exceeds file size {len(mapping)}"
                )
            tensors.append(
                GGUFTensorDescriptor(
                    name,
                    dimensions,
                    ggml_type_id,
                    quant_type,
                    relative_offset,
                    absolute_offset,
                    byte_size,
                    start,
                    size,
                )
            )

        by_offset = sorted(tensors, key=lambda tensor: (tensor.data_offset, tensor.name))
        for previous, current in pairwise(by_offset):
            if current.data_offset < previous.data_end:
                raise CorruptGGUFError(
                    f"tensor byte ranges overlap: {previous.name!r} ends at "
                    f"{previous.data_end}, {current.name!r} starts at {current.data_offset}"
                )

        if any(
            QUANT_LAYOUTS[tensor.quant_type].family is not QuantizationFamily.DENSE
            for tensor in tensors
        ):
            quantization_version = next(
                (
                    entry
                    for entry in metadata
                    if entry.key == "general.quantization_version"
                ),
                None,
            )
            if (
                quantization_version is None
                or quantization_version.value_type is not GGUFValueType.UINT32
            ):
                raise CorruptGGUFError(
                    "quantized GGUF tensors require UINT32 general.quantization_version"
                )

        return GGUFContainer(
            self.path,
            len(mapping),
            version,
            byte_order,
            alignment,
            data_offset,
            tuple(metadata),
            tuple(tensors),
        )


def open_gguf(
    path: str | Path, *, limits: GGUFParserLimits | None = None
) -> MemoryMappedGGUF:
    """Open and index a GGUF v2/v3 file without reading tensor payloads."""
    return MemoryMappedGGUF(path, limits=limits)

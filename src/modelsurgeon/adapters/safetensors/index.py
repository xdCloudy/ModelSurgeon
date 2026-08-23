"""Lazy safetensors and sharded-index metadata inspection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

_MAX_HEADER_BYTES = 100 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class SafetensorsIndexError(ValueError):
    """Raised when a safetensors container or shard index is malformed."""


@dataclass(frozen=True, slots=True)
class SafetensorEntry:
    """One tensor's payload location and type without materializing its data."""

    tensor_name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str
    data_offset: int
    byte_size: int

    def to_record(self) -> dict[str, str | int | list[int]]:
        return {
            "tensor_name": self.tensor_name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "shard": self.shard,
            "data_offset": self.data_offset,
            "byte_size": self.byte_size,
        }


def _product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _read_header(path: Path) -> tuple[dict[str, object], int, int]:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise SafetensorsIndexError(f"{path} is too short for a safetensors header")
        header_size = int.from_bytes(prefix, "little", signed=False)
        if header_size == 0 or header_size > _MAX_HEADER_BYTES:
            raise SafetensorsIndexError(f"{path} has invalid header size {header_size}")
        if 8 + header_size > file_size:
            raise SafetensorsIndexError(f"{path} header extends beyond end of file")
        header_bytes = stream.read(header_size)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsIndexError(f"{path} contains an invalid JSON header") from exc
    if not isinstance(header, dict):
        raise SafetensorsIndexError(f"{path} safetensors header must be a JSON object")
    return cast(dict[str, object], header), 8 + header_size, file_size


def inspect_safetensors_file(path: Path) -> tuple[SafetensorEntry, ...]:
    """Read only one container's header and validate all tensor byte ranges."""
    if not path.is_file():
        raise SafetensorsIndexError(f"safetensors shard does not exist: {path}")
    header, data_start, file_size = _read_header(path)
    data_bytes = file_size - data_start
    entries: list[SafetensorEntry] = []
    ranges: list[tuple[int, int, str]] = []
    for tensor_name, raw_metadata in header.items():
        if tensor_name == "__metadata__":
            if not isinstance(raw_metadata, Mapping):
                raise SafetensorsIndexError(f"{path} __metadata__ must be an object")
            continue
        if not tensor_name or not isinstance(raw_metadata, Mapping):
            raise SafetensorsIndexError(f"{path} tensor {tensor_name!r} metadata must be an object")
        dtype = raw_metadata.get("dtype")
        raw_shape = raw_metadata.get("shape")
        raw_offsets = raw_metadata.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            raise SafetensorsIndexError(f"{path} tensor {tensor_name!r} has unsupported dtype")
        if not isinstance(raw_shape, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in raw_shape
        ):
            raise SafetensorsIndexError(f"{path} tensor {tensor_name!r} has invalid shape")
        if (
            not isinstance(raw_offsets, list)
            or len(raw_offsets) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in raw_offsets
            )
        ):
            raise SafetensorsIndexError(f"{path} tensor {tensor_name!r} has invalid data offsets")
        start, end = raw_offsets
        if start < 0 or end < start or end > data_bytes:
            raise SafetensorsIndexError(
                f"{path} tensor {tensor_name!r} byte range is out of bounds"
            )
        shape = tuple(raw_shape)
        byte_size = end - start
        expected_bytes = _product(shape) * _DTYPE_BYTES[dtype]
        if byte_size != expected_bytes:
            raise SafetensorsIndexError(
                f"{path} tensor {tensor_name!r} declares {byte_size} bytes; "
                f"shape and dtype require {expected_bytes}"
            )
        entries.append(
            SafetensorEntry(
                tensor_name=tensor_name,
                shape=shape,
                dtype=dtype,
                shard=path.name,
                data_offset=data_start + start,
                byte_size=byte_size,
            )
        )
        ranges.append((start, end, tensor_name))

    for previous, current in zip(sorted(ranges), sorted(ranges)[1:], strict=False):
        if current[0] < previous[1]:
            raise SafetensorsIndexError(
                f"{path} tensors {previous[2]!r} and {current[2]!r} have overlapping payloads"
            )
    return tuple(sorted(entries, key=lambda entry: entry.tensor_name))


def _load_shard_index(path: Path) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsIndexError(f"invalid safetensors shard index {path}") from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("weight_map"), Mapping):
        raise SafetensorsIndexError(f"{path} must contain a weight_map object")
    weight_map = value["weight_map"]
    assert isinstance(weight_map, Mapping)
    if not weight_map or not all(
        isinstance(name, str) and name and isinstance(shard, str) and shard
        for name, shard in weight_map.items()
    ):
        raise SafetensorsIndexError(f"{path} weight_map contains invalid entries")
    return {str(name): str(shard) for name, shard in weight_map.items()}


def inspect_safetensors(path: Path) -> tuple[SafetensorEntry, ...]:
    """Inspect a single file, shard index, or model directory lazily."""
    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        if index_path.is_file():
            path = index_path
        else:
            candidates = sorted(path.glob("*.safetensors"))
            if len(candidates) != 1:
                raise SafetensorsIndexError(
                    f"{path} must contain one safetensors file or model.safetensors.index.json"
                )
            path = candidates[0]
    if path.suffix == ".safetensors":
        return inspect_safetensors_file(path)
    if not path.name.endswith(".safetensors.index.json"):
        raise SafetensorsIndexError(f"unsupported safetensors input: {path}")

    weight_map = _load_shard_index(path)
    expected_by_shard: dict[str, set[str]] = {}
    for tensor_name, shard in weight_map.items():
        expected_by_shard.setdefault(shard, set()).add(tensor_name)

    result: list[SafetensorEntry] = []
    for shard, expected_names in sorted(expected_by_shard.items()):
        shard_entries = inspect_safetensors_file(path.parent / shard)
        actual_names = {entry.tensor_name for entry in shard_entries}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise SafetensorsIndexError(
                f"shard {shard} disagrees with weight_map; missing={missing}, extra={extra}"
            )
        result.extend(shard_entries)
    return tuple(sorted(result, key=lambda entry: entry.tensor_name))

"""Atomic publication of physically modified safetensors checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from modelsurgeon.adapters.safetensors.index import inspect_safetensors
from modelsurgeon.surgery.checkpoint_destination import AtomicCheckpointDestination

SAFETENSORS_CHECKPOINT_SCHEMA_VERSION: Final[int] = 1
_HASH_CHUNK_BYTES = 1024 * 1024
_DTYPE_CODES = {
    "torch.bool": "BOOL",
    "torch.uint8": "U8",
    "torch.int8": "I8",
    "torch.int16": "I16",
    "torch.uint16": "U16",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.int32": "I32",
    "torch.uint32": "U32",
    "torch.float32": "F32",
    "torch.int64": "I64",
    "torch.uint64": "U64",
    "torch.float64": "F64",
}


class SafetensorsCheckpointError(RuntimeError):
    """Raised before publication when checkpoint integrity cannot be established."""


@dataclass(frozen=True, slots=True)
class SafetensorsTensorRecord:
    tensor_name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str
    byte_size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SafetensorsCheckpointReport:
    destination: Path
    tensors: tuple[SafetensorsTensorRecord, ...]
    source_shard_sha256: tuple[tuple[str, str], ...]
    max_shard_bytes: int
    schema_version: int = SAFETENSORS_CHECKPOINT_SCHEMA_VERSION

    @property
    def sharded(self) -> bool:
        return len({tensor.shard for tensor in self.tensors}) > 1


@dataclass(frozen=True, slots=True)
class _PlannedTensor:
    name: str
    tensor: Any
    shape: tuple[int, ...]
    dtype: str
    byte_size: int
    sha256: str


def write_safetensors_checkpoint_atomic(
    source: str | Path,
    destination: str | Path,
    tensors: Mapping[str, Any],
    configuration: Mapping[str, object],
    *,
    max_shard_bytes: int = 5 * 1024 * 1024 * 1024,
    max_tensor_bytes: int = 2 * 1024 * 1024 * 1024,
) -> SafetensorsCheckpointReport:
    """Stage, verify, and atomically publish one immutable safetensors checkpoint."""

    if max_shard_bytes <= 0 or max_tensor_bytes <= 0:
        raise SafetensorsCheckpointError("checkpoint byte limits must be positive")
    torch = __import__("torch")
    planned = _plan_tensors(torch, tensors, max_shard_bytes, max_tensor_bytes)
    try:
        config_bytes = (json.dumps(configuration, indent=2, sort_keys=True) + "\n").encode()
    except (TypeError, ValueError) as error:
        raise SafetensorsCheckpointError("configuration must be JSON serializable") from error

    source_path = Path(source).expanduser().absolute().resolve(strict=False)
    source_hashes = _source_shard_hashes(source_path)
    groups = _group_shards(planned, max_shard_bytes)
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise SafetensorsCheckpointError(
            "safetensors checkpoint writing requires the hf extra"
        ) from error

    destination_path = Path(destination)
    with AtomicCheckpointDestination(source_path, destination_path) as target:
        target.staging_path.mkdir()
        shard_names = _shard_names(len(groups))
        weight_map: dict[str, str] = {}
        try:
            for shard_name, group in zip(shard_names, groups, strict=True):
                save_file(
                    {item.name: item.tensor for item in group}, target.staging_path / shard_name
                )
                weight_map.update((item.name, shard_name) for item in group)
            if len(groups) > 1:
                index = {
                    "metadata": {"total_size": sum(item.byte_size for item in planned)},
                    "weight_map": dict(sorted(weight_map.items())),
                }
                _write_fsynced(
                    target.staging_path / "model.safetensors.index.json",
                    (json.dumps(index, indent=2, sort_keys=True) + "\n").encode(),
                )
            _write_fsynced(target.staging_path / "config.json", config_bytes)
            for shard_name in shard_names:
                _fsync_file(target.staging_path / shard_name)
            records = _verify_staging(target.staging_path, planned, weight_map)
            if _source_shard_hashes(source_path) != source_hashes:
                raise SafetensorsCheckpointError("source checkpoint shards changed during write")
            published = target.publish()
        except (OSError, ValueError, SafetensorsCheckpointError) as error:
            if isinstance(error, SafetensorsCheckpointError):
                raise
            raise SafetensorsCheckpointError("safetensors checkpoint staging failed") from error
    return SafetensorsCheckpointReport(published, records, source_hashes, max_shard_bytes)


def _plan_tensors(
    torch: Any,
    tensors: Mapping[str, Any],
    max_shard_bytes: int,
    max_tensor_bytes: int,
) -> tuple[_PlannedTensor, ...]:
    if not tensors:
        raise SafetensorsCheckpointError("checkpoint must contain at least one tensor")
    result: list[_PlannedTensor] = []
    for name, tensor in sorted(tensors.items()):
        if not isinstance(name, str) or not name or name == "__metadata__" or "/" in name:
            raise SafetensorsCheckpointError(f"invalid safetensors tensor name {name!r}")
        if not isinstance(tensor, torch.Tensor):
            raise SafetensorsCheckpointError(f"checkpoint value {name!r} is not a torch tensor")
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise SafetensorsCheckpointError(
                f"checkpoint tensor {name!r} must be contiguous CPU data"
            )
        dtype = _DTYPE_CODES.get(str(tensor.dtype))
        if dtype is None:
            raise SafetensorsCheckpointError(f"checkpoint tensor {name!r} has unsupported dtype")
        byte_size = int(tensor.numel()) * int(tensor.element_size())
        if byte_size > max_tensor_bytes or byte_size > max_shard_bytes:
            raise SafetensorsCheckpointError(f"checkpoint tensor {name!r} exceeds byte limits")
        result.append(
            _PlannedTensor(
                name,
                tensor,
                tuple(int(value) for value in tensor.shape),
                dtype,
                byte_size,
                _tensor_sha256(tensor),
            )
        )
    return tuple(result)


def _tensor_sha256(tensor: Any) -> str:
    raw = tensor.detach().view(dtype=__import__("torch").uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def _group_shards(
    tensors: tuple[_PlannedTensor, ...], max_shard_bytes: int
) -> tuple[tuple[_PlannedTensor, ...], ...]:
    groups: list[list[_PlannedTensor]] = [[]]
    size = 0
    for tensor in tensors:
        if groups[-1] and size + tensor.byte_size > max_shard_bytes:
            groups.append([])
            size = 0
        groups[-1].append(tensor)
        size += tensor.byte_size
    return tuple(tuple(group) for group in groups)


def _shard_names(count: int) -> tuple[str, ...]:
    if count == 1:
        return ("model.safetensors",)
    return tuple(f"model-{index:05d}-of-{count:05d}.safetensors" for index in range(1, count + 1))


def _source_shard_hashes(source: Path) -> tuple[tuple[str, str], ...]:
    entries = inspect_safetensors(source)
    root = source if source.is_dir() else source.parent
    paths = tuple(sorted({entry.shard for entry in entries}))
    return tuple((name, _file_sha256(root / name)) for name in paths)


def _verify_staging(
    staging: Path,
    planned: tuple[_PlannedTensor, ...],
    weight_map: Mapping[str, str],
) -> tuple[SafetensorsTensorRecord, ...]:
    entries = inspect_safetensors(staging)
    expected = {item.name: item for item in planned}
    if {entry.tensor_name for entry in entries} != set(expected):
        raise SafetensorsCheckpointError("reloaded tensor names do not match checkpoint plan")
    records: list[SafetensorsTensorRecord] = []
    for entry in entries:
        item = expected[entry.tensor_name]
        shard = weight_map[entry.tensor_name]
        checksum = _payload_sha256(staging / shard, entry.data_offset, entry.byte_size)
        if (entry.shape, entry.dtype, entry.byte_size, checksum) != (
            item.shape,
            item.dtype,
            item.byte_size,
            item.sha256,
        ):
            raise SafetensorsCheckpointError(
                f"reloaded tensor {entry.tensor_name!r} does not match checkpoint plan"
            )
        records.append(
            SafetensorsTensorRecord(
                entry.tensor_name,
                entry.shape,
                entry.dtype,
                shard,
                entry.byte_size,
                checksum,
            )
        )
    return tuple(records)


def _payload_sha256(path: Path, offset: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        remaining = length
        while remaining:
            block = stream.read(min(_HASH_CHUNK_BYTES, remaining))
            if not block:
                raise SafetensorsCheckpointError("safetensors payload ended during verification")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())

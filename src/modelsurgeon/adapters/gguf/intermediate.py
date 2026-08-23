"""Checksummed, resumable, bounded disk-backed tensor intermediates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from functools import reduce
from operator import mul
from pathlib import Path
from typing import Literal

DISK_TENSOR_SCHEMA_VERSION: Literal[1] = 1
_CHUNK_NAME = re.compile(r"^chunk-([0-9]{8})\.bin$")


class DiskTensorError(ValueError):
    """Base error for invalid or corrupt disk-backed tensor state."""


class StaleDiskTensorError(DiskTensorError):
    """Raised when scratch state differs from its committed manifest."""


@dataclass(frozen=True, slots=True)
class DiskTensorSpec:
    """Immutable identity, shape, type, and allocation bound for an intermediate."""

    tensor_id: str
    shape: tuple[int, ...]
    dtype: str
    item_size: int
    chunk_bytes: int
    plan_sha256: str

    def __post_init__(self) -> None:
        if not self.tensor_id or not self.dtype:
            raise DiskTensorError("tensor identity and dtype cannot be empty")
        if not self.shape or any(value <= 0 for value in self.shape):
            raise DiskTensorError("tensor shape dimensions must be positive")
        if self.item_size <= 0 or self.chunk_bytes <= 0:
            raise DiskTensorError("item size and chunk bytes must be positive")
        if self.chunk_bytes % self.item_size:
            raise DiskTensorError("chunk bytes must be divisible by item size")
        if not re.fullmatch(r"[0-9a-f]{64}", self.plan_sha256):
            raise DiskTensorError("plan checksum must be lowercase SHA-256")

    @property
    def total_bytes(self) -> int:
        return reduce(mul, self.shape, 1) * self.item_size


@dataclass(frozen=True, slots=True)
class DiskTensorChunk:
    index: int
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DiskTensorManifest:
    spec: DiskTensorSpec
    chunks: tuple[DiskTensorChunk, ...]
    complete: bool
    schema_version: Literal[1] = DISK_TENSOR_SCHEMA_VERSION

    @property
    def completed_bytes(self) -> int:
        return sum(chunk.byte_count for chunk in self.chunks)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "spec": {
                "tensor_id": self.spec.tensor_id,
                "shape": list(self.spec.shape),
                "dtype": self.spec.dtype,
                "item_size": self.spec.item_size,
                "chunk_bytes": self.spec.chunk_bytes,
                "plan_sha256": self.spec.plan_sha256,
                "total_bytes": self.spec.total_bytes,
            },
            "chunks": [
                {
                    "index": chunk.index,
                    "byte_count": chunk.byte_count,
                    "sha256": chunk.sha256,
                }
                for chunk in self.chunks
            ],
            "complete": self.complete,
        }


def _exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise DiskTensorError(f"{label} fields do not match schema")


def _manifest_from_record(record: object) -> DiskTensorManifest:
    if not isinstance(record, dict):
        raise DiskTensorError("disk tensor manifest must be an object")
    _exact_keys(record, {"schema_version", "spec", "chunks", "complete"}, "manifest")
    if record["schema_version"] != DISK_TENSOR_SCHEMA_VERSION:
        raise DiskTensorError("unsupported disk tensor manifest schema")
    raw_spec = record["spec"]
    if not isinstance(raw_spec, dict):
        raise DiskTensorError("disk tensor spec must be an object")
    _exact_keys(
        raw_spec,
        {
            "tensor_id",
            "shape",
            "dtype",
            "item_size",
            "chunk_bytes",
            "plan_sha256",
            "total_bytes",
        },
        "spec",
    )
    shape = raw_spec["shape"]
    if not isinstance(shape, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in shape
    ):
        raise DiskTensorError("manifest tensor shape must contain integers")
    strings = (raw_spec["tensor_id"], raw_spec["dtype"], raw_spec["plan_sha256"])
    integers = (
        raw_spec["item_size"],
        raw_spec["chunk_bytes"],
        raw_spec["total_bytes"],
    )
    if any(not isinstance(value, str) for value in strings) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in integers
    ):
        raise DiskTensorError("manifest tensor spec uses invalid field types")
    spec = DiskTensorSpec(
        raw_spec["tensor_id"],
        tuple(shape),
        raw_spec["dtype"],
        raw_spec["item_size"],
        raw_spec["chunk_bytes"],
        raw_spec["plan_sha256"],
    )
    if raw_spec["total_bytes"] != spec.total_bytes:
        raise DiskTensorError("manifest total bytes do not match shape and dtype")
    raw_chunks = record["chunks"]
    if not isinstance(raw_chunks, list):
        raise DiskTensorError("manifest chunks must be an array")
    chunks: list[DiskTensorChunk] = []
    for expected_index, raw in enumerate(raw_chunks):
        if not isinstance(raw, dict):
            raise DiskTensorError("manifest chunk must be an object")
        _exact_keys(raw, {"index", "byte_count", "sha256"}, "chunk")
        if raw["index"] != expected_index:
            raise DiskTensorError("manifest chunk indices must be contiguous")
        count = raw["byte_count"]
        digest = raw["sha256"]
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or count > spec.chunk_bytes
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise DiskTensorError("manifest chunk record is invalid")
        chunks.append(DiskTensorChunk(expected_index, count, digest))
    complete = record["complete"]
    if not isinstance(complete, bool):
        raise DiskTensorError("manifest complete flag must be boolean")
    manifest = DiskTensorManifest(spec, tuple(chunks), complete)
    if manifest.completed_bytes > spec.total_bytes or complete != (
        manifest.completed_bytes == spec.total_bytes
    ):
        raise DiskTensorError("manifest completion does not match committed bytes")
    return manifest


def _hash_file(path: Path, read_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(read_bytes):
            digest.update(data)
    return digest.hexdigest()


class DiskBackedTensor:
    """Append-only chunk store whose manifest is committed after each chunk."""

    def __init__(self, path: Path, manifest: DiskTensorManifest) -> None:
        self.path = path
        self.manifest = manifest

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def peak_buffer_bytes(self) -> int:
        return self.manifest.spec.chunk_bytes

    @classmethod
    def create(cls, path: str | Path, spec: DiskTensorSpec) -> DiskBackedTensor:
        target = Path(path).resolve()
        try:
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise StaleDiskTensorError(f"scratch tensor path already exists: {target}") from error
        tensor = cls(target, DiskTensorManifest(spec, (), False))
        try:
            tensor._commit_manifest()
        except BaseException:
            shutil.rmtree(target)
            raise
        return tensor

    @classmethod
    def resume(
        cls,
        path: str | Path,
        expected_spec: DiskTensorSpec,
        *,
        recover_stale: bool = False,
    ) -> DiskBackedTensor:
        target = Path(path).resolve()
        manifest_path = target / "manifest.json"
        temporary = target / ".manifest.json.tmp"
        if temporary.exists():
            if not recover_stale:
                raise StaleDiskTensorError("interrupted manifest commit detected")
            temporary.unlink()
        try:
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DiskTensorError("disk tensor manifest cannot be read") from error
        manifest = _manifest_from_record(record)
        if manifest.spec != expected_spec:
            raise StaleDiskTensorError("scratch tensor belongs to a different input or plan")
        expected_names = {f"chunk-{chunk.index:08d}.bin" for chunk in manifest.chunks}
        actual_names = {
            item.name for item in target.iterdir() if _CHUNK_NAME.fullmatch(item.name)
        }
        orphans = actual_names - expected_names
        if orphans and not recover_stale:
            raise StaleDiskTensorError(f"uncommitted scratch chunks detected: {sorted(orphans)}")
        for name in orphans:
            (target / name).unlink()
        for chunk in manifest.chunks:
            chunk_path = target / f"chunk-{chunk.index:08d}.bin"
            if (
                not chunk_path.is_file()
                or chunk_path.stat().st_size != chunk.byte_count
                or _hash_file(chunk_path, expected_spec.chunk_bytes) != chunk.sha256
            ):
                raise StaleDiskTensorError(
                    f"scratch chunk {chunk.index} is missing, truncated, or corrupt"
                )
        return cls(target, manifest)

    def _commit_manifest(self) -> None:
        temporary = self.path / ".manifest.json.tmp"
        payload = json.dumps(
            self.manifest.to_record(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.manifest_path)

    def append(self, data: bytes | bytearray | memoryview) -> DiskTensorChunk:
        if self.manifest.complete:
            raise DiskTensorError("disk tensor is already complete")
        view = memoryview(data).cast("B")
        remaining = self.manifest.spec.total_bytes - self.manifest.completed_bytes
        expected = min(self.manifest.spec.chunk_bytes, remaining)
        if len(view) != expected:
            raise DiskTensorError(f"next tensor chunk must contain exactly {expected} bytes")
        index = len(self.manifest.chunks)
        final_path = self.path / f"chunk-{index:08d}.bin"
        temporary = self.path / f".chunk-{index:08d}.tmp"
        digest = hashlib.sha256(view).hexdigest()
        with temporary.open("xb") as stream:
            stream.write(view)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, final_path)
        chunk = DiskTensorChunk(index, len(view), digest)
        chunks = (*self.manifest.chunks, chunk)
        self.manifest = DiskTensorManifest(
            self.manifest.spec,
            chunks,
            sum(item.byte_count for item in chunks) == self.manifest.spec.total_bytes,
        )
        self._commit_manifest()
        return chunk

    def iter_chunks(self) -> Iterator[bytes]:
        for chunk in self.manifest.chunks:
            path = self.path / f"chunk-{chunk.index:08d}.bin"
            data = path.read_bytes()
            if len(data) != chunk.byte_count or hashlib.sha256(data).hexdigest() != chunk.sha256:
                raise StaleDiskTensorError(f"scratch chunk {chunk.index} changed after resume")
            yield data

    def cleanup(self) -> None:
        """Remove this exact scratch tensor directory and all committed chunks."""

        shutil.rmtree(self.path)

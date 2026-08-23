"""Tensor-boundary checkpointing for resumable transactional GGUF output."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from modelsurgeon.adapters.gguf.container import open_gguf
from modelsurgeon.adapters.gguf.disk import GGUFDiskPlan, monitor_gguf_disk
from modelsurgeon.adapters.gguf.quantization import ByteOrder
from modelsurgeon.adapters.gguf.writer import (
    GGUFWriteError,
    GGUFWriteMetadata,
    GGUFWriteResult,
    GGUFWriteTensor,
    plan_gguf_output,
)

GGUF_RESUME_SCHEMA_VERSION: Literal[1] = 1


class GGUFResumeError(GGUFWriteError):
    """Raised when a staged output cannot be safely resumed."""


@dataclass(frozen=True, slots=True)
class GGUFResumeTensor:
    name: str
    end_offset: int
    sha256: str


@dataclass(frozen=True, slots=True)
class GGUFResumeManifest:
    input_sha256: str
    plan_sha256: str
    committed_bytes: int
    committed_sha256: str
    completed_tensors: tuple[GGUFResumeTensor, ...]
    schema_version: Literal[1] = GGUF_RESUME_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "input_sha256": self.input_sha256,
            "plan_sha256": self.plan_sha256,
            "committed_bytes": self.committed_bytes,
            "committed_sha256": self.committed_sha256,
            "completed_tensors": [
                {
                    "name": tensor.name,
                    "end_offset": tensor.end_offset,
                    "sha256": tensor.sha256,
                }
                for tensor in self.completed_tensors
            ],
        }


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _manifest(record: object) -> GGUFResumeManifest:
    if not isinstance(record, dict) or set(record) != {
        "schema_version",
        "input_sha256",
        "plan_sha256",
        "committed_bytes",
        "committed_sha256",
        "completed_tensors",
    }:
        raise GGUFResumeError("resume manifest fields do not match schema")
    if record["schema_version"] != GGUF_RESUME_SCHEMA_VERSION:
        raise GGUFResumeError("unsupported GGUF resume manifest schema")
    committed_bytes = record["committed_bytes"]
    completed = record["completed_tensors"]
    if (
        not _is_digest(record["input_sha256"])
        or not _is_digest(record["plan_sha256"])
        or not _is_digest(record["committed_sha256"])
        or not isinstance(committed_bytes, int)
        or isinstance(committed_bytes, bool)
        or committed_bytes < 0
        or not isinstance(completed, list)
    ):
        raise GGUFResumeError("resume manifest contains invalid field types")
    tensors: list[GGUFResumeTensor] = []
    for raw in completed:
        if not isinstance(raw, dict) or set(raw) != {"name", "end_offset", "sha256"}:
            raise GGUFResumeError("resume tensor fields do not match schema")
        if (
            not isinstance(raw["name"], str)
            or not isinstance(raw["end_offset"], int)
            or isinstance(raw["end_offset"], bool)
            or not _is_digest(raw["sha256"])
        ):
            raise GGUFResumeError("resume tensor contains invalid field types")
        tensors.append(
            GGUFResumeTensor(raw["name"], raw["end_offset"], raw["sha256"])
        )
    return GGUFResumeManifest(
        record["input_sha256"],
        record["plan_sha256"],
        committed_bytes,
        record["committed_sha256"],
        tuple(tensors),
    )


def _paths(destination: Path) -> tuple[Path, Path, Path]:
    staging = destination.with_name(f".{destination.name}.modelsurgeon.resume")
    manifest = destination.with_name(f".{destination.name}.modelsurgeon.resume.json")
    temporary = destination.with_name(f".{destination.name}.modelsurgeon.resume.json.tmp")
    return staging, manifest, temporary


def _hash_prefix(path: Path, byte_count: int) -> hashlib._Hash:
    digest = hashlib.sha256()
    remaining = byte_count
    with path.open("rb") as stream:
        while remaining:
            data = stream.read(min(1024 * 1024, remaining))
            if not data:
                raise GGUFResumeError("staged output is shorter than its committed range")
            digest.update(data)
            remaining -= len(data)
    return digest


def _commit(path: Path, temporary: Path, manifest: GGUFResumeManifest) -> None:
    payload = json.dumps(
        manifest.to_record(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write(stream: BinaryIO, digest: hashlib._Hash, data: bytes | memoryview) -> None:
    stream.write(data)
    digest.update(data)


def _plan_digest(header: bytes, total_bytes: int) -> str:
    digest = hashlib.sha256(header)
    digest.update(total_bytes.to_bytes(8, "little"))
    return digest.hexdigest()


def _validate_manifest_layout(
    manifest: GGUFResumeManifest,
    tensor_names: tuple[str, ...],
    tensor_ends: tuple[int, ...],
    data_offset: int,
) -> None:
    completed = manifest.completed_tensors
    if len(completed) > len(tensor_names) or tuple(item.name for item in completed) != tensor_names[
        : len(completed)
    ]:
        raise GGUFResumeError("completed tensors do not match the current output plan")
    if any(
        item.end_offset != tensor_ends[index] for index, item in enumerate(completed)
    ):
        raise GGUFResumeError("committed tensor offsets do not match the current output plan")
    expected_bytes = completed[-1].end_offset if completed else data_offset
    if manifest.committed_bytes != expected_bytes:
        raise GGUFResumeError("committed byte count is not a tensor boundary")


def write_gguf_resumably(
    destination: str | Path,
    metadata: tuple[GGUFWriteMetadata, ...],
    tensors: tuple[GGUFWriteTensor, ...],
    disk_plan: GGUFDiskPlan,
    *,
    input_sha256: str,
    version: int = 3,
    byte_order: ByteOrder = ByteOrder.LITTLE,
    alignment: int = 32,
    checkpoint_hook: Callable[[int], None] | None = None,
) -> GGUFWriteResult:
    """Resume at the last committed tensor and publish only a complete valid GGUF."""

    if not _is_digest(input_sha256):
        raise GGUFResumeError("input identity must be lowercase SHA-256")
    output = Path(destination).resolve()
    if output != disk_plan.output_path:
        raise GGUFResumeError("disk plan output does not match writer destination")
    if output.exists():
        raise GGUFResumeError("GGUF destination already exists")
    layout = plan_gguf_output(
        metadata, tensors, version=version, byte_order=byte_order, alignment=alignment
    )
    if layout.total_bytes > disk_plan.estimate.aligned_output_bytes:
        raise GGUFResumeError("planned GGUF size exceeds the disk preflight estimate")
    plan_sha256 = _plan_digest(layout.header, layout.total_bytes)
    staging, manifest_path, temporary = _paths(output)
    if temporary.exists():
        temporary.unlink()
    if staging.exists() != manifest_path.exists():
        raise GGUFResumeError("staged output and resume manifest are incomplete")

    if not staging.exists():
        output_digest = hashlib.sha256()
        with staging.open("xb") as stream:
            _write(stream, output_digest, layout.header)
            _write(stream, output_digest, bytes(layout.data_offset - len(layout.header)))
            stream.flush()
            os.fsync(stream.fileno())
        manifest = GGUFResumeManifest(
            input_sha256,
            plan_sha256,
            layout.data_offset,
            output_digest.hexdigest(),
            (),
        )
        _commit(manifest_path, temporary, manifest)
    else:
        try:
            manifest = _manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise GGUFResumeError("resume manifest cannot be read") from error
        if manifest.input_sha256 != input_sha256 or manifest.plan_sha256 != plan_sha256:
            raise GGUFResumeError("changed input or output plan invalidates resume")
        _validate_manifest_layout(
            manifest,
            tuple(item.name for item in layout.tensors),
            tuple(item.data_offset + item.byte_size for item in layout.tensors),
            layout.data_offset,
        )
        if staging.stat().st_size < manifest.committed_bytes:
            raise GGUFResumeError("staged output is shorter than its committed range")
        output_digest = _hash_prefix(staging, manifest.committed_bytes)
        if output_digest.hexdigest() != manifest.committed_sha256:
            raise GGUFResumeError("staged output committed checksum mismatch")
        with staging.open("r+b") as stream:
            stream.truncate(manifest.committed_bytes)

    monitor_gguf_disk(
        disk_plan,
        output_remaining_bytes=layout.total_bytes - manifest.committed_bytes,
        scratch_remaining_bytes=disk_plan.estimate.scratch_bytes,
    )
    completed_count = len(manifest.completed_tensors)
    completed = list(manifest.completed_tensors)
    with staging.open("ab") as stream:
        cursor = manifest.committed_bytes
        for index in range(completed_count, len(tensors)):
            tensor = tensors[index]
            planned = layout.tensors[index]
            padding = planned.data_offset - cursor
            if padding:
                _write(stream, output_digest, bytes(padding))
                cursor += padding
            tensor_digest = hashlib.sha256()
            written = 0
            for chunk in tensor.chunks:
                view = memoryview(chunk).cast("B")
                if written > planned.byte_size - len(view):
                    raise GGUFResumeError(f"tensor {tensor.name!r} supplied too many bytes")
                _write(stream, output_digest, view)
                tensor_digest.update(view)
                written += len(view)
            if written != planned.byte_size:
                raise GGUFResumeError(
                    f"tensor {tensor.name!r} supplied {written} bytes; expected "
                    f"{planned.byte_size}"
                )
            cursor += written
            stream.flush()
            os.fsync(stream.fileno())
            completed.append(
                GGUFResumeTensor(tensor.name, cursor, tensor_digest.hexdigest())
            )
            manifest = GGUFResumeManifest(
                input_sha256,
                plan_sha256,
                cursor,
                output_digest.hexdigest(),
                tuple(completed),
            )
            _commit(manifest_path, temporary, manifest)
            if checkpoint_hook is not None:
                checkpoint_hook(index)

    if manifest.committed_bytes != layout.total_bytes:
        raise GGUFResumeError("completed output does not match planned size")
    with open_gguf(staging) as validation:
        if validation.container.file_size != layout.total_bytes:
            raise GGUFResumeError("staged GGUF failed final size validation")
    try:
        os.link(staging, output)
    except FileExistsError as error:
        raise GGUFResumeError("GGUF destination appeared during publication") from error
    staging.unlink()
    manifest_path.unlink()
    return GGUFWriteResult(
        output,
        layout.total_bytes,
        manifest.committed_sha256,
        layout.tensors,
        tuple((item.name, item.sha256) for item in manifest.completed_tensors),
    )


def discard_resumable_gguf(destination: str | Path) -> None:
    """Discard only the exact unpublished staging and manifest files for a destination."""

    output = Path(destination).resolve()
    for path in _paths(output):
        path.unlink(missing_ok=True)

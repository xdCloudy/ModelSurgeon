"""Bounded low-rank replacement of one native quantized GGUF tensor."""

from __future__ import annotations

import hashlib
import importlib
import math
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    CodecRegistry,
    GGUFDiscovery,
    GGUFDiskPlan,
    GGUFTensorHandle,
    GGUFTensorReader,
    GGUFWriteMetadata,
    GGUFWriteResult,
    GGUFWriteTensor,
    MemoryMappedGGUF,
    copy_unchanged_gguf_tensor,
    discover_gguf_components,
    open_gguf,
    write_gguf_resumably,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.gguf_alignment import GGUFQuantizedTensorEdit
from modelsurgeon.surgery.selective_requant import (
    ChangedGGUFFloatChunk,
    GGUFRequantizationLimits,
    GGUFRequantizationReport,
    SelectiveGGUFRequantizer,
)

NATIVE_GGUF_LOW_RANK_SCHEMA_VERSION: Final[int] = 1


class NativeGGUFLowRankError(ValueError):
    """Raised before publish when selected-tensor low-rank replacement is unsafe."""


@dataclass(frozen=True, slots=True)
class NativeGGUFLowRankLimits:
    copy_chunk_bytes: int = 4 * 1024 * 1024
    max_tensor_values: int = 1_048_576
    max_workspace_bytes: int = 512 * 1024 * 1024
    requantization: GGUFRequantizationLimits = field(default_factory=GGUFRequantizationLimits)

    def __post_init__(self) -> None:
        if (
            min(
                self.copy_chunk_bytes,
                self.max_tensor_values,
                self.max_workspace_bytes,
            )
            <= 0
        ):
            raise NativeGGUFLowRankError("native low-rank limits must be positive")


@dataclass(frozen=True, slots=True)
class NativeGGUFLowRankResult:
    write_result: GGUFWriteResult
    output_discovery: GGUFDiscovery
    tensor_name: str
    component_id: ComponentId
    requested_rank: int
    effective_rank: int
    relative_frobenius_error: float
    reconstruction_max_absolute_error: float
    requantization_mean_squared_error: float
    requantization_max_absolute_error: float
    unchanged_tensor_sha256: tuple[tuple[str, str], ...]
    requantization_report: GGUFRequantizationReport
    estimated_workspace_bytes: int
    peak_decode_working_bytes: int
    source_sha256: str
    schema_version: int = NATIVE_GGUF_LOW_RANK_SCHEMA_VERSION


def execute_native_gguf_low_rank_replacement(
    source: MemoryMappedGGUF,
    family: ModelFamily,
    tensor_name: str,
    rank: int,
    destination: str | Path,
    disk_plan: GGUFDiskPlan,
    codecs: CodecRegistry,
    *,
    limits: NativeGGUFLowRankLimits | None = None,
) -> NativeGGUFLowRankResult:
    """Decode, approximate, and requantize one selected matrix without a model-wide decode."""

    resolved = limits or NativeGGUFLowRankLimits()
    discovery = discover_gguf_components(source.container, family=family)
    selected = next(
        (item for item in discovery.tensors if item.descriptor.name == tensor_name), None
    )
    if selected is None:
        raise NativeGGUFLowRankError(f"selected GGUF tensor {tensor_name!r} is absent")
    shape = selected.descriptor.dimensions
    if len(shape) != 2 or rank <= 0 or rank >= min(shape):
        raise NativeGGUFLowRankError("low-rank GGUF replacement requires rank below both axes")
    value_count = math.prod(shape)
    if value_count > resolved.max_tensor_values:
        raise NativeGGUFLowRankError("selected GGUF tensor exceeds decoded-value limit")
    rows, columns = shape[1], shape[0]
    minor = min(rows, columns)
    workspace = 4 * value_count + 8 * (3 * value_count + rows * minor + minor * columns + minor)
    if workspace > resolved.max_workspace_bytes:
        raise NativeGGUFLowRankError(
            f"selected GGUF low-rank workspace {workspace} exceeds limit "
            f"{resolved.max_workspace_bytes}"
        )

    reader = GGUFTensorReader(source)
    handle = reader.index.tensor(tensor_name)
    codec = codecs.resolve(handle.quant_type)
    chunk_bytes = min(resolved.copy_chunk_bytes, reader.limits.max_chunk_bytes)
    decoded = array("f")
    peak_decode = 0
    for chunk in reader.iter_chunks(handle, max_chunk_bytes=chunk_bytes):
        before = len(decoded)
        operation = codec.decode_blocks(
            memoryview(chunk.data), decoded, byte_order=source.container.byte_order
        )
        added = len(decoded) - before
        if operation.element_count != added:
            raise NativeGGUFLowRankError("selected codec returned inconsistent decode counts")
        peak_decode = max(peak_decode, len(chunk.data) + len(decoded) * 4)
    if len(decoded) != value_count:
        raise NativeGGUFLowRankError("selected tensor decode disagrees with physical shape")

    try:
        np = importlib.import_module("numpy")
    except ImportError as error:  # pragma: no cover - gguf dependency boundary
        raise NativeGGUFLowRankError("native GGUF low-rank surgery requires NumPy") from error
    matrix = np.asarray(decoded, dtype=np.float32).reshape(rows, columns).astype(np.float64)
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    approximation = (u[:, :rank] * singular[:rank]) @ vh[:rank, :]
    residual = matrix - approximation
    denominator = float(np.linalg.norm(matrix))
    residual_norm = float(np.linalg.norm(residual))
    relative = 0.0 if denominator == 0.0 else residual_norm / denominator
    maximum = float(np.max(np.abs(residual)))
    changed_values = array("f", approximation.astype(np.float32).ravel())

    edit = GGUFQuantizedTensorEdit(
        selected.component_id,
        handle.quant_type,
        handle.quant_type,
        shape,
        shape,
        (),
        handle.byte_size,
    )
    requantizer = SelectiveGGUFRequantizer(resolved.requantization)

    def changed_chunks() -> Iterator[bytes]:
        for encoded in requantizer.iter_encoded_tensor(
            edit,
            (ChangedGGUFFloatChunk(selected.component_id, 0, changed_values),),
            codecs,
            byte_order=source.container.byte_order,
        ):
            yield encoded.payload

    unchanged: list[tuple[str, str]] = []
    write_tensors: list[GGUFWriteTensor] = []
    for item in reader.index.tensors:
        descriptor = reader.descriptor(item)
        if item.name == tensor_name:
            write_tensors.append(
                GGUFWriteTensor(
                    item.name, item.dimensions, descriptor.ggml_type_id, changed_chunks()
                )
            )
            continue
        copy = copy_unchanged_gguf_tensor(reader, item, max_chunk_bytes=chunk_bytes)
        unchanged.append((item.name, _tensor_sha256(reader, item, chunk_bytes)))
        write_tensors.append(copy.as_write_tensor())
    source_digest = _file_sha256(source.container.path, chunk_bytes)
    metadata = tuple(
        GGUFWriteMetadata(entry.key, entry.value_type, entry.value, entry.element_type)
        for entry in source.container.metadata
    )
    write_result = write_gguf_resumably(
        destination,
        metadata,
        tuple(write_tensors),
        disk_plan,
        input_sha256=source_digest,
        version=source.container.version,
        byte_order=source.container.byte_order,
        alignment=source.container.alignment,
    )
    report = requantizer.report()
    if not report.complete:
        raise NativeGGUFLowRankError("selected tensor requantization did not complete")
    if _file_sha256(source.container.path, chunk_bytes) != source_digest:
        raise NativeGGUFLowRankError("source GGUF changed during low-rank replacement")
    output_hashes = dict(write_result.tensor_sha256)
    for name, digest in unchanged:
        if output_hashes.get(name) != digest:
            raise NativeGGUFLowRankError(f"unchanged tensor {name!r} differs in output")
    with open_gguf(write_result.path) as output:
        output_discovery = discover_gguf_components(output.container, family=discovery.family)
    output_selected = next(
        item for item in output_discovery.tensors if item.descriptor.name == tensor_name
    )
    if output_selected.descriptor.dimensions != shape:
        raise NativeGGUFLowRankError("low-rank output tensor shape changed unexpectedly")
    summaries = report.error_summaries
    sample_count = sum(item.error.sample_count for item in summaries)
    requant_mse = (
        sum(item.error.mean_squared_error * item.error.sample_count for item in summaries)
        / sample_count
    )
    requant_max = max(item.error.max_absolute_error for item in summaries)
    return NativeGGUFLowRankResult(
        write_result,
        output_discovery,
        tensor_name,
        selected.component_id,
        rank,
        rank,
        relative,
        maximum,
        requant_mse,
        requant_max,
        tuple(unchanged),
        report,
        workspace,
        peak_decode,
        source_digest,
    )


def _tensor_sha256(reader: GGUFTensorReader, handle: GGUFTensorHandle, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    for chunk in reader.iter_chunks(handle, max_chunk_bytes=chunk_bytes):
        digest.update(chunk.data)
    return digest.hexdigest()


def _file_sha256(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()

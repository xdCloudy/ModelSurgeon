"""Native byte-preserving GGUF transformer-layer removal."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    GGUFDiscovery,
    GGUFDiskPlan,
    GGUFTensorHandle,
    GGUFTensorReader,
    GGUFValueType,
    GGUFWriteMetadata,
    GGUFWriteResult,
    GGUFWriteTensor,
    MemoryMappedGGUF,
    copy_unchanged_gguf_tensor,
    discover_gguf_components,
    open_gguf,
    resolve_gguf_architecture,
    write_gguf_resumably,
)
from modelsurgeon.adapters.gguf.architecture import MetadataSemantic

NATIVE_GGUF_LAYER_REMOVAL_SCHEMA_VERSION: Final[int] = 1


class NativeGGUFLayerRemovalError(ValueError):
    """Raised when native layer removal cannot preserve the validated contract."""


@dataclass(frozen=True, slots=True)
class NativeGGUFLayerRemovalPlan:
    family: ModelFamily
    architecture: str
    old_layer_count: int
    new_layer_count: int
    removed_layers: tuple[int, ...]
    layer_mapping: tuple[tuple[int, int | None], ...]
    block_count_metadata_key: str
    schema_version: int = NATIVE_GGUF_LAYER_REMOVAL_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class RetainedGGUFTensor:
    source_name: str
    output_name: str
    payload_sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class NativeGGUFLayerRemovalResult:
    write_result: GGUFWriteResult
    output_discovery: GGUFDiscovery
    plan: NativeGGUFLayerRemovalPlan
    retained_tensors: tuple[RetainedGGUFTensor, ...]
    omitted_tensor_names: tuple[str, ...]
    source_sha256: str
    peak_copy_buffer_bytes: int


def plan_native_gguf_transformer_layer_removal(
    discovery: GGUFDiscovery,
    removed_layers: tuple[int, ...],
) -> NativeGGUFLayerRemovalPlan:
    """Create the canonical complete-block omission and renumbering plan."""

    if not removed_layers or removed_layers != tuple(sorted(set(removed_layers))):
        raise NativeGGUFLayerRemovalError("removed GGUF layers must be non-empty and canonical")
    if removed_layers[0] < 0 or removed_layers[-1] >= discovery.shape.layers:
        raise NativeGGUFLayerRemovalError("removed GGUF layer is outside block_count")
    if len(removed_layers) >= discovery.shape.layers:
        raise NativeGGUFLayerRemovalError("native GGUF surgery cannot remove every layer")
    try:
        architecture = resolve_gguf_architecture(discovery.architecture, family=discovery.family)
    except ValueError as error:
        raise NativeGGUFLayerRemovalError(str(error)) from error
    if architecture.contract.version != discovery.contract_version:
        raise NativeGGUFLayerRemovalError("GGUF discovery contract version changed before planning")
    removed = set(removed_layers)
    next_index = 0
    mapping: list[tuple[int, int | None]] = []
    for old_index in range(discovery.shape.layers):
        if old_index in removed:
            mapping.append((old_index, None))
        else:
            mapping.append((old_index, next_index))
            next_index += 1
    return NativeGGUFLayerRemovalPlan(
        discovery.family,
        discovery.architecture,
        discovery.shape.layers,
        next_index,
        removed_layers,
        tuple(mapping),
        architecture.metadata_key(MetadataSemantic.BLOCK_COUNT),
    )


def execute_native_gguf_transformer_layer_removal(
    source: MemoryMappedGGUF,
    plan: NativeGGUFLayerRemovalPlan,
    destination: str | Path,
    disk_plan: GGUFDiskPlan,
    *,
    copy_chunk_bytes: int = 4 * 1024 * 1024,
) -> NativeGGUFLayerRemovalResult:
    """Direct-copy retained encoded tensors into a resumable shortened GGUF."""

    if copy_chunk_bytes <= 0:
        raise NativeGGUFLayerRemovalError("GGUF layer copy chunk limit must be positive")
    source_discovery = discover_gguf_components(source.container, family=plan.family)
    if (
        source_discovery.family is not plan.family
        or source_discovery.architecture != plan.architecture
        or source_discovery.shape.layers != plan.old_layer_count
    ):
        raise NativeGGUFLayerRemovalError("source GGUF identity disagrees with layer plan")
    architecture = resolve_gguf_architecture(plan.architecture, family=source_discovery.family)
    reader = GGUFTensorReader(source)
    chunk_bytes = min(copy_chunk_bytes, reader.limits.max_chunk_bytes)
    retained: list[RetainedGGUFTensor] = []
    omitted: list[str] = []
    write_tensors = []
    peak = 0
    for handle in reader.index.tensors:
        try:
            output_name = architecture.rename_tensor_blocks(handle.name, plan.layer_mapping)
        except ValueError as error:
            raise NativeGGUFLayerRemovalError(str(error)) from error
        if output_name is None:
            omitted.append(handle.name)
            continue
        copy = copy_unchanged_gguf_tensor(reader, handle, max_chunk_bytes=chunk_bytes)
        descriptor = reader.descriptor(handle)
        digest = _tensor_sha256(reader, handle, chunk_bytes)
        retained.append(RetainedGGUFTensor(handle.name, output_name, digest, handle.byte_size))
        write_tensors.append(
            GGUFWriteTensor(output_name, handle.dimensions, descriptor.ggml_type_id, copy.chunks())
        )
        peak = max(peak, copy.peak_payload_buffer_bytes)
    if not omitted:
        raise NativeGGUFLayerRemovalError("layer plan omitted no physical tensors")
    if len({item.output_name for item in retained}) != len(retained):
        raise NativeGGUFLayerRemovalError("layer plan produces duplicate output tensor names")
    metadata = _updated_metadata(source, plan)
    source_digest = _file_sha256(source.container.path, chunk_bytes)
    result = write_gguf_resumably(
        destination,
        metadata,
        tuple(write_tensors),
        disk_plan,
        input_sha256=source_digest,
        version=source.container.version,
        byte_order=source.container.byte_order,
        alignment=source.container.alignment,
    )
    if _file_sha256(source.container.path, chunk_bytes) != source_digest:
        raise NativeGGUFLayerRemovalError("source GGUF changed during layer removal")
    output_hashes = dict(result.tensor_sha256)
    for item in retained:
        if output_hashes.get(item.output_name) != item.payload_sha256:
            raise NativeGGUFLayerRemovalError(
                f"retained tensor {item.source_name!r} is not byte-identical in output"
            )
    with open_gguf(result.path) as output:
        output_discovery = discover_gguf_components(
            output.container, family=source_discovery.family
        )
    if output_discovery.shape.layers != plan.new_layer_count:
        raise NativeGGUFLayerRemovalError("output GGUF block_count disagrees with layer plan")
    actual_names = {item.descriptor.name for item in output_discovery.tensors}
    expected_names = {item.output_name for item in retained}
    if actual_names != expected_names:
        raise NativeGGUFLayerRemovalError(
            "output GGUF canonical tensor mapping disagrees with plan"
        )
    return NativeGGUFLayerRemovalResult(
        result,
        output_discovery,
        plan,
        tuple(retained),
        tuple(omitted),
        source_digest,
        peak,
    )


def _updated_metadata(
    source: MemoryMappedGGUF, plan: NativeGGUFLayerRemovalPlan
) -> tuple[GGUFWriteMetadata, ...]:
    found = False
    result: list[GGUFWriteMetadata] = []
    for entry in source.container.metadata:
        value = entry.value
        if entry.key == plan.block_count_metadata_key:
            found = True
            if entry.value_type not in {GGUFValueType.UINT32, GGUFValueType.UINT64}:
                raise NativeGGUFLayerRemovalError("GGUF block_count metadata has invalid type")
            value = plan.new_layer_count
        result.append(
            GGUFWriteMetadata(
                entry.key,
                entry.value_type,
                cast("int", value) if entry.key == plan.block_count_metadata_key else entry.value,
                entry.element_type,
            )
        )
    if not found:
        raise NativeGGUFLayerRemovalError("GGUF block_count metadata is missing")
    return tuple(result)


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

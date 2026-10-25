"""Fail-closed layout and exact encoded slicing for Qwen3.5 MoE GGUF files."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path

from modelsurgeon.adapters.gguf.container import (
    GGUFContainer,
    GGUFValueType,
    MemoryMappedGGUF,
    open_gguf,
)
from modelsurgeon.adapters.gguf.direct_copy import copy_unchanged_gguf_tensor
from modelsurgeon.adapters.gguf.disk import GGUFDiskPlan
from modelsurgeon.adapters.gguf.quantization import GGUF_STORAGE_LAYOUTS, ByteOrder
from modelsurgeon.adapters.gguf.tensor_reader import (
    GGUFTensorHandle,
    GGUFTensorReader,
)
from modelsurgeon.adapters.gguf.writer import (
    GGUFWriteMetadata,
    GGUFWriteResult,
    GGUFWriteTensor,
    write_gguf_transactionally,
)


class Qwen35MoeSurgeryError(ValueError):
    """The file is outside the narrow validated Qwen3.5 MoE surgery contract."""


@dataclass(frozen=True, slots=True)
class Qwen35MoeBlockLayout:
    layer: int
    router: str
    gate_experts: str
    up_experts: str
    down_experts: str

    @property
    def sliced_tensor_names(self) -> tuple[str, ...]:
        return (self.router, self.gate_experts, self.up_experts, self.down_experts)


@dataclass(frozen=True, slots=True)
class Qwen35MoeGGUFLayout:
    block_count: int
    expert_count: int
    expert_used_count: int
    embedding_length: int
    expert_feed_forward_length: int
    blocks: tuple[Qwen35MoeBlockLayout, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "architecture": "qwen35moe",
            "contract_version": 1,
            "block_count": self.block_count,
            "expert_count": self.expert_count,
            "expert_used_count": self.expert_used_count,
            "embedding_length": self.embedding_length,
            "expert_feed_forward_length": self.expert_feed_forward_length,
            "sliced_tensor_count": sum(len(item.sliced_tensor_names) for item in self.blocks),
            "shared_experts_preserved": True,
            "hybrid_attention_tensors_preserved": True,
        }


@dataclass(frozen=True, slots=True)
class Qwen35MoeSurgeryResult:
    write_result: GGUFWriteResult
    output_layout: Qwen35MoeGGUFLayout
    unchanged_tensor_sha256: tuple[tuple[str, str], ...]
    peak_payload_buffer_bytes: int


def _positive_metadata(container: GGUFContainer, key: str) -> int:
    entry = container.metadata_entry(key)
    if (
        entry is None
        or entry.value_type not in (GGUFValueType.UINT32, GGUFValueType.UINT64)
        or isinstance(entry.value, bool)
        or not isinstance(entry.value, int)
        or entry.value <= 0
    ):
        raise Qwen35MoeSurgeryError(f"GGUF metadata {key!r} must be a positive integer")
    return entry.value


def build_qwen35moe_gguf_layout(container: GGUFContainer) -> Qwen35MoeGGUFLayout:
    """Validate only routed-expert axes; all other Qwen3.5 tensors stay opaque."""

    architecture = container.metadata_entry("general.architecture")
    if (
        architecture is None
        or architecture.value_type is not GGUFValueType.STRING
        or architecture.value != "qwen35moe"
    ):
        raise Qwen35MoeSurgeryError("GGUF general.architecture must be exactly 'qwen35moe'")

    block_count = _positive_metadata(container, "qwen35moe.block_count")
    expert_count = _positive_metadata(container, "qwen35moe.expert_count")
    expert_used_count = _positive_metadata(container, "qwen35moe.expert_used_count")
    embedding_length = _positive_metadata(container, "qwen35moe.embedding_length")
    expert_ffn = _positive_metadata(container, "qwen35moe.expert_feed_forward_length")
    if expert_used_count > expert_count:
        raise Qwen35MoeSurgeryError("expert_used_count cannot exceed expert_count")

    by_name = {tensor.name: tensor for tensor in container.tensors}
    blocks: list[Qwen35MoeBlockLayout] = []
    expected_names: set[str] = set()
    for layer in range(block_count):
        prefix = f"blk.{layer}."
        layout = Qwen35MoeBlockLayout(
            layer,
            prefix + "ffn_gate_inp.weight",
            prefix + "ffn_gate_exps.weight",
            prefix + "ffn_up_exps.weight",
            prefix + "ffn_down_exps.weight",
        )
        expected_names.update(layout.sliced_tensor_names)
        router = by_name.get(layout.router)
        if router is None or router.dimensions != (embedding_length, expert_count):
            raise Qwen35MoeSurgeryError(
                f"{layout.router!r} must have dimensions ({embedding_length}, {expert_count})"
            )
        if router.quant_type.value != "F32":
            raise Qwen35MoeSurgeryError(f"router {layout.router!r} must use F32 storage")
        for name, dimensions in (
            (layout.gate_experts, (embedding_length, expert_ffn, expert_count)),
            (layout.up_experts, (embedding_length, expert_ffn, expert_count)),
            (layout.down_experts, (expert_ffn, embedding_length, expert_count)),
        ):
            tensor = by_name.get(name)
            if tensor is None or tensor.dimensions != dimensions:
                raise Qwen35MoeSurgeryError(
                    f"{name!r} dimensions do not match the declared Qwen3.5 expert layout"
                )
            per_expert_values = prod(dimensions[:-1])
            storage = GGUF_STORAGE_LAYOUTS[tensor.quant_type]
            if (
                per_expert_values % storage.block_size
                or tensor.byte_size % expert_count
                or tensor.byte_size // expert_count % storage.type_size
            ):
                raise Qwen35MoeSurgeryError(
                    f"{name!r} expert slices do not align to complete encoded blocks"
                )
        blocks.append(layout)

    routed_components = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_inp"}
    found_names: set[str] = set()
    for tensor in container.tensors:
        if not tensor.name.startswith("blk."):
            continue
        parts = tensor.name.split(".", 2)
        if len(parts) < 3:
            raise Qwen35MoeSurgeryError(f"malformed block tensor name {tensor.name!r}")
        if parts[2].split(".", 1)[0] in routed_components:
            found_names.add(tensor.name)
    if found_names != expected_names:
        unexpected = sorted(found_names - expected_names)
        missing = sorted(expected_names - found_names)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing[:4]))
        if unexpected:
            detail.append("unmapped " + ", ".join(unexpected[:4]))
        raise Qwen35MoeSurgeryError("routed expert tensor inventory mismatch: " + "; ".join(detail))

    return Qwen35MoeGGUFLayout(
        block_count,
        expert_count,
        expert_used_count,
        embedding_length,
        expert_ffn,
        tuple(blocks),
    )


def _validated_retained_experts(
    layout: Qwen35MoeGGUFLayout,
    retained_by_layer: Mapping[int, tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    if set(retained_by_layer) != set(range(layout.block_count)):
        raise Qwen35MoeSurgeryError("retained expert IDs must be supplied for every block")
    retained_counts: set[int] = set()
    result: list[tuple[int, ...]] = []
    for layer in range(layout.block_count):
        indices = retained_by_layer[layer]
        if (
            not indices
            or tuple(sorted(set(indices))) != indices
            or any(index < 0 or index >= layout.expert_count for index in indices)
        ):
            raise Qwen35MoeSurgeryError(
                f"retained expert IDs for block {layer} must be sorted, unique, and in range"
            )
        retained_counts.add(len(indices))
        result.append(indices)
    if len(retained_counts) != 1:
        raise Qwen35MoeSurgeryError("Qwen3.5 requires a uniform expert count across blocks")
    retained_count = next(iter(retained_counts))
    if retained_count < layout.expert_used_count or retained_count >= layout.expert_count:
        raise Qwen35MoeSurgeryError(
            "retained expert count must be at least expert_used_count and smaller than source"
        )
    return tuple(result)


def rank_qwen35moe_experts_by_router_norm(
    source: MemoryMappedGGUF,
    layout: Qwen35MoeGGUFLayout,
) -> tuple[tuple[int, ...], ...]:
    """Return per-layer expert IDs ordered by ascending static router-column L2 norm."""

    if source.closed:
        raise Qwen35MoeSurgeryError("Qwen3.5 source mapping is closed")
    if build_qwen35moe_gguf_layout(source.container) != layout:
        raise Qwen35MoeSurgeryError("Qwen3.5 layout does not describe this source file")
    reader = GGUFTensorReader(source)
    prefix = "<" if source.container.byte_order is ByteOrder.LITTLE else ">"
    rankings: list[tuple[int, ...]] = []
    for block in layout.blocks:
        handle = reader.index.tensor(block.router)
        squared_norms = [0.0] * layout.expert_count
        values_per_expert = layout.embedding_length
        for chunk in reader.iter_chunks(handle):
            if len(chunk.data) % 4:
                raise Qwen35MoeSurgeryError("F32 router payload is not scalar-aligned")
            values = struct.iter_unpack(prefix + "f", chunk.data)
            first_value = chunk.block_offset * handle.logical_block_values
            for offset, (value,) in enumerate(values, start=first_value):
                if not math.isfinite(value):
                    raise Qwen35MoeSurgeryError(
                        f"router {block.router!r} contains a non-finite weight"
                    )
                expert = offset // values_per_expert
                squared_norms[expert] += value * value
        if len(squared_norms) != layout.expert_count:
            raise Qwen35MoeSurgeryError("router expert dimension disagrees with metadata")
        if any(not math.isfinite(value) for value in squared_norms):
            raise Qwen35MoeSurgeryError("router-column norm overflowed finite arithmetic")
        rankings.append(
            tuple(
                sorted(
                    range(layout.expert_count),
                    key=lambda index: (squared_norms[index], index),
                )
            )
        )
    return tuple(rankings)


def _metadata(source: MemoryMappedGGUF, retained_count: int) -> tuple[GGUFWriteMetadata, ...]:
    updated = False
    result: list[GGUFWriteMetadata] = []
    for entry in source.container.metadata:
        value = entry.value
        if entry.key == "qwen35moe.expert_count":
            if entry.value_type not in (GGUFValueType.UINT32, GGUFValueType.UINT64):
                raise Qwen35MoeSurgeryError("qwen35moe.expert_count must use an unsigned integer")
            value = retained_count
            updated = True
        result.append(GGUFWriteMetadata(entry.key, entry.value_type, value, entry.element_type))
    if not updated:
        raise Qwen35MoeSurgeryError("qwen35moe.expert_count metadata is missing")
    return tuple(result)


def _digest(reader: GGUFTensorReader, handle: GGUFTensorHandle) -> str:
    digest = hashlib.sha256()
    for chunk in reader.iter_chunks(handle):
        digest.update(chunk.data)
    return digest.hexdigest()


def _selected_expert_chunks(
    reader: GGUFTensorReader,
    handle: GGUFTensorHandle,
    *,
    source_expert_count: int,
    retained_indices: tuple[int, ...],
) -> Iterator[bytes]:
    if handle.block_count % source_expert_count:
        raise Qwen35MoeSurgeryError(f"{handle.name!r} is not divided into expert blocks")
    blocks_per_expert = handle.block_count // source_expert_count
    max_blocks = reader.limits.max_chunk_bytes // handle.encoded_block_bytes
    if blocks_per_expert <= 0 or max_blocks <= 0:
        raise Qwen35MoeSurgeryError(f"{handle.name!r} has invalid encoded expert block geometry")
    for expert_index in retained_indices:
        offset = expert_index * blocks_per_expert
        remaining = blocks_per_expert
        while remaining:
            count = min(remaining, max_blocks)
            yield reader.read_blocks(handle, offset, count).data
            offset += count
            remaining -= count


def execute_qwen35moe_expert_pruning(
    source: MemoryMappedGGUF,
    retained_experts_by_layer: Mapping[int, tuple[int, ...]],
    destination: str | Path,
    disk_plan: GGUFDiskPlan,
) -> Qwen35MoeSurgeryResult:
    """Remove explicitly selected routed experts while preserving other tensors."""

    if source.closed:
        raise Qwen35MoeSurgeryError("Qwen3.5 source mapping is closed")
    layout = build_qwen35moe_gguf_layout(source.container)
    retained = _validated_retained_experts(layout, retained_experts_by_layer)
    reader = GGUFTensorReader(source)
    changed_names: dict[str, tuple[int, ...]] = {}
    for block, retained_ids in zip(layout.blocks, retained, strict=True):
        changed_names.update({name: retained_ids for name in block.sliced_tensor_names})

    tensors: list[GGUFWriteTensor] = []
    unchanged_hashes: list[tuple[str, str]] = []
    peak_payload = 0
    for handle in reader.index.tensors:
        descriptor = reader.descriptor(handle)
        selected_ids = changed_names.get(handle.name)
        if selected_ids is None:
            copy = copy_unchanged_gguf_tensor(
                reader, handle, max_chunk_bytes=reader.limits.max_chunk_bytes
            )
            tensors.append(copy.as_write_tensor())
            unchanged_hashes.append((handle.name, _digest(reader, handle)))
            peak_payload = max(peak_payload, copy.peak_payload_buffer_bytes)
            continue
        dimensions = (*handle.dimensions[:-1], len(selected_ids))
        tensors.append(
            GGUFWriteTensor(
                handle.name,
                dimensions,
                descriptor.ggml_type_id,
                _selected_expert_chunks(
                    reader,
                    handle,
                    source_expert_count=layout.expert_count,
                    retained_indices=selected_ids,
                ),
            )
        )
        peak_payload = max(
            peak_payload,
            min(
                handle.encoded_block_bytes * handle.block_count // layout.expert_count,
                reader.limits.max_chunk_bytes,
            ),
        )

    write_result = write_gguf_transactionally(
        destination,
        _metadata(source, len(retained[0])),
        tuple(tensors),
        disk_plan,
        version=source.container.version,
        byte_order=source.container.byte_order,
        alignment=source.container.alignment,
    )
    output_hashes = dict(write_result.tensor_sha256)
    for name, digest in unchanged_hashes:
        if output_hashes.get(name) != digest:
            raise Qwen35MoeSurgeryError(f"untouched tensor {name!r} changed during surgery")
    with open_gguf(write_result.path) as output:
        output_layout = build_qwen35moe_gguf_layout(output.container)
        if output_layout.expert_count != len(retained[0]):
            raise Qwen35MoeSurgeryError("output expert_count disagrees with the surgery plan")
        if output_layout.expert_used_count != layout.expert_used_count:
            raise Qwen35MoeSurgeryError("output expert_used_count changed unexpectedly")
    return Qwen35MoeSurgeryResult(
        write_result,
        output_layout,
        tuple(sorted(unchanged_hashes)),
        peak_payload,
    )

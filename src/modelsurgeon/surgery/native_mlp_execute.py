"""Low-memory execution of coupled native quantized GGUF MLP removal."""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from modelsurgeon.adapters.gguf import (
    CodecRegistry,
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
    write_gguf_resumably,
)
from modelsurgeon.surgery.gguf_alignment import (
    GGUFQuantizedMutationPlan,
    GGUFQuantizedTensorEdit,
    QuantizedEditStrategy,
)
from modelsurgeon.surgery.native_mlp_plan import NativeGGUFMLPRemovalPlan
from modelsurgeon.surgery.physical_plan import PhysicalMutationPlan, PhysicalTensorEdit
from modelsurgeon.surgery.selective_requant import (
    ChangedGGUFFloatChunk,
    GGUFRequantizationErrorSummary,
    GGUFRequantizationLimits,
    SelectiveGGUFRequantizer,
)


class NativeGGUFMLPExecutionError(ValueError):
    """Raised when execution diverges from a validated native MLP plan."""


@dataclass(frozen=True, slots=True)
class NativeGGUFMLPExecutionLimits:
    copy_chunk_bytes: int = 4 * 1024 * 1024
    max_row_working_bytes: int = 16 * 1024 * 1024
    requantization: GGUFRequantizationLimits = field(
        default_factory=GGUFRequantizationLimits
    )

    def __post_init__(self) -> None:
        if self.copy_chunk_bytes <= 0 or self.max_row_working_bytes <= 0:
            raise NativeGGUFMLPExecutionError("native MLP execution limits must be positive")


@dataclass(frozen=True, slots=True)
class NativeGGUFMLPExecutionResult:
    write_result: GGUFWriteResult
    output_discovery: GGUFDiscovery
    unchanged_tensor_sha256: tuple[tuple[str, str], ...]
    requantization_errors: tuple[GGUFRequantizationErrorSummary, ...]
    peak_row_working_bytes: int


def _file_sha256(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk_bytes):
            digest.update(data)
    return digest.hexdigest()


def _tensor_sha256(
    reader: GGUFTensorReader, handle: GGUFTensorHandle, chunk_bytes: int
) -> str:
    digest = hashlib.sha256()
    for chunk in reader.iter_chunks(handle, max_chunk_bytes=chunk_bytes):
        digest.update(chunk.data)
    return digest.hexdigest()


def _metadata(
    source: MemoryMappedGGUF, plan: NativeGGUFMLPRemovalPlan
) -> tuple[GGUFWriteMetadata, ...]:
    updates = {
        item.key: item.value for item in plan.physical_plan.metadata_updates
    }
    known = {entry.key for entry in source.container.metadata}
    missing = set(updates) - known
    if missing:
        raise NativeGGUFMLPExecutionError(
            "planned metadata keys are absent from source: " + ", ".join(sorted(missing))
        )
    output: list[GGUFWriteMetadata] = []
    for entry in source.container.metadata:
        value = updates.get(entry.key, entry.value)
        if entry.key in updates and entry.value_type not in {
            GGUFValueType.UINT32,
            GGUFValueType.UINT64,
        }:
            raise NativeGGUFMLPExecutionError(
                f"planned numeric metadata {entry.key!r} has incompatible source type"
            )
        if entry.key in updates and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise NativeGGUFMLPExecutionError(
                f"planned numeric metadata {entry.key!r} must be a positive integer"
            )
        output.append(
            GGUFWriteMetadata(
                entry.key,
                entry.value_type,
                cast("int", value) if entry.key in updates else entry.value,
                entry.element_type,
            )
        )
    return tuple(output)


def _kept_block_runs(block_count: int, removed: set[int]) -> Iterator[tuple[int, int]]:
    start: int | None = None
    for block in range(block_count + 1):
        keep = block < block_count and block not in removed
        if keep and start is None:
            start = block
        elif not keep and start is not None:
            yield start, block - start
            start = None


class _ChangedTensorSource:
    def __init__(
        self,
        reader: GGUFTensorReader,
        handle: GGUFTensorHandle,
        plan: PhysicalMutationPlan,
        physical: PhysicalTensorEdit,
        quantized: GGUFQuantizedTensorEdit,
        codecs: CodecRegistry,
        limits: NativeGGUFMLPExecutionLimits,
    ) -> None:
        self.reader = reader
        self.handle = handle
        self.plan = plan
        self.physical = physical
        self.quantized = quantized
        self.codecs = codecs
        self.limits = limits
        self.errors: list[GGUFRequantizationErrorSummary] = []
        self.peak_row_working_bytes = 0

    def _outer_copy(self, removed: set[int]) -> Iterator[bytes]:
        blocks_per_row = self.handle.dimensions[0] // self.handle.logical_block_values
        row_count = self.handle.block_count // blocks_per_row
        max_blocks = min(
            self.reader.limits.max_chunk_bytes,
            self.limits.copy_chunk_bytes,
        ) // self.handle.encoded_block_bytes
        if max_blocks <= 0:
            raise NativeGGUFMLPExecutionError(
                f"copy limit cannot hold one {self.handle.quant_type.value} block"
            )
        for row in range(row_count):
            if row not in removed:
                consumed = 0
                while consumed < blocks_per_row:
                    count = min(max_blocks, blocks_per_row - consumed)
                    yield self.reader.read_blocks(
                        self.handle, row * blocks_per_row + consumed, count
                    ).data
                    consumed += count

    def _direct_axis0(self, removed_indices: tuple[int, ...]) -> Iterator[bytes]:
        block_values = self.handle.logical_block_values
        removed_blocks = {index // block_values for index in removed_indices}
        blocks_per_row = self.handle.dimensions[0] // block_values
        rows = self.handle.block_count // blocks_per_row
        max_blocks = min(
            self.reader.limits.max_chunk_bytes,
            self.limits.copy_chunk_bytes,
        ) // self.handle.encoded_block_bytes
        if max_blocks <= 0:
            raise NativeGGUFMLPExecutionError(
                f"copy limit cannot hold one {self.handle.quant_type.value} block"
            )
        for row in range(rows):
            for offset, count in _kept_block_runs(blocks_per_row, removed_blocks):
                consumed = 0
                while consumed < count:
                    chunk_blocks = min(max_blocks, count - consumed)
                    yield self.reader.read_blocks(
                        self.handle,
                        row * blocks_per_row + offset + consumed,
                        chunk_blocks,
                    ).data
                    consumed += chunk_blocks

    def _repack_axis0(self, removed_indices: tuple[int, ...]) -> Iterator[bytes]:
        codec = self.codecs.resolve(self.quantized.quant_type)
        destination_codec = self.codecs.resolve(
            self.quantized.destination_quant_type
        )
        if codec.identity.quant_type is not destination_codec.identity.quant_type:
            raise NativeGGUFMLPExecutionError(
                "native MLP execution currently requires unchanged tensor codec"
            )
        old_width = self.physical.old_shape[0]
        new_width = self.physical.new_shape[0]
        block_values = codec.layout.block_size
        blocks_per_row = old_width // block_values
        rows = self.handle.block_count // blocks_per_row
        # The requantizer may slice the retained row once, so account for that
        # bounded copy in addition to the decoded and retained arrays.
        required = (old_width + 2 * new_width) * 4 + min(
            self.limits.copy_chunk_bytes,
            blocks_per_row * codec.layout.type_size,
        )
        if required > self.limits.max_row_working_bytes:
            raise NativeGGUFMLPExecutionError(
                f"one MLP row requires {required} working bytes, exceeding "
                f"limit {self.limits.max_row_working_bytes}"
            )
        self.peak_row_working_bytes = max(self.peak_row_working_bytes, required)
        removed = set(removed_indices)
        max_blocks = min(
            self.reader.limits.max_chunk_bytes,
            self.limits.copy_chunk_bytes,
        ) // codec.layout.type_size
        if max_blocks <= 0:
            raise NativeGGUFMLPExecutionError(
                f"copy limit cannot hold one {self.handle.quant_type.value} block"
            )
        new_blocks_per_row = new_width // destination_codec.layout.block_size
        for row in range(rows):
            decoded = array("f")
            row_block = row * blocks_per_row
            consumed = 0
            while consumed < blocks_per_row:
                count = min(max_blocks, blocks_per_row - consumed)
                chunk = self.reader.read_blocks(
                    self.handle, row_block + consumed, count
                )
                codec.decode_blocks(
                    memoryview(chunk.data),
                    decoded,
                    byte_order=self.reader.source.container.byte_order,
                )
                consumed += count
            retained = array(
                "f",
                (value for index, value in enumerate(decoded) if index not in removed),
            )
            if len(retained) != new_width:
                raise NativeGGUFMLPExecutionError(
                    "filtered MLP row width disagrees with physical plan"
                )
            requantizer = SelectiveGGUFRequantizer(self.limits.requantization)
            encoded = requantizer.iter_encoded(
                self.quantized_plan,
                (
                    ChangedGGUFFloatChunk(
                        self.physical.component_id,
                        row * new_blocks_per_row,
                        retained,
                    ),
                ),
                self.codecs,
                byte_order=self.reader.source.container.byte_order,
            )
            for encoded_chunk in encoded:
                yield encoded_chunk.payload
            report = requantizer.report()
            self.errors.extend(report.error_summaries)
            combined = required + report.peak_working_bytes
            if combined > self.limits.max_row_working_bytes:
                raise NativeGGUFMLPExecutionError(
                    f"one MLP row and requantization require {combined} working bytes, "
                    f"exceeding limit {self.limits.max_row_working_bytes}"
                )
            self.peak_row_working_bytes = max(self.peak_row_working_bytes, combined)

    @property
    def quantized_plan(self) -> GGUFQuantizedMutationPlan:
        return GGUFQuantizedMutationPlan(
            self.plan,
            (self.quantized,),
        )

    def chunks(self) -> Iterator[bytes]:
        axis0 = next((item for item in self.quantized.axis_edits if item.axis == 0), None)
        outer = next((item for item in self.quantized.axis_edits if item.axis == 1), None)
        if axis0 is None:
            if outer is None:
                raise NativeGGUFMLPExecutionError("changed tensor has no executable axis edit")
            yield from self._outer_copy(set(outer.removed_indices))
        elif axis0.strategy is QuantizedEditStrategy.DIRECT_BLOCK_COPY:
            yield from self._direct_axis0(axis0.removed_indices)
        elif axis0.strategy is QuantizedEditStrategy.REPACK_CONTIGUOUS_AXIS:
            yield from self._repack_axis0(axis0.removed_indices)
        else:
            raise NativeGGUFMLPExecutionError("unsupported contiguous-axis execution strategy")


def execute_native_gguf_mlp_channel_removal(
    source: MemoryMappedGGUF,
    plan: NativeGGUFMLPRemovalPlan,
    destination: str | Path,
    disk_plan: GGUFDiskPlan,
    codecs: CodecRegistry,
    *,
    limits: NativeGGUFMLPExecutionLimits | None = None,
) -> NativeGGUFMLPExecutionResult:
    """Stream a resumable, validated GGUF without constructing a float model copy."""

    execution_limits = limits or NativeGGUFMLPExecutionLimits()
    reader = GGUFTensorReader(source)
    physical_by_name = {
        edit.locator: edit for edit in plan.physical_plan.tensor_edits
    }
    quantized_by_component = {
        edit.component_id: edit for edit in plan.quantized_plan.tensor_edits
    }
    changed_sources: list[_ChangedTensorSource] = []
    seen_changed: set[str] = set()
    write_tensors: list[GGUFWriteTensor] = []
    unchanged_hashes: list[tuple[str, str]] = []
    for handle in reader.index.tensors:
        physical = physical_by_name.get(handle.name)
        if physical is None:
            copy = copy_unchanged_gguf_tensor(
                reader,
                handle,
                max_chunk_bytes=min(
                    execution_limits.copy_chunk_bytes, reader.limits.max_chunk_bytes
                ),
            )
            write_tensors.append(copy.as_write_tensor())
            unchanged_hashes.append(
                (
                    handle.name,
                    _tensor_sha256(reader, handle, copy.max_chunk_bytes),
                )
            )
            continue
        quantized = quantized_by_component[physical.component_id]
        if (
            physical.old_shape != handle.dimensions
            or quantized.old_shape != handle.dimensions
            or quantized.new_shape != physical.new_shape
            or quantized.quant_type is not handle.quant_type
            or quantized.encoded_size != physical.new_storage_bytes
        ):
            raise NativeGGUFMLPExecutionError(
                f"source tensor {handle.name!r} disagrees with the validated plan"
            )
        seen_changed.add(handle.name)
        changed = _ChangedTensorSource(
            reader,
            handle,
            plan.physical_plan,
            physical,
            quantized,
            codecs,
            execution_limits,
        )
        changed_sources.append(changed)
        descriptor = reader.descriptor(handle)
        write_tensors.append(
            GGUFWriteTensor(
                handle.name,
                physical.new_shape,
                descriptor.ggml_type_id,
                changed.chunks(),
            )
        )
    missing_changed = set(physical_by_name) - seen_changed
    if missing_changed:
        raise NativeGGUFMLPExecutionError(
            "planned tensors are absent from source: "
            + ", ".join(sorted(missing_changed))
        )
    input_digest = _file_sha256(source.container.path, execution_limits.copy_chunk_bytes)
    result = write_gguf_resumably(
        destination,
        _metadata(source, plan),
        tuple(write_tensors),
        disk_plan,
        input_sha256=input_digest,
        version=source.container.version,
        byte_order=source.container.byte_order,
        alignment=source.container.alignment,
    )
    output_hashes = dict(result.tensor_sha256)
    for name, digest in unchanged_hashes:
        if output_hashes.get(name) != digest:
            raise NativeGGUFMLPExecutionError(
                f"untouched tensor {name!r} is not byte-identical in output"
            )
    with open_gguf(result.path) as output:
        discovery = discover_gguf_components(output.container, family=plan.family)
    expected_width = cast("int", plan.physical_plan.metadata_updates[0].value)
    if discovery.shape.feed_forward_length != expected_width:
        raise NativeGGUFMLPExecutionError("output graph feed-forward width disagrees with plan")
    output_shapes = {
        tensor.descriptor.name: tensor.descriptor.dimensions
        for tensor in discovery.tensors
    }
    for edit in plan.physical_plan.tensor_edits:
        if output_shapes.get(edit.locator) != edit.new_shape:
            raise NativeGGUFMLPExecutionError(
                f"output tensor {edit.locator!r} shape disagrees with plan"
            )
    return NativeGGUFMLPExecutionResult(
        result,
        discovery,
        tuple(unchanged_hashes),
        tuple(error for source_item in changed_sources for error in source_item.errors),
        max((item.peak_row_working_bytes for item in changed_sources), default=0),
    )

"""Bounded transactional execution of native quantized GGUF head removal."""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

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
    QuantizationError,
    TensorRole,
    copy_unchanged_gguf_tensor,
    discover_gguf_components,
    open_gguf,
    write_gguf_resumably,
)
from modelsurgeon.surgery.attention_head_rules import (
    AttentionHeadEditStrategy,
    AttentionHeadTensorRule,
    NativeGGUFAttentionHeadRules,
)
from modelsurgeon.surgery.selective_requant import GGUFRequantizationLimits


class NativeGGUFAttentionHeadExecutionError(ValueError):
    """Raised before publishing output that diverges from validated head rules."""


@dataclass(frozen=True, slots=True)
class NativeGGUFAttentionHeadExecutionLimits:
    copy_chunk_bytes: int = 4 * 1024 * 1024
    max_row_working_bytes: int = 16 * 1024 * 1024
    requantization: GGUFRequantizationLimits = field(
        default_factory=GGUFRequantizationLimits
    )

    def __post_init__(self) -> None:
        if self.copy_chunk_bytes <= 0 or self.max_row_working_bytes <= 0:
            raise NativeGGUFAttentionHeadExecutionError(
                "attention-head execution limits must be positive"
            )


@dataclass(frozen=True, slots=True)
class AttentionHeadRequantizationError:
    tensor_name: str
    row: int
    error: QuantizationError


@dataclass(frozen=True, slots=True)
class NativeGGUFAttentionHeadExecutionResult:
    write_result: GGUFWriteResult
    output_discovery: GGUFDiscovery
    unchanged_tensor_sha256: tuple[tuple[str, str], ...]
    requantization_errors: tuple[AttentionHeadRequantizationError, ...]
    peak_row_working_bytes: int


def _file_sha256(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(chunk_bytes):
            digest.update(data)
    return digest.hexdigest()


def _tensor_sha256(
    reader: GGUFTensorReader,
    handle: GGUFTensorHandle,
    chunk_bytes: int,
) -> str:
    digest = hashlib.sha256()
    for chunk in reader.iter_chunks(handle, max_chunk_bytes=chunk_bytes):
        digest.update(chunk.data)
    return digest.hexdigest()


def _metadata(
    source: MemoryMappedGGUF,
    rules: NativeGGUFAttentionHeadRules,
) -> tuple[GGUFWriteMetadata, ...]:
    updates = dict(rules.metadata_updates)
    output: list[GGUFWriteMetadata] = []
    seen: set[str] = set()
    for entry in source.container.metadata:
        if entry.key not in updates:
            output.append(
                GGUFWriteMetadata(
                    entry.key,
                    entry.value_type,
                    entry.value,
                    entry.element_type,
                )
            )
            continue
        if entry.value_type not in {GGUFValueType.UINT32, GGUFValueType.UINT64}:
            raise NativeGGUFAttentionHeadExecutionError(
                f"head metadata {entry.key!r} is not an unsigned integer"
            )
        value = updates[entry.key]
        output.append(GGUFWriteMetadata(entry.key, entry.value_type, value))
        seen.add(entry.key)
    for key in sorted(set(updates) - seen):
        output.append(GGUFWriteMetadata(key, GGUFValueType.UINT32, updates[key]))
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


class _AttentionTensorSource:
    def __init__(
        self,
        reader: GGUFTensorReader,
        handle: GGUFTensorHandle,
        rule: AttentionHeadTensorRule,
        codecs: CodecRegistry,
        limits: NativeGGUFAttentionHeadExecutionLimits,
    ) -> None:
        self.reader = reader
        self.handle = handle
        self.rule = rule
        self.codecs = codecs
        self.limits = limits
        self.errors: list[AttentionHeadRequantizationError] = []
        self.peak_row_working_bytes = 0

    @property
    def max_blocks(self) -> int:
        blocks = min(
            self.reader.limits.max_chunk_bytes,
            self.limits.copy_chunk_bytes,
        ) // self.handle.encoded_block_bytes
        if blocks <= 0:
            raise NativeGGUFAttentionHeadExecutionError(
                f"copy limit cannot hold one {self.handle.quant_type.value} block"
            )
        return blocks

    def _outer_copy(self) -> Iterator[bytes]:
        removed = set(self.rule.removed_indices)
        blocks_per_row = self.handle.dimensions[0] // self.handle.logical_block_values
        rows = self.handle.block_count // blocks_per_row
        for row in range(rows):
            if row in removed:
                continue
            consumed = 0
            while consumed < blocks_per_row:
                count = min(self.max_blocks, blocks_per_row - consumed)
                yield self.reader.read_blocks(
                    self.handle,
                    row * blocks_per_row + consumed,
                    count,
                ).data
                consumed += count

    def _direct_axis0(self) -> Iterator[bytes]:
        block_values = self.handle.logical_block_values
        removed_blocks = {index // block_values for index in self.rule.removed_indices}
        blocks_per_row = self.handle.dimensions[0] // block_values
        rows = self.handle.block_count // blocks_per_row
        for row in range(rows):
            for offset, count in _kept_block_runs(blocks_per_row, removed_blocks):
                consumed = 0
                while consumed < count:
                    size = min(self.max_blocks, count - consumed)
                    yield self.reader.read_blocks(
                        self.handle,
                        row * blocks_per_row + offset + consumed,
                        size,
                    ).data
                    consumed += size

    def _repack_axis0(self) -> Iterator[bytes]:
        codec = self.codecs.resolve(self.rule.quant_type)
        old_width = self.rule.old_shape[0]
        new_width = self.rule.new_shape[0]
        blocks_per_row = old_width // codec.layout.block_size
        new_blocks = new_width // codec.layout.block_size
        rows = self.handle.block_count // blocks_per_row
        old_encoded = blocks_per_row * codec.layout.type_size
        requant = self.limits.requantization
        max_output_blocks = min(
            requant.max_encoded_chunk_bytes // codec.layout.type_size,
            requant.max_validation_values // codec.layout.block_size,
            requant.max_working_bytes
            // (2 * codec.layout.type_size + 8 * codec.layout.block_size),
        )
        if max_output_blocks <= 0:
            raise NativeGGUFAttentionHeadExecutionError(
                f"requantization limits cannot hold one {self.rule.quant_type.value} block"
            )
        chunk_blocks = min(new_blocks, max_output_blocks)
        chunk_values = chunk_blocks * codec.layout.block_size
        chunk_encoded = chunk_blocks * codec.layout.type_size
        required = (old_width + 2 * new_width + chunk_values) * 4 + old_encoded
        required += 2 * chunk_encoded + chunk_values * 8
        if required > self.limits.max_row_working_bytes:
            raise NativeGGUFAttentionHeadExecutionError(
                f"one attention row requires {required} working bytes, exceeding "
                f"limit {self.limits.max_row_working_bytes}"
            )
        self.peak_row_working_bytes = required
        removed = set(self.rule.removed_indices)
        for row in range(rows):
            decoded = array("f")
            consumed = 0
            while consumed < blocks_per_row:
                count = min(self.max_blocks, blocks_per_row - consumed)
                chunk = self.reader.read_blocks(
                    self.handle,
                    row * blocks_per_row + consumed,
                    count,
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
                raise NativeGGUFAttentionHeadExecutionError(
                    f"filtered row for {self.rule.tensor_name!r} disagrees with rules"
                )
            consumed_blocks = 0
            while consumed_blocks < new_blocks:
                count = min(max_output_blocks, new_blocks - consumed_blocks)
                start = consumed_blocks * codec.layout.block_size
                end = start + count * codec.layout.block_size
                values = retained[start:end]
                encoded = bytearray(count * codec.layout.type_size)
                operation = codec.encode_blocks(
                    values,
                    memoryview(encoded),
                    byte_order=self.reader.source.container.byte_order,
                )
                validation = codec.validate_blocks(
                    memoryview(encoded),
                    byte_order=self.reader.source.container.byte_order,
                )
                validation.require_valid()
                checked = array("d")
                codec.decode_blocks(
                    memoryview(encoded),
                    checked,
                    byte_order=self.reader.source.container.byte_order,
                )
                if operation.block_count != count or len(checked) != len(values):
                    raise NativeGGUFAttentionHeadExecutionError(
                        f"codec counts for {self.rule.tensor_name!r} disagree with rules"
                    )
                error = codec.estimate_error(values, checked)
                if (
                    requant.max_absolute_error is not None
                    and error.max_absolute_error > requant.max_absolute_error
                ):
                    raise NativeGGUFAttentionHeadExecutionError(
                        f"requantized max error {error.max_absolute_error} exceeds "
                        f"ceiling {requant.max_absolute_error}"
                    )
                if (
                    requant.max_mean_squared_error is not None
                    and error.mean_squared_error > requant.max_mean_squared_error
                ):
                    raise NativeGGUFAttentionHeadExecutionError(
                        f"requantized mean squared error {error.mean_squared_error} "
                        f"exceeds ceiling {requant.max_mean_squared_error}"
                    )
                self.errors.append(
                    AttentionHeadRequantizationError(
                        self.rule.tensor_name,
                        row,
                        error,
                    )
                )
                yield bytes(encoded)
                consumed_blocks += count

    def chunks(self) -> Iterator[bytes]:
        if self.rule.strategy is AttentionHeadEditStrategy.WHOLE_HEAD_SLICE_COPY:
            yield from self._outer_copy()
        elif self.rule.strategy is AttentionHeadEditStrategy.DIRECT_BLOCK_COPY:
            yield from self._direct_axis0()
        elif self.rule.strategy is AttentionHeadEditStrategy.REPACK_CONTIGUOUS_AXIS:
            yield from self._repack_axis0()
        else:
            raise NativeGGUFAttentionHeadExecutionError(
                f"changed tensor {self.rule.tensor_name!r} has no execution strategy"
            )


def execute_native_gguf_attention_head_removal(
    source: MemoryMappedGGUF,
    rules: NativeGGUFAttentionHeadRules,
    destination: str | Path,
    disk_plan: GGUFDiskPlan,
    codecs: CodecRegistry,
    *,
    limits: NativeGGUFAttentionHeadExecutionLimits | None = None,
) -> NativeGGUFAttentionHeadExecutionResult:
    """Apply validated model-wide Q/K/V/O rules without a float model copy."""

    execution_limits = limits or NativeGGUFAttentionHeadExecutionLimits()
    discovery = discover_gguf_components(source.container, family=rules.family)
    if (
        discovery.architecture != rules.architecture
        or discovery.shape.attention_heads != rules.old_query_heads
        or discovery.shape.kv_heads != rules.old_kv_heads
        or discovery.shape.key_length != rules.key_head_width
        or discovery.shape.value_length != rules.value_head_width
    ):
        raise NativeGGUFAttentionHeadExecutionError(
            "source attention geometry disagrees with validated rules"
        )
    expected_names = {
        item.descriptor.name
        for item in discovery.tensors
        if item.mapping.role
        in {
            TensorRole.ATTENTION_Q,
            TensorRole.ATTENTION_K,
            TensorRole.ATTENTION_V,
            TensorRole.ATTENTION_O,
        }
        and item.descriptor.name.endswith(".weight")
    }
    rule_names = [item.tensor_name for item in rules.tensor_rules]
    if (
        len(rule_names) != len(set(rule_names))
        or set(rule_names) != expected_names
        or rules.new_query_heads <= 0
        or rules.new_kv_heads <= 0
        or rules.new_query_heads % rules.new_kv_heads
    ):
        raise NativeGGUFAttentionHeadExecutionError(
            "attention rules do not exactly cover a valid model-wide Q/K/V/O closure"
        )
    reader = GGUFTensorReader(source)
    rule_by_name = {item.tensor_name: item for item in rules.tensor_rules if item.changed}
    if len(rule_by_name) != sum(item.changed for item in rules.tensor_rules):
        raise NativeGGUFAttentionHeadExecutionError("changed tensor rules are not unique")
    changed_sources: list[_AttentionTensorSource] = []
    unchanged_hashes: list[tuple[str, str]] = []
    write_tensors: list[GGUFWriteTensor] = []
    seen: set[str] = set()
    for handle in reader.index.tensors:
        rule = rule_by_name.get(handle.name)
        if rule is None:
            copy = copy_unchanged_gguf_tensor(
                reader,
                handle,
                max_chunk_bytes=min(
                    execution_limits.copy_chunk_bytes,
                    reader.limits.max_chunk_bytes,
                ),
            )
            write_tensors.append(copy.as_write_tensor())
            unchanged_hashes.append(
                (handle.name, _tensor_sha256(reader, handle, copy.max_chunk_bytes))
            )
            continue
        if handle.dimensions != rule.old_shape or handle.quant_type is not rule.quant_type:
            raise NativeGGUFAttentionHeadExecutionError(
                f"source tensor {handle.name!r} disagrees with validated rules"
            )
        seen.add(handle.name)
        changed = _AttentionTensorSource(
            reader,
            handle,
            rule,
            codecs,
            execution_limits,
        )
        changed_sources.append(changed)
        write_tensors.append(
            GGUFWriteTensor(
                handle.name,
                rule.new_shape,
                reader.descriptor(handle).ggml_type_id,
                changed.chunks(),
            )
        )
    missing = set(rule_by_name) - seen
    if missing:
        raise NativeGGUFAttentionHeadExecutionError(
            "planned tensors are absent from source: " + ", ".join(sorted(missing))
        )
    result = write_gguf_resumably(
        destination,
        _metadata(source, rules),
        tuple(write_tensors),
        disk_plan,
        input_sha256=_file_sha256(
            source.container.path,
            execution_limits.copy_chunk_bytes,
        ),
        version=source.container.version,
        byte_order=source.container.byte_order,
        alignment=source.container.alignment,
    )
    output_hashes = dict(result.tensor_sha256)
    for name, digest in unchanged_hashes:
        if output_hashes.get(name) != digest:
            raise NativeGGUFAttentionHeadExecutionError(
                f"untouched tensor {name!r} is not byte-identical in output"
            )
    with open_gguf(result.path) as output:
        output_discovery = discover_gguf_components(
            output.container,
            family=rules.family,
        )
    if (
        output_discovery.shape.attention_heads != rules.new_query_heads
        or output_discovery.shape.kv_heads != rules.new_kv_heads
        or output_discovery.shape.key_length != rules.key_head_width
        or output_discovery.shape.value_length != rules.value_head_width
    ):
        raise NativeGGUFAttentionHeadExecutionError(
            "output attention geometry disagrees with validated rules"
        )
    shapes = {
        item.descriptor.name: item.descriptor.dimensions
        for item in output_discovery.tensors
    }
    for rule in rules.tensor_rules:
        if shapes.get(rule.tensor_name) != rule.new_shape:
            raise NativeGGUFAttentionHeadExecutionError(
                f"output tensor {rule.tensor_name!r} shape disagrees with rules"
            )
    return NativeGGUFAttentionHeadExecutionResult(
        result,
        output_discovery,
        tuple(unchanged_hashes),
        tuple(error for item in changed_sources for error in item.errors),
        max((item.peak_row_working_bytes for item in changed_sources), default=0),
    )

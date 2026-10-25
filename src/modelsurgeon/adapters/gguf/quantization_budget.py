"""Exact-byte planning and bounded codec-loss estimates for GGUF tensors.

The estimator measures local round-trip error against the decoded source tensor;
the planner consumes those or other supplied per-tensor costs. Neither claims
that additive tensor loss predicts model quality, and neither publishes model
artifacts. Callers must still write, reload, and evaluate selected artifacts.

Its multi-choice Pareto dynamic program is an independent implementation
informed by RCO's discrete budget search ([Apache-2.0, revision
9a1e09c07d468109cbe60a1b87d5036034a79d10](https://github.com/IST-DASLab/RCO/tree/9a1e09c07d468109cbe60a1b87d5036034a79d10/src/search/quant.py));
no upstream code is copied. Unlike RCO's average-bit proxy, each transition
uses the exact GGUF encoded size and alignment cost.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import prod

from modelsurgeon.adapters.gguf.container import (
    GGML_TYPE_IDS,
    GGUFContainer,
    GGUFTensorDescriptor,
    gguf_tensor_byte_size,
)
from modelsurgeon.adapters.gguf.disk import GGUFDiskPlan
from modelsurgeon.adapters.gguf.quantization import (
    ByteOrder,
    CodecRegistry,
    GGMLQuantizationType,
    QuantizationCodec,
)
from modelsurgeon.adapters.gguf.tensor_reader import GGUFTensorHandle, GGUFTensorReader
from modelsurgeon.adapters.gguf.writer import (
    GGUFWriteMetadata,
    GGUFWriteResult,
    GGUFWriteTensor,
    plan_gguf_output,
    write_gguf_transactionally,
)


class GGUFQuantizationBudgetError(ValueError):
    """Raised when byte-budget planning inputs are invalid or infeasible."""


@dataclass(frozen=True, slots=True)
class GGUFQuantizationAlternative:
    quant_type: GGMLQuantizationType
    estimated_loss: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.quant_type, GGMLQuantizationType)
            or isinstance(self.estimated_loss, bool)
            or not math.isfinite(self.estimated_loss)
            or self.estimated_loss < 0
        ):
            raise GGUFQuantizationBudgetError(
                "quantization alternatives require a ggml type and finite non-negative loss"
            )


@dataclass(frozen=True, slots=True)
class GGUFQuantizationChoice:
    tensor_name: str
    source_quant_type: GGMLQuantizationType
    output_quant_type: GGMLQuantizationType
    encoded_bytes: int
    relative_offset: int
    estimated_loss: float


@dataclass(frozen=True, slots=True)
class GGUFQuantizationBudgetPlan:
    target_file_bytes: int
    planned_file_bytes: int
    estimated_loss: float
    choices: tuple[GGUFQuantizationChoice, ...]
    explored_states: int


@dataclass(frozen=True, slots=True)
class GGUFQuantizationLossEstimate:
    """Codec round-trip loss against the encoded source tensor's decoded values."""

    quant_type: GGMLQuantizationType
    sample_count: int
    mean_absolute_error: float
    mean_squared_error: float
    max_absolute_error: float
    estimated_peak_working_bytes: int
    activation_weighted_mean_squared_error: float | None = None

    def __post_init__(self) -> None:
        if (
            self.sample_count <= 0
            or self.estimated_peak_working_bytes <= 0
            or any(
                not math.isfinite(value) or value < 0
                for value in (
                    self.mean_absolute_error,
                    self.mean_squared_error,
                    self.max_absolute_error,
                )
            )
            or (
                self.activation_weighted_mean_squared_error is not None
                and (
                    not math.isfinite(self.activation_weighted_mean_squared_error)
                    or self.activation_weighted_mean_squared_error < 0
                )
            )
        ):
            raise GGUFQuantizationBudgetError(
                "codec loss estimates must be finite and non-negative"
            )

    def to_alternative(
        self, *, activation_weighted: bool = False
    ) -> GGUFQuantizationAlternative:
        """Convert a per-element MSE proxy to summed loss for additive ranking."""
        loss = self.mean_squared_error
        if activation_weighted:
            if self.activation_weighted_mean_squared_error is None:
                raise GGUFQuantizationBudgetError(
                    "activation-weighted loss was not requested for this estimate"
                )
            loss = self.activation_weighted_mean_squared_error
        return GGUFQuantizationAlternative(
            self.quant_type, loss * self.sample_count
        )


@dataclass(frozen=True, slots=True)
class _State:
    relative_end: int
    loss: float
    previous_index: int | None
    choice_index: int | None


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _ggml_type_id(quant_type: GGMLQuantizationType) -> int:
    for type_id, candidate in GGML_TYPE_IDS.items():
        if candidate is quant_type:
            return type_id
    raise GGUFQuantizationBudgetError(f"no GGML type ID exists for {quant_type.value}")


def estimate_tensor_quantization_losses(
    reader: GGUFTensorReader,
    handle: GGUFTensorHandle,
    candidate_types: Sequence[GGMLQuantizationType],
    codecs: CodecRegistry,
    *,
    byte_order: ByteOrder,
    input_activation_second_moments: Sequence[float] | None = None,
    input_axis: int | None = None,
    max_working_bytes: int = 16 * 1024 * 1024,
) -> tuple[GGUFQuantizationLossEstimate, ...]:
    """Measure bounded codec round-trip errors for one encoded GGUF tensor.

    Candidate losses compare each destination encoding with the decoded values
    of the source tensor. The source quantization error relative to original
    full-precision weights is unknown, and these local errors are not estimates
    of language-model quality. The output MSE is an optional planner proxy.
    When input activation second moments and a matrix input axis are supplied,
    the estimator also reports output-error MSE under a diagonal activation
    covariance approximation. Raw second moments retain scale differences
    between layers for additive planner costs.
    """
    if (
        isinstance(max_working_bytes, bool)
        or not isinstance(max_working_bytes, int)
        or max_working_bytes <= 0
    ):
        raise GGUFQuantizationBudgetError("working-memory limit must be a positive byte count")

    descriptor = reader.descriptor(handle)
    source_codec = codecs.resolve(handle.quant_type)
    activation_moments: Sequence[float] | None = None
    if (input_activation_second_moments is None) != (input_axis is None):
        raise GGUFQuantizationBudgetError(
            "activation moments and input axis must be supplied together"
        )
    if input_activation_second_moments is not None:
        if (
            len(descriptor.dimensions) != 2
            or isinstance(input_axis, bool)
            or input_axis not in (0, 1)
        ):
            raise GGUFQuantizationBudgetError(
                "activation-weighted estimates require a 2D tensor and input axis 0 or 1"
            )
        assert input_axis is not None
        expected_moments = descriptor.dimensions[input_axis]
        if len(input_activation_second_moments) != expected_moments:
            raise GGUFQuantizationBudgetError(
                "activation second-moment count must match the declared tensor input axis"
            )
        try:
            activation_sum = math.fsum(
                _activation_second_moment(value, index)
                for index, value in enumerate(input_activation_second_moments)
            )
        except (OverflowError, TypeError, ValueError) as error:
            raise GGUFQuantizationBudgetError(
                "activation second moments must be finite non-negative numbers"
            ) from error
        if not math.isfinite(activation_sum) or activation_sum <= 0:
            raise GGUFQuantizationBudgetError(
                "activation second moments must have a positive finite sum"
            )
        activation_moments = input_activation_second_moments
    elif input_axis is not None:
        raise GGUFQuantizationBudgetError("input axis requires activation second moments")
    requested: tuple[GGMLQuantizationType, ...]
    if not candidate_types:
        requested = (handle.quant_type,)
    else:
        if any(not isinstance(item, GGMLQuantizationType) for item in candidate_types):
            raise GGUFQuantizationBudgetError("candidate types must be ggml quantization types")
        if len(set(candidate_types)) != len(candidate_types):
            raise GGUFQuantizationBudgetError("candidate quantization types must be unique")
        requested = tuple(candidate_types)
        if handle.quant_type not in requested:
            requested = (*requested, handle.quant_type)
    quant_types = tuple(sorted(requested, key=lambda item: item.value))

    destination_codecs = {}
    for quant_type in quant_types:
        destination_codecs[quant_type] = (
            source_codec if quant_type is handle.quant_type else codecs.resolve(quant_type)
        )
        # Validate target block compatibility before reading any tensor payload.
        try:
            gguf_tensor_byte_size(quant_type, descriptor.dimensions)
        except ValueError as error:
            raise GGUFQuantizationBudgetError(
                f"tensor {handle.name!r} cannot use {quant_type.value}: {error}"
            ) from error

    element_count = prod(descriptor.dimensions)
    if element_count != handle.block_count * source_codec.layout.block_size:
        raise GGUFQuantizationBudgetError(
            f"tensor {handle.name!r} element count disagrees with its source codec blocks"
        )
    common_block_values = math.lcm(
        source_codec.layout.block_size,
        *(codec.layout.block_size for codec in destination_codecs.values()),
    )
    packed_bytes_per_value = source_codec.layout.type_size // source_codec.layout.block_size + max(
        codec.layout.type_size // codec.layout.block_size for codec in destination_codecs.values()
    )
    # Two decoded Python float lists can consume roughly 64 bytes per value;
    # packed source/destination data and codec scratch are budgeted separately.
    estimated_bytes_per_value = 64 + packed_bytes_per_value
    max_values_by_memory = max_working_bytes // estimated_bytes_per_value
    max_values_by_reader = (
        reader.limits.max_chunk_bytes // source_codec.layout.type_size
    ) * source_codec.layout.block_size
    chunk_values = min(max_values_by_memory, max_values_by_reader)
    chunk_values = chunk_values // common_block_values * common_block_values
    if chunk_values < common_block_values:
        raise GGUFQuantizationBudgetError(
            "working-memory limit cannot hold one common source/destination codec block"
        )
    estimated_peak_working_bytes = chunk_values * estimated_bytes_per_value

    squared_error: dict[GGMLQuantizationType, float] = dict.fromkeys(quant_types, 0.0)
    absolute_error: dict[GGMLQuantizationType, float] = dict.fromkeys(quant_types, 0.0)
    max_absolute_error: dict[GGMLQuantizationType, float] = dict.fromkeys(quant_types, 0.0)
    activation_weighted_error: dict[GGMLQuantizationType, float] = dict.fromkeys(
        quant_types, 0.0
    )
    compensation: dict[GGMLQuantizationType, float] = dict.fromkeys(quant_types, 0.0)
    activation_compensation: dict[GGMLQuantizationType, float] = dict.fromkeys(
        quant_types, 0.0
    )
    input_stride = (
        prod(descriptor.dimensions[:input_axis])
        if input_axis is not None
        else 1
    )
    input_width = (
        descriptor.dimensions[input_axis]
        if input_axis is not None
        else 1
    )
    block_count_per_chunk = chunk_values // source_codec.layout.block_size
    processed = 0
    block_offset = 0
    while block_offset < handle.block_count:
        blocks = min(block_count_per_chunk, handle.block_count - block_offset)
        # The final tail must remain aligned for every destination codec.
        values_in_chunk = blocks * source_codec.layout.block_size
        if values_in_chunk % common_block_values:
            raise GGUFQuantizationBudgetError(
                "source tensor tail is not aligned to every destination codec block"
            )
        chunk = reader.read_blocks(handle, block_offset, blocks)
        source_values: list[float] = []
        source_operation = source_codec.decode_blocks(
            memoryview(chunk.data), source_values, byte_order=byte_order
        )
        if source_operation.element_count != values_in_chunk:
            raise GGUFQuantizationBudgetError("source codec decoded an unexpected element count")
        for quant_type, codec in destination_codecs.items():
            if quant_type is handle.quant_type:
                continue
            encoded_size = codec.layout.encoded_size(values_in_chunk)
            encoded = bytearray(encoded_size)
            encode_operation = codec.encode_blocks(
                source_values, memoryview(encoded), byte_order=byte_order
            )
            if encode_operation.element_count != values_in_chunk:
                raise GGUFQuantizationBudgetError(
                    f"{quant_type.value} encoder processed an unexpected element count"
                )
            reconstructed: list[float] = []
            decode_operation = codec.decode_blocks(
                memoryview(encoded), reconstructed, byte_order=byte_order
            )
            if decode_operation.element_count != values_in_chunk:
                raise GGUFQuantizationBudgetError(
                    f"{quant_type.value} decoder produced an unexpected element count"
                )
            loss_metrics = codec.estimate_error(source_values, reconstructed)
            term = loss_metrics.mean_squared_error * loss_metrics.sample_count
            adjusted = term - compensation[quant_type]
            updated = squared_error[quant_type] + adjusted
            compensation[quant_type] = (updated - squared_error[quant_type]) - adjusted
            squared_error[quant_type] = updated
            absolute_error[quant_type] += (
                loss_metrics.mean_absolute_error * loss_metrics.sample_count
            )
            max_absolute_error[quant_type] = max(
                max_absolute_error[quant_type], loss_metrics.max_absolute_error
            )
            if activation_moments is not None:
                weighted_terms = (
                    (source_value - reconstructed_value) ** 2
                    * float(
                        activation_moments[
                            ((block_offset * source_codec.layout.block_size + index)
                             // input_stride)
                            % input_width
                        ]
                    )
                    for index, (source_value, reconstructed_value) in enumerate(
                        zip(source_values, reconstructed, strict=True)
                    )
                )
                weighted_term = math.fsum(weighted_terms)
                if not math.isfinite(weighted_term):
                    raise GGUFQuantizationBudgetError(
                        "activation-weighted codec loss overflowed"
                    )
                adjusted_weighted = (
                    weighted_term - activation_compensation[quant_type]
                )
                updated_weighted = activation_weighted_error[quant_type] + adjusted_weighted
                activation_compensation[quant_type] = (
                    updated_weighted - activation_weighted_error[quant_type]
                ) - adjusted_weighted
                activation_weighted_error[quant_type] = updated_weighted
        processed += values_in_chunk
        block_offset += blocks

    if processed != element_count:
        raise GGUFQuantizationBudgetError("codec estimator did not cover the complete tensor")
    return tuple(
        GGUFQuantizationLossEstimate(
            quant_type,
            element_count,
            absolute_error[quant_type] / element_count,
            squared_error[quant_type] / element_count,
            max_absolute_error[quant_type],
            estimated_peak_working_bytes,
            (
                activation_weighted_error[quant_type] / element_count
                if activation_moments is not None
                else None
            ),
        )
        for quant_type in quant_types
    )


def _activation_second_moment(value: float, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GGUFQuantizationBudgetError(
            f"activation second moment at index {index} must be numeric"
        )
    moment = float(value)
    if not math.isfinite(moment) or moment < 0:
        raise GGUFQuantizationBudgetError(
            f"activation second moment at index {index} must be finite and non-negative"
        )
    return moment


def execute_gguf_quantization_plan(
    reader: GGUFTensorReader,
    plan: GGUFQuantizationBudgetPlan,
    codecs: CodecRegistry,
    disk_plan: GGUFDiskPlan,
    *,
    max_working_bytes: int = 16 * 1024 * 1024,
) -> GGUFWriteResult:
    """Stream a validated per-tensor plan into an atomic GGUF artifact.

    Unchanged tensors are copied byte-for-byte. Changed tensors are decoded and
    re-encoded in common source/destination codec block chunks. The source stays
    read-only, and the writer stages, reparses, then publishes the new file.
    This function does not evaluate model quality or authorize promotion.
    """
    if (
        isinstance(max_working_bytes, bool)
        or not isinstance(max_working_bytes, int)
        or max_working_bytes <= 0
    ):
        raise GGUFQuantizationBudgetError("working-memory limit must be a positive byte count")
    container = reader.source.container
    if len(plan.choices) != len(container.tensors):
        raise GGUFQuantizationBudgetError("quantization plan does not cover every source tensor")

    metadata = tuple(
        GGUFWriteMetadata(
            entry.key,
            entry.value_type,
            entry.value,
            entry.element_type,
        )
        for entry in container.metadata
    )
    writer_tensors: list[GGUFWriteTensor] = []
    for tensor, choice in zip(container.tensors, plan.choices, strict=True):
        if choice.tensor_name != tensor.name or choice.source_quant_type is not tensor.quant_type:
            raise GGUFQuantizationBudgetError(
                "quantization plan tensor identity or source type does not match this GGUF"
            )
        handle = reader.index.tensor(tensor.name)
        unchanged = choice.output_quant_type is tensor.quant_type
        source_codec = None if unchanged else codecs.resolve(tensor.quant_type)
        target_codec = None if unchanged else codecs.resolve(choice.output_quant_type)
        try:
            planned_bytes = gguf_tensor_byte_size(choice.output_quant_type, tensor.dimensions)
        except ValueError as error:
            raise GGUFQuantizationBudgetError(
                f"tensor {tensor.name!r} cannot use {choice.output_quant_type.value}: {error}"
            ) from error
        if choice.encoded_bytes != planned_bytes:
            raise GGUFQuantizationBudgetError(
                f"plan byte count for {tensor.name!r} does not match its codec layout"
            )
        type_id = _ggml_type_id(choice.output_quant_type)
        writer_tensors.append(
            GGUFWriteTensor(
                tensor.name,
                tensor.dimensions,
                type_id,
                _encoded_tensor_chunks(
                    reader,
                    handle,
                    source_codec,
                    target_codec,
                    unchanged=unchanged,
                    byte_order=container.byte_order,
                    max_working_bytes=max_working_bytes,
                ),
            )
        )

    output_layout = plan_gguf_output(
        metadata,
        tuple(writer_tensors),
        version=container.version,
        byte_order=container.byte_order,
        alignment=container.alignment,
    )
    if output_layout.total_bytes != plan.planned_file_bytes:
        raise GGUFQuantizationBudgetError(
            "plan file size disagrees with the GGUF writer's exact output layout"
        )
    if output_layout.total_bytes > plan.target_file_bytes:
        raise GGUFQuantizationBudgetError("planned GGUF exceeds its hard file-size ceiling")
    for choice, output_tensor in zip(plan.choices, output_layout.tensors, strict=True):
        if (
            choice.relative_offset != output_tensor.relative_offset
            or choice.encoded_bytes != output_tensor.byte_size
        ):
            raise GGUFQuantizationBudgetError(
                f"planned offset or encoded size differs for tensor {choice.tensor_name!r}"
            )

    result = write_gguf_transactionally(
        disk_plan.output_path,
        metadata,
        tuple(writer_tensors),
        disk_plan,
        version=container.version,
        byte_order=container.byte_order,
        alignment=container.alignment,
    )
    return result


def _encoded_tensor_chunks(
    reader: GGUFTensorReader,
    handle: GGUFTensorHandle,
    source_codec: QuantizationCodec | None,
    target_codec: QuantizationCodec | None,
    *,
    unchanged: bool,
    byte_order: ByteOrder,
    max_working_bytes: int,
) -> Iterator[bytes]:
    """Create a lazy byte stream; changed chunks align to both codec block grids."""
    if unchanged:
        return (chunk.data for chunk in reader.iter_chunks(handle))

    if source_codec is None or target_codec is None:
        raise GGUFQuantizationBudgetError(
            "changed tensors require exact source and destination codecs"
        )

    source_layout = source_codec.layout
    target_layout = target_codec.layout
    common_block_values = math.lcm(source_layout.block_size, target_layout.block_size)
    packed_bytes_per_value = (
        source_layout.type_size // source_layout.block_size
        + target_layout.type_size // target_layout.block_size
    )
    estimated_bytes_per_value = 64 + packed_bytes_per_value
    max_values = max_working_bytes // estimated_bytes_per_value
    max_reader_values = (
        reader.limits.max_chunk_bytes // source_layout.type_size
    ) * source_layout.block_size
    chunk_values = min(max_values, max_reader_values)
    chunk_values = chunk_values // common_block_values * common_block_values
    if chunk_values < common_block_values:
        raise GGUFQuantizationBudgetError(
            "working-memory limit cannot hold one common source/destination codec block"
        )
    source_blocks_per_chunk = chunk_values // source_layout.block_size

    def chunks() -> Iterator[bytes]:
        block_offset = 0
        while block_offset < handle.block_count:
            block_count = min(
                source_blocks_per_chunk,
                handle.block_count - block_offset,
            )
            value_count = block_count * source_layout.block_size
            if value_count % common_block_values:
                raise GGUFQuantizationBudgetError(
                    "source tensor tail is not aligned to the destination codec block"
                )
            source_chunk = reader.read_blocks(handle, block_offset, block_count)
            values: list[float] = []
            decoded = source_codec.decode_blocks(
                memoryview(source_chunk.data), values, byte_order=byte_order
            )
            if decoded.element_count != value_count:
                raise GGUFQuantizationBudgetError(
                    "source decoder returned an unexpected value count"
                )
            encoded = bytearray(target_layout.encoded_size(value_count))
            operation = target_codec.encode_blocks(
                values, memoryview(encoded), byte_order=byte_order
            )
            if operation.element_count != value_count:
                raise GGUFQuantizationBudgetError(
                    "destination encoder returned an unexpected value count"
                )
            yield bytes(encoded)
            block_offset += block_count

    return chunks()


def _tensor_alternatives(
    tensor: GGUFTensorDescriptor,
    proposed: Sequence[GGUFQuantizationAlternative],
    codecs: CodecRegistry,
) -> tuple[tuple[GGMLQuantizationType, int, float], ...]:
    """Validate choices and return (type, exact bytes, loss) in stable order."""
    by_type: dict[GGMLQuantizationType, GGUFQuantizationAlternative] = {
        tensor.quant_type: GGUFQuantizationAlternative(tensor.quant_type, 0.0)
    }
    for option in proposed:
        if option.quant_type in by_type:
            if option.quant_type is tensor.quant_type:
                # Keeping source bytes is lossless and does not require a writer.
                continue
            raise GGUFQuantizationBudgetError(
                f"tensor {tensor.name!r} repeats quantization option {option.quant_type.value}"
            )
        if option.quant_type not in GGML_TYPE_IDS.values():
            raise GGUFQuantizationBudgetError(
                f"tensor {tensor.name!r} has no GGML type ID for {option.quant_type.value}"
            )
        try:
            codecs.resolve(option.quant_type)
            size = gguf_tensor_byte_size(option.quant_type, tensor.dimensions)
        except ValueError as error:
            raise GGUFQuantizationBudgetError(
                f"tensor {tensor.name!r} cannot use {option.quant_type.value}: {error}"
            ) from error
        by_type[option.quant_type] = option

    result = []
    for quant_type, option in sorted(by_type.items(), key=lambda item: item[0].value):
        if quant_type is tensor.quant_type:
            size = tensor.byte_size
        else:
            size = gguf_tensor_byte_size(quant_type, tensor.dimensions)
        result.append((quant_type, size, option.estimated_loss))
    return tuple(result)


def plan_gguf_quantization_budget(
    container: GGUFContainer,
    alternatives: Mapping[str, Sequence[GGUFQuantizationAlternative]],
    codecs: CodecRegistry,
    *,
    target_file_bytes: int,
    max_states: int = 250_000,
    max_frontier_states: int = 50_000,
) -> GGUFQuantizationBudgetPlan:
    """Choose per-tensor codecs with minimum supplied loss under an exact file cap.

    Costs include the GGUF data offset, tensor ordering, alignment padding, and
    encoded block sizes. Metadata and tensor descriptors retain their source
    lengths, so changing a ggml type does not change the header's byte length.
    The Pareto dynamic program is bounded by ``max_states`` and
    ``max_frontier_states``; it fails closed instead of dropping exactness when
    either limit is exceeded.

    A tensor omitted from ``alternatives`` is copied in its source codec. A
    source-codec option is always included with zero loss. Every changed codec
    must have an exact writer registered in ``codecs``.
    """
    if (
        isinstance(target_file_bytes, bool)
        or not isinstance(target_file_bytes, int)
        or target_file_bytes <= 0
    ):
        raise GGUFQuantizationBudgetError("target file size must be a positive byte count")
    if (
        isinstance(max_states, bool)
        or isinstance(max_frontier_states, bool)
        or not isinstance(max_states, int)
        or not isinstance(max_frontier_states, int)
        or max_states <= 0
        or max_frontier_states <= 0
    ):
        raise GGUFQuantizationBudgetError("planner state limits must be positive")
    if not container.tensors:
        raise GGUFQuantizationBudgetError("cannot plan a GGUF with no tensors")

    tensor_names = {tensor.name for tensor in container.tensors}
    unknown_names = sorted(set(alternatives) - tensor_names)
    if unknown_names:
        raise GGUFQuantizationBudgetError(
            f"alternatives reference unknown tensors: {', '.join(unknown_names[:5])}"
        )

    options = tuple(
        _tensor_alternatives(tensor, alternatives.get(tensor.name, ()), codecs)
        for tensor in container.tensors
    )
    writer_metadata = tuple(
        GGUFWriteMetadata(entry.key, entry.value_type, entry.value, entry.element_type)
        for entry in container.metadata
    )
    source_writer_tensors = tuple(
        GGUFWriteTensor(
            tensor.name,
            tensor.dimensions,
            _ggml_type_id(tensor.quant_type),
            (),
        )
        for tensor in container.tensors
    )
    output_data_offset = plan_gguf_output(
        writer_metadata,
        source_writer_tensors,
        version=container.version,
        byte_order=container.byte_order,
        alignment=container.alignment,
    ).data_offset

    # Each layer keeps nondominated (encoded end offset, estimated loss) states.
    # Parent indexes allow exact reconstruction while retaining deterministic ties.
    history: list[tuple[_State, ...]] = [(_State(0, 0.0, None, None),)]
    explored = 1
    for tensor_index, tensor_options in enumerate(options):
        previous = history[-1]
        next_by_end: dict[int, _State] = {}
        for previous_index, state in enumerate(previous):
            tensor_start = _align(state.relative_end, container.alignment)
            for choice_index, (_, size, loss) in enumerate(tensor_options):
                explored += 1
                if explored > max_states:
                    raise GGUFQuantizationBudgetError(
                        f"planner exceeded its {max_states} state budget"
                    )
                relative_end = tensor_start + size
                if output_data_offset + relative_end > target_file_bytes:
                    continue
                combined_loss = state.loss + loss
                if not math.isfinite(combined_loss):
                    raise GGUFQuantizationBudgetError("combined estimated loss overflowed")
                candidate = _State(relative_end, combined_loss, previous_index, choice_index)
                current = next_by_end.get(relative_end)
                if current is None or (combined_loss, choice_index, previous_index) < (
                    current.loss,
                    current.choice_index if current.choice_index is not None else -1,
                    current.previous_index if current.previous_index is not None else -1,
                ):
                    next_by_end[relative_end] = candidate
                if len(next_by_end) > max_frontier_states * 4:
                    raise GGUFQuantizationBudgetError(
                        "planner exceeded its bounded intermediate frontier"
                    )

        frontier: list[_State] = []
        best_loss = math.inf
        for _, state in sorted(next_by_end.items()):
            if state.loss < best_loss:
                frontier.append(state)
                best_loss = state.loss
        if len(frontier) > max_frontier_states:
            raise GGUFQuantizationBudgetError(
                f"planner exceeded its {max_frontier_states}-state Pareto frontier"
            )
        if not frontier:
            raise GGUFQuantizationBudgetError(
                f"no quantization assignment fits {target_file_bytes} bytes "
                f"after tensor {container.tensors[tensor_index].name!r}"
            )
        history.append(tuple(frontier))

    final_index, final_state = min(
        enumerate(history[-1]),
        key=lambda item: (item[1].loss, item[1].relative_end, item[0]),
    )
    selected: list[tuple[GGMLQuantizationType, int, float, int]] = []
    state_index = final_index
    for tensor_index in range(len(container.tensors) - 1, -1, -1):
        state = history[tensor_index + 1][state_index]
        assert state.previous_index is not None and state.choice_index is not None
        quant_type, size, loss = options[tensor_index][state.choice_index]
        tensor_start = _align(
            history[tensor_index][state.previous_index].relative_end, container.alignment
        )
        selected.append((quant_type, size, loss, tensor_start))
        state_index = state.previous_index
    selected.reverse()

    choices = tuple(
        GGUFQuantizationChoice(
            tensor.name,
            tensor.quant_type,
            quant_type,
            size,
            relative_offset,
            loss,
        )
        for tensor, (quant_type, size, loss, relative_offset) in zip(
            container.tensors, selected, strict=True
        )
    )
    return GGUFQuantizationBudgetPlan(
        target_file_bytes,
        output_data_offset + final_state.relative_end,
        final_state.loss,
        choices,
        explored,
    )


__all__ = [
    "GGUFQuantizationAlternative",
    "GGUFQuantizationBudgetError",
    "GGUFQuantizationBudgetPlan",
    "GGUFQuantizationChoice",
    "plan_gguf_quantization_budget",
]

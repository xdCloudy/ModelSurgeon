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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod

from modelsurgeon.adapters.gguf.container import (
    GGML_TYPE_IDS,
    GGUFContainer,
    GGUFTensorDescriptor,
    gguf_tensor_byte_size,
)
from modelsurgeon.adapters.gguf.quantization import (
    ByteOrder,
    CodecRegistry,
    GGMLQuantizationType,
)
from modelsurgeon.adapters.gguf.tensor_reader import GGUFTensorHandle, GGUFTensorReader


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
        ):
            raise GGUFQuantizationBudgetError(
                "codec loss estimates must be finite and non-negative"
            )

    def to_alternative(self) -> GGUFQuantizationAlternative:
        """Convert tensor MSE to summed squared error for global additive ranking."""
        return GGUFQuantizationAlternative(
            self.quant_type, self.mean_squared_error * self.sample_count
        )


@dataclass(frozen=True, slots=True)
class _State:
    relative_end: int
    loss: float
    previous_index: int | None
    choice_index: int | None


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def estimate_tensor_quantization_losses(
    reader: GGUFTensorReader,
    handle: GGUFTensorHandle,
    candidate_types: Sequence[GGMLQuantizationType],
    codecs: CodecRegistry,
    *,
    byte_order: ByteOrder,
    max_working_bytes: int = 16 * 1024 * 1024,
) -> tuple[GGUFQuantizationLossEstimate, ...]:
    """Measure bounded codec round-trip errors for one encoded GGUF tensor.

    Candidate losses compare each destination encoding with the decoded values
    of the source tensor. The source quantization error relative to original
    full-precision weights is unknown, and these local errors are not estimates
    of language-model quality. The output MSE is an optional planner proxy.
    """
    if (
        isinstance(max_working_bytes, bool)
        or not isinstance(max_working_bytes, int)
        or max_working_bytes <= 0
    ):
        raise GGUFQuantizationBudgetError("working-memory limit must be a positive byte count")

    descriptor = reader.descriptor(handle)
    source_codec = codecs.resolve(handle.quant_type)
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
    compensation: dict[GGMLQuantizationType, float] = dict.fromkeys(quant_types, 0.0)
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
        )
        for quant_type in quant_types
    )


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
                if container.data_offset + relative_end > target_file_bytes:
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
        container.data_offset + final_state.relative_end,
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

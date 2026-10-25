"""Exact-byte planning for per-tensor GGUF quantization assignments.

The planner consumes externally measured or estimated per-tensor costs. It does
not quantize tensors or claim that an additive tensor loss predicts model
quality; callers must still write, reload, and evaluate selected artifacts.

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

from modelsurgeon.adapters.gguf.container import (
    GGML_TYPE_IDS,
    GGUFContainer,
    GGUFTensorDescriptor,
    gguf_tensor_byte_size,
)
from modelsurgeon.adapters.gguf.quantization import CodecRegistry, GGMLQuantizationType


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
class _State:
    relative_end: int
    loss: float
    previous_index: int | None
    choice_index: int | None


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


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

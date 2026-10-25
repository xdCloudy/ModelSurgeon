from __future__ import annotations

import itertools
import random
from pathlib import Path

import pytest

from modelsurgeon.adapters.gguf.container import (
    GGML_TYPE_IDS,
    GGUFContainer,
    GGUFTensorDescriptor,
    GGUFValueType,
)
from modelsurgeon.adapters.gguf.q4_k_codec import Q4_K_CODEC
from modelsurgeon.adapters.gguf.q5_k_codec import Q5_K_CODEC
from modelsurgeon.adapters.gguf.q6_k_codec import Q6_K_CODEC
from modelsurgeon.adapters.gguf.quantization import (
    ByteOrder,
    CodecRegistry,
    GGMLQuantizationType,
)
from modelsurgeon.adapters.gguf.quantization_budget import (
    GGUFQuantizationAlternative,
    GGUFQuantizationBudgetError,
    plan_gguf_quantization_budget,
)
from modelsurgeon.adapters.gguf.writer import (
    GGUFWriteMetadata,
    GGUFWriteTensor,
    plan_gguf_output,
)

_METADATA = (GGUFWriteMetadata("general.architecture", GGUFValueType.STRING, "test"),)


def _type_id(quant_type: GGMLQuantizationType) -> int:
    return next(type_id for type_id, candidate in GGML_TYPE_IDS.items() if candidate is quant_type)


def _container() -> GGUFContainer:
    names = ("blk.0.weight", "blk.1.weight", "output.weight")
    writer_tensors = tuple(
        GGUFWriteTensor(name, (256, 2), _type_id(GGMLQuantizationType.Q8_0), ()) for name in names
    )
    output_layout = plan_gguf_output(_METADATA, writer_tensors, alignment=32)
    tensors = tuple(
        GGUFTensorDescriptor(
            name=name,
            dimensions=(256, 2),
            ggml_type_id=_type_id(GGMLQuantizationType.Q8_0),
            quant_type=GGMLQuantizationType.Q8_0,
            relative_offset=output_layout.tensors[index].relative_offset,
            data_offset=output_layout.tensors[index].data_offset,
            byte_size=544,
            descriptor_offset=index * 64,
            descriptor_size=64,
        )
        for index, name in enumerate(names)
    )
    return GGUFContainer(
        Path("fixture.gguf"),
        output_layout.total_bytes,
        3,
        ByteOrder.LITTLE,
        32,
        output_layout.data_offset,
        (),
        tensors,
    )


def _codecs() -> CodecRegistry:
    result = CodecRegistry()
    for codec in (Q4_K_CODEC, Q5_K_CODEC, Q6_K_CODEC):
        result.register(codec)
    return result


def _alternatives() -> dict[str, tuple[GGUFQuantizationAlternative, ...]]:
    costs = ((8.0, 4.0, 3.0), (7.0, 3.0, 2.0), (6.0, 1.0, 0.5))
    types = (
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
    )
    return {
        tensor.name: tuple(
            GGUFQuantizationAlternative(quant_type, loss)
            for quant_type, loss in zip(types, losses, strict=True)
        )
        for tensor, losses in zip(_container().tensors, costs, strict=True)
    }


def _file_size(container: GGUFContainer, assignment: tuple[GGMLQuantizationType, ...]) -> int:
    output_layout = plan_gguf_output(
        _METADATA,
        tuple(
            GGUFWriteTensor(tensor.name, tensor.dimensions, _type_id(quant_type), ())
            for tensor, quant_type in zip(container.tensors, assignment, strict=True)
        ),
        alignment=container.alignment,
    )
    assert output_layout.data_offset == container.data_offset
    return output_layout.total_bytes


def _loss(assignment: tuple[GGMLQuantizationType, ...]) -> float:
    known = {
        "blk.0.weight": {
            GGMLQuantizationType.Q4_K: 8.0,
            GGMLQuantizationType.Q5_K: 4.0,
            GGMLQuantizationType.Q6_K: 3.0,
        },
        "blk.1.weight": {
            GGMLQuantizationType.Q4_K: 7.0,
            GGMLQuantizationType.Q5_K: 3.0,
            GGMLQuantizationType.Q6_K: 2.0,
        },
        "output.weight": {
            GGMLQuantizationType.Q4_K: 6.0,
            GGMLQuantizationType.Q5_K: 1.0,
            GGMLQuantizationType.Q6_K: 0.5,
        },
    }
    return sum(
        known[tensor.name][quant_type]
        for tensor, quant_type in zip(_container().tensors, assignment, strict=True)
    )


def test_planner_minimizes_loss_with_exact_aligned_file_costs() -> None:
    container = _container()
    target = container.data_offset + 928
    plan = plan_gguf_quantization_budget(
        container,
        _alternatives(),
        _codecs(),
        target_file_bytes=target,
    )

    assignment = tuple(choice.output_quant_type for choice in plan.choices)
    choices = tuple(
        item
        for item in (
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K,
        )
    )
    feasible = (
        (candidate, _loss(candidate))
        for candidate in itertools.product(choices, repeat=len(container.tensors))
        if _file_size(container, candidate) <= target
    )
    expected = min(feasible, key=lambda item: (item[1], _file_size(container, item[0])))

    assert assignment == expected[0]
    assert assignment == (
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
    )
    assert plan.estimated_loss == expected[1]
    assert plan.planned_file_bytes == _file_size(container, assignment) == target
    assert tuple(choice.relative_offset for choice in plan.choices) == (0, 288, 576)
    assert tuple(choice.encoded_bytes for choice in plan.choices) == (288, 288, 352)


def test_planner_matches_exhaustive_oracle_for_seeded_small_assignments() -> None:
    container = _container()
    options = (
        GGMLQuantizationType.Q4_K,
        GGMLQuantizationType.Q5_K,
        GGMLQuantizationType.Q6_K,
        GGMLQuantizationType.Q8_0,
    )
    for seed in range(40):
        rng = random.Random(seed)
        losses = {
            tensor.name: {
                quant_type: (0.0 if quant_type is GGMLQuantizationType.Q8_0 else rng.random() * 10)
                for quant_type in options
            }
            for tensor in container.tensors
        }
        alternatives = {
            tensor.name: tuple(
                GGUFQuantizationAlternative(quant_type, losses[tensor.name][quant_type])
                for quant_type in options[:-1]
            )
            for tensor in container.tensors
        }
        target = rng.randint(container.data_offset + 864, container.file_size)
        plan = plan_gguf_quantization_budget(
            container, alternatives, _codecs(), target_file_bytes=target
        )
        feasible = (
            (
                sum(
                    losses[tensor.name][quant]
                    for tensor, quant in zip(container.tensors, assignment, strict=True)
                ),
                _file_size(container, assignment),
                assignment,
            )
            for assignment in itertools.product(options, repeat=len(container.tensors))
            if _file_size(container, assignment) <= target
        )
        expected_loss, expected_bytes, expected_assignment = min(feasible)
        actual_assignment = tuple(choice.output_quant_type for choice in plan.choices)
        assert actual_assignment == expected_assignment
        assert plan.estimated_loss == pytest.approx(expected_loss)
        assert plan.planned_file_bytes == expected_bytes


def test_planner_includes_source_type_and_rejects_infeasible_byte_cap() -> None:
    container = _container()
    plan = plan_gguf_quantization_budget(
        container, {}, _codecs(), target_file_bytes=container.file_size
    )
    assert all(choice.output_quant_type is GGMLQuantizationType.Q8_0 for choice in plan.choices)
    assert plan.planned_file_bytes == container.file_size
    assert plan.estimated_loss == 0

    with pytest.raises(GGUFQuantizationBudgetError, match="no quantization assignment fits"):
        plan_gguf_quantization_budget(container, _alternatives(), _codecs(), target_file_bytes=900)


def test_planner_rejects_unknown_tensor_and_unavailable_writer() -> None:
    container = _container()
    unavailable = {"blk.0.weight": (GGUFQuantizationAlternative(GGMLQuantizationType.IQ4_XS, 0.1),)}
    with pytest.raises(GGUFQuantizationBudgetError, match="cannot use IQ4_XS"):
        plan_gguf_quantization_budget(
            container, unavailable, _codecs(), target_file_bytes=container.file_size
        )
    with pytest.raises(GGUFQuantizationBudgetError, match="unknown tensors"):
        plan_gguf_quantization_budget(
            container,
            {"not.a.tensor": ()},
            _codecs(),
            target_file_bytes=container.file_size,
        )


def test_planner_fails_closed_when_state_budget_is_exceeded() -> None:
    container = _container()
    with pytest.raises(GGUFQuantizationBudgetError, match="state budget"):
        plan_gguf_quantization_budget(
            container,
            _alternatives(),
            _codecs(),
            target_file_bytes=container.file_size,
            max_states=1,
        )


def test_planner_requires_integer_limits_and_finite_nonnegative_losses() -> None:
    container = _container()
    with pytest.raises(GGUFQuantizationBudgetError, match="positive byte count"):
        plan_gguf_quantization_budget(
            container, {}, _codecs(), target_file_bytes=container.file_size + 0.5
        )
    with pytest.raises(GGUFQuantizationBudgetError, match="state limits"):
        plan_gguf_quantization_budget(
            container,
            {},
            _codecs(),
            target_file_bytes=container.file_size,
            max_states=10.5,
        )
    with pytest.raises(GGUFQuantizationBudgetError, match="finite non-negative loss"):
        GGUFQuantizationAlternative(GGMLQuantizationType.Q5_K, float("nan"))

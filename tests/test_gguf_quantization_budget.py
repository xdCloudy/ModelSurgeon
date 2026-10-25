from __future__ import annotations

import hashlib
import itertools
import math
import random
from pathlib import Path

import pytest

from modelsurgeon.adapters.gguf.container import (
    GGML_TYPE_IDS,
    GGUFContainer,
    GGUFMetadataEntry,
    GGUFTensorDescriptor,
    GGUFValueType,
    open_gguf,
)
from modelsurgeon.adapters.gguf.disk import GGUFDiskEstimate, preflight_gguf_disk
from modelsurgeon.adapters.gguf.q4_k_codec import Q4_K_CODEC
from modelsurgeon.adapters.gguf.q5_k_codec import Q5_K_CODEC
from modelsurgeon.adapters.gguf.q6_k_codec import Q6_K_CODEC
from modelsurgeon.adapters.gguf.q8_codec import Q8_0_CODEC
from modelsurgeon.adapters.gguf.quantization import (
    ByteOrder,
    CodecRegistry,
    GGMLQuantizationType,
)
from modelsurgeon.adapters.gguf.quantization_budget import (
    GGUFQuantizationAlternative,
    GGUFQuantizationBudgetError,
    estimate_tensor_quantization_losses,
    execute_gguf_quantization_plan,
    plan_gguf_quantization_budget,
)
from modelsurgeon.adapters.gguf.tensor_reader import GGUFTensorReader
from modelsurgeon.adapters.gguf.writer import (
    GGUFWriteMetadata,
    GGUFWriteTensor,
    plan_gguf_output,
    write_gguf_transactionally,
)

_METADATA = (
    GGUFWriteMetadata("general.architecture", GGUFValueType.STRING, "test"),
    GGUFWriteMetadata("general.quantization_version", GGUFValueType.UINT32, 2),
    GGUFWriteMetadata("general.alignment", GGUFValueType.UINT32, 32),
)


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
        tuple(
            GGUFMetadataEntry(
                item.key,
                item.value_type,
                item.value,
                0,
                0,
                item.element_type,
            )
            for item in _METADATA
        ),
        tensors,
    )


def _codecs() -> CodecRegistry:
    result = CodecRegistry()
    for codec in (Q4_K_CODEC, Q5_K_CODEC, Q6_K_CODEC):
        result.register(codec)
    return result


def _estimator_codecs() -> CodecRegistry:
    result = _codecs()
    result.register(Q8_0_CODEC)
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


def test_streaming_codec_estimates_feed_the_exact_byte_planner(tmp_path: Path) -> None:
    rng = random.Random(91)
    values = [rng.uniform(-1.0, 1.0) for _ in range(512)]
    encoded_source = bytearray(Q8_0_CODEC.layout.encoded_size(len(values)))
    Q8_0_CODEC.encode_blocks(values, memoryview(encoded_source), byte_order=ByteOrder.LITTLE)
    tensor = GGUFWriteTensor(
        "tensor.weight",
        (256, 2),
        _type_id(GGMLQuantizationType.Q8_0),
        (bytes(encoded_source),),
    )
    untouched_source = bytearray(Q8_0_CODEC.layout.encoded_size(len(values)))
    Q8_0_CODEC.encode_blocks(
        tuple(reversed(values)), memoryview(untouched_source), byte_order=ByteOrder.LITTLE
    )
    untouched_tensor = GGUFWriteTensor(
        "untouched.weight",
        (256, 2),
        _type_id(GGMLQuantizationType.Q8_0),
        (bytes(untouched_source),),
    )
    source_tensors = (tensor, untouched_tensor)
    output_layout = plan_gguf_output(_METADATA, source_tensors, alignment=32)
    source_path = tmp_path / "source.gguf"
    disk = preflight_gguf_disk(
        source_path,
        tmp_path,
        GGUFDiskEstimate(output_layout.total_bytes, 0, alignment_bytes=output_layout.alignment),
    )
    write_gguf_transactionally(source_path, _METADATA, source_tensors, disk)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    with open_gguf(source_path) as mapped:
        reader = GGUFTensorReader(mapped)
        estimates = estimate_tensor_quantization_losses(
            reader,
            reader.index.tensor("tensor.weight"),
            (
                GGMLQuantizationType.Q4_K,
                GGMLQuantizationType.Q5_K,
                GGMLQuantizationType.Q6_K,
            ),
            _estimator_codecs(),
            byte_order=ByteOrder.LITTLE,
            max_working_bytes=32 * 1024,
        )
        by_type = {estimate.quant_type: estimate for estimate in estimates}
        assert set(by_type) == {
            GGMLQuantizationType.Q4_K,
            GGMLQuantizationType.Q5_K,
            GGMLQuantizationType.Q6_K,
            GGMLQuantizationType.Q8_0,
        }
        assert all(estimate.sample_count == 512 for estimate in estimates)
        assert by_type[GGMLQuantizationType.Q8_0].mean_squared_error == 0
        assert all(estimate.estimated_peak_working_bytes <= 32 * 1024 for estimate in estimates)
        assert (
            by_type[GGMLQuantizationType.Q5_K].mean_squared_error
            < by_type[GGMLQuantizationType.Q4_K].mean_squared_error
        )

        activation_moments = tuple(float(index + 1) for index in range(256))
        activation_weighted = estimate_tensor_quantization_losses(
            reader,
            reader.index.tensor("tensor.weight"),
            (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K),
            _estimator_codecs(),
            byte_order=ByteOrder.LITTLE,
            input_activation_second_moments=activation_moments,
            input_axis=0,
            max_working_bytes=32 * 1024,
        )
        weighted_by_type = {estimate.quant_type: estimate for estimate in activation_weighted}
        assert all(
            estimate.activation_weighted_mean_squared_error is not None
            for estimate in activation_weighted
        )
        assert (
            weighted_by_type[GGMLQuantizationType.Q8_0]
            .activation_weighted_mean_squared_error
            == 0
        )
        source_values: list[float] = []
        source_handle = reader.index.tensor("tensor.weight")
        source_chunk = reader.read_blocks(source_handle, 0, source_handle.block_count)
        Q8_0_CODEC.decode_blocks(
            memoryview(source_chunk.data), source_values, byte_order=ByteOrder.LITTLE
        )
        q4_encoded = bytearray(Q4_K_CODEC.layout.encoded_size(len(source_values)))
        Q4_K_CODEC.encode_blocks(
            source_values, memoryview(q4_encoded), byte_order=ByteOrder.LITTLE
        )
        q4_reconstructed: list[float] = []
        Q4_K_CODEC.decode_blocks(
            memoryview(q4_encoded), q4_reconstructed, byte_order=ByteOrder.LITTLE
        )
        expected_weighted_mse = math.fsum(
            (source - reconstructed) ** 2 * activation_moments[index % 256]
            for index, (source, reconstructed) in enumerate(
                zip(source_values, q4_reconstructed, strict=True)
            )
        ) / len(source_values)
        q4_weighted = weighted_by_type[GGMLQuantizationType.Q4_K]
        assert q4_weighted.activation_weighted_mean_squared_error == pytest.approx(
            expected_weighted_mse
        )
        assert q4_weighted.to_alternative(activation_weighted=True).estimated_loss == pytest.approx(
            expected_weighted_mse * len(source_values)
        )
        basis_forward_error = math.fsum(
            (
                (source_values[input_index + 256 * output_index]
                 - q4_reconstructed[input_index + 256 * output_index])
                * math.sqrt(256 * activation_moments[input_index])
            )
            ** 2
            for output_index in range(2)
            for input_index in range(256)
        ) / 256
        assert q4_weighted.to_alternative(activation_weighted=True).estimated_loss == pytest.approx(
            basis_forward_error
        )
        axis1_moments = (0.5, 2.0)
        axis1_estimates = estimate_tensor_quantization_losses(
            reader,
            source_handle,
            (GGMLQuantizationType.Q4_K,),
            _estimator_codecs(),
            byte_order=ByteOrder.LITTLE,
            input_activation_second_moments=axis1_moments,
            input_axis=1,
            max_working_bytes=32 * 1024,
        )
        expected_axis1_mse = math.fsum(
            (source - reconstructed) ** 2 * axis1_moments[(index // 256) % 2]
            for index, (source, reconstructed) in enumerate(
                zip(source_values, q4_reconstructed, strict=True)
            )
        ) / len(source_values)
        assert axis1_estimates[0].activation_weighted_mean_squared_error == pytest.approx(
            expected_axis1_mse
        )
        with pytest.raises(GGUFQuantizationBudgetError, match="activation-weighted loss"):
            by_type[GGMLQuantizationType.Q4_K].to_alternative(activation_weighted=True)
        with pytest.raises(GGUFQuantizationBudgetError, match="match the declared"):
            estimate_tensor_quantization_losses(
                reader,
                source_handle,
                (GGMLQuantizationType.Q4_K,),
                _estimator_codecs(),
                byte_order=ByteOrder.LITTLE,
                input_activation_second_moments=(1.0, 2.0),
                input_axis=0,
            )
        with pytest.raises(GGUFQuantizationBudgetError, match="positive finite sum"):
            estimate_tensor_quantization_losses(
                reader,
                source_handle,
                (GGMLQuantizationType.Q4_K,),
                _estimator_codecs(),
                byte_order=ByteOrder.LITTLE,
                input_activation_second_moments=(0.0,) * 256,
                input_axis=0,
            )

        alternatives = {
            "tensor.weight": tuple(
                estimate.to_alternative()
                for estimate in estimates
                if estimate.quant_type is not GGMLQuantizationType.Q8_0
            )
        }
        target_bytes = mapped.container.data_offset + 352
        target_bytes += 544
        plan = plan_gguf_quantization_budget(
            mapped.container,
            alternatives,
            _estimator_codecs(),
            target_file_bytes=target_bytes,
        )
        assert plan.choices[0].output_quant_type is GGMLQuantizationType.Q5_K
        assert plan.planned_file_bytes == target_bytes
        assert plan.choices[1].output_quant_type is GGMLQuantizationType.Q8_0

        candidate_path = tmp_path / "candidate.gguf"
        candidate_layout = plan_gguf_output(
            _METADATA,
            (
                GGUFWriteTensor("tensor.weight", (256, 2), _type_id(GGMLQuantizationType.Q5_K), ()),
                GGUFWriteTensor(
                    "untouched.weight", (256, 2), _type_id(GGMLQuantizationType.Q8_0), ()
                ),
            ),
            alignment=32,
        )
        candidate_disk = preflight_gguf_disk(
            candidate_path,
            tmp_path,
            GGUFDiskEstimate(
                candidate_layout.total_bytes,
                0,
                alignment_bytes=candidate_layout.alignment,
            ),
        )
        result = execute_gguf_quantization_plan(
            reader,
            plan,
            _estimator_codecs(),
            candidate_disk,
            max_working_bytes=32 * 1024,
        )
        assert result.file_size == target_bytes
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
        with open_gguf(candidate_path) as candidate:
            assert (
                candidate.container.tensor("tensor.weight").quant_type is GGMLQuantizationType.Q5_K
            )
            assert (
                candidate.container.tensor("untouched.weight").quant_type
                is GGMLQuantizationType.Q8_0
            )
            assert mapped.raw_bytes(
                mapped.container.tensor("untouched.weight").data_offset, 544
            ) == candidate.raw_bytes(
                candidate.container.tensor("untouched.weight").data_offset, 544
            )

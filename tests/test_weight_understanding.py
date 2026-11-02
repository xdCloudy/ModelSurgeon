"""Tests for the controlled synthetic weight-understanding protocol fixture."""

from __future__ import annotations

import numpy as np
import pytest

from modelsurgeon.evaluation.weight_understanding import (
    SmallReluNetwork,
    WeightUnderstandingStudyError,
    _identify_by_functional_probes,
    run_weight_understanding_study,
)


def test_synthetic_task_and_causal_probes_are_invariant_to_hidden_symmetries() -> None:
    result = run_weight_understanding_study(seed=17, repetitions=24)

    assert result.examples == 4 * 24
    assert result.task_accuracy == 1.0
    assert result.max_function_error < 1e-12
    assert result.max_causal_effect_error < 1e-12
    assert result.task_names == ("and", "or", "xor", "copy_first")


def test_study_record_labels_its_evidence_scope_and_is_reproducible() -> None:
    first = run_weight_understanding_study(seed=123, repetitions=8)
    second = run_weight_understanding_study(seed=123, repetitions=8)

    assert first == second
    record = first.to_record()
    assert record["status"] == "fixture_protocol_only"
    assert record["metrics"] == {
        "functional_probe_task_accuracy": 1.0,
        "maximum_function_error_after_symmetry": first.max_function_error,
        "maximum_causal_effect_error_after_symmetry": first.max_causal_effect_error,
    }
    assert "does_not_establish_understanding_of_pretrained_weights" in record["non_claims"]


@pytest.mark.parametrize("seed", [-1, 1 << 32, True, 1.5])
def test_study_rejects_invalid_seeds(seed: object) -> None:
    with pytest.raises(WeightUnderstandingStudyError, match="seed"):
        run_weight_understanding_study(seed=seed, repetitions=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("repetitions", [0, 257, True, 2.5])
def test_study_rejects_unbounded_or_invalid_repetitions(repetitions: object) -> None:
    with pytest.raises(WeightUnderstandingStudyError, match="repetitions"):
        run_weight_understanding_study(seed=1, repetitions=repetitions)  # type: ignore[arg-type]


def test_network_copies_and_freezes_weight_arrays() -> None:
    input_weight = np.asarray(((1.0,), (0.0,)))
    network = SmallReluNetwork(
        input_weight=input_weight,
        input_bias=np.asarray((0.0,)),
        output_weight=np.asarray((1.0,)),
    )
    input_weight[0, 0] = 100.0

    assert network.forward(np.asarray(((1.0, 0.0),))).tolist() == [1.0]
    with pytest.raises(ValueError):
        network.input_weight[0, 0] = 2.0


@pytest.mark.parametrize(
    "input_weight,input_bias,output_weight",
    [
        (np.asarray((1.0,)), np.asarray((0.0,)), np.asarray((1.0,))),
        (np.asarray(((1.0,),)), np.asarray((0.0, 0.0)), np.asarray((1.0,))),
        (np.asarray(((np.inf,),)), np.asarray((0.0,)), np.asarray((1.0,))),
    ],
)
def test_network_rejects_invalid_tensor_shapes_and_values(
    input_weight: np.ndarray, input_bias: np.ndarray, output_weight: np.ndarray
) -> None:
    with pytest.raises(WeightUnderstandingStudyError):
        SmallReluNetwork(input_weight, input_bias, output_weight)


def test_network_rejects_wrong_input_shape_and_hidden_index() -> None:
    network = SmallReluNetwork(np.asarray(((1.0,), (0.0,))), np.asarray((0.0,)), np.asarray((1.0,)))

    with pytest.raises(WeightUnderstandingStudyError, match="input width"):
        network.forward(np.asarray(((1.0,),)))
    with pytest.raises(WeightUnderstandingStudyError, match="index"):
        network.without_hidden_unit(1)


def test_functional_probe_classifier_abstains_on_unknown_output() -> None:
    network = SmallReluNetwork(np.asarray(((0.0,), (0.0,))), np.asarray((0.5,)), np.asarray((1.0,)))

    with pytest.raises(WeightUnderstandingStudyError, match="supported task"):
        _identify_by_functional_probes(
            network,
            {
                "zero": np.asarray((0.0, 0.0, 0.0, 0.0)),
                "one": np.asarray((1.0, 1.0, 1.0, 1.0)),
            },
        )

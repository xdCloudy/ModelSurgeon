"""Controlled synthetic probes for native weight and circuit understanding research.

This module validates a small, exact ground-truth fixture and its independent
functional probes. It does not train a Meta-Surgeon or establish model
understanding; it provides a deterministic protocol foundation for those studies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

WEIGHT_UNDERSTANDING_STUDY_VERSION: Final[str] = "1"
MAX_TRANSFORM_REPETITIONS: Final[int] = 256
_INPUTS: Final[NDArray[np.float64]] = np.asarray(
    ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)), dtype=np.float64
)
_TASK_OUTPUTS: Final[dict[str, NDArray[np.float64]]] = {
    "and": np.asarray((0.0, 0.0, 0.0, 1.0)),
    "or": np.asarray((0.0, 1.0, 1.0, 1.0)),
    "xor": np.asarray((0.0, 1.0, 1.0, 0.0)),
    "copy_first": np.asarray((0.0, 0.0, 1.0, 1.0)),
}


class WeightUnderstandingStudyError(ValueError):
    """Raised when a synthetic weight-understanding study is malformed."""


def _frozen_array(value: NDArray[np.float64], *, name: str) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.size == 0 or not np.isfinite(result).all():
        raise WeightUnderstandingStudyError(f"{name} must contain finite values")
    if np.max(np.abs(result)) > 1e6:
        raise WeightUnderstandingStudyError(f"{name} exceeds the supported value bound")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SmallReluNetwork:
    """A bounded one-hidden-layer scalar ReLU network with explicit dense weights."""

    input_weight: NDArray[np.float64]
    input_bias: NDArray[np.float64]
    output_weight: NDArray[np.float64]
    output_bias: float = 0.0

    def __post_init__(self) -> None:
        input_weight = _frozen_array(self.input_weight, name="input_weight")
        input_bias = _frozen_array(self.input_bias, name="input_bias")
        output_weight = _frozen_array(self.output_weight, name="output_weight")
        if input_weight.ndim != 2 or input_bias.ndim != 1 or output_weight.ndim != 1:
            raise WeightUnderstandingStudyError("network weights have invalid ranks")
        if input_weight.shape[1] != len(input_bias) or len(input_bias) != len(output_weight):
            raise WeightUnderstandingStudyError("network hidden dimensions do not match")
        if not math.isfinite(self.output_bias) or abs(self.output_bias) > 1e6:
            raise WeightUnderstandingStudyError("output_bias must be finite and bounded")
        object.__setattr__(self, "input_weight", input_weight)
        object.__setattr__(self, "input_bias", input_bias)
        object.__setattr__(self, "output_weight", output_weight)

    @property
    def hidden_units(self) -> int:
        return len(self.input_bias)

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the network for a two-dimensional batch of input rows."""

        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.input_weight.shape[0]:
            raise WeightUnderstandingStudyError("inputs do not match the network input width")
        if not np.isfinite(values).all():
            raise WeightUnderstandingStudyError("inputs must contain finite values")
        hidden = np.maximum(values @ self.input_weight + self.input_bias, 0.0)
        return hidden @ self.output_weight + self.output_bias

    def without_hidden_unit(self, index: int) -> SmallReluNetwork:
        """Return an immutable copy with one hidden unit's outgoing edge removed."""

        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < self.hidden_units
        ):
            raise WeightUnderstandingStudyError("hidden-unit index is out of range")
        output_weight = np.array(self.output_weight, copy=True)
        output_weight[index] = 0.0
        return SmallReluNetwork(
            self.input_weight,
            self.input_bias,
            output_weight,
            self.output_bias,
        )


@dataclass(frozen=True, slots=True)
class WeightUnderstandingStudyResult:
    """Deterministic fixture verification results, not a learned-model score."""

    seed: int
    repetitions: int
    examples: int
    task_accuracy: float
    max_function_error: float
    max_causal_effect_error: float
    task_names: tuple[str, ...]
    version: str = WEIGHT_UNDERSTANDING_STUDY_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "study": "synthetic_weight_function_and_circuit_probes",
            "status": "fixture_protocol_only",
            "seed": self.seed,
            "repetitions_per_task": self.repetitions,
            "example_count": self.examples,
            "task_names": list(self.task_names),
            "metrics": {
                "functional_probe_task_accuracy": self.task_accuracy,
                "maximum_function_error_after_symmetry": self.max_function_error,
                "maximum_causal_effect_error_after_symmetry": self.max_causal_effect_error,
            },
            "non_claims": [
                "does_not_train_or_evaluate_a_meta_surgeon",
                "does_not_use_or_validate_teacher_annotations",
                "does_not_establish_understanding_of_pretrained_weights",
                "does_not_establish_optimization_effectiveness",
            ],
        }


def _task_networks() -> dict[str, SmallReluNetwork]:
    """Return analytic circuits with unique truth tables on the binary input domain."""

    return {
        "and": SmallReluNetwork(
            np.asarray(((1.0,), (1.0,))), np.asarray((-1.0,)), np.asarray((1.0,))
        ),
        "or": SmallReluNetwork(
            np.asarray(((1.0, 1.0), (1.0, 1.0))),
            np.asarray((0.0, -1.0)),
            np.asarray((1.0, -1.0)),
        ),
        "xor": SmallReluNetwork(
            np.asarray(((1.0, -1.0), (-1.0, 1.0))),
            np.asarray((0.0, 0.0)),
            np.asarray((1.0, 1.0)),
        ),
        "copy_first": SmallReluNetwork(
            np.asarray(((1.0,), (0.0,))), np.asarray((0.0,)), np.asarray((1.0,))
        ),
    }


def _symmetry_transform(
    network: SmallReluNetwork, rng: np.random.Generator
) -> tuple[SmallReluNetwork, tuple[int, ...]]:
    """Apply hidden-unit permutation and positive ReLU rescaling exactly."""

    permutation = rng.permutation(network.hidden_units)
    scale = rng.uniform(0.125, 8.0, size=network.hidden_units)
    input_weight = network.input_weight[:, permutation] * scale
    input_bias = network.input_bias[permutation] * scale
    output_weight = network.output_weight[permutation] / scale
    transformed = SmallReluNetwork(input_weight, input_bias, output_weight, network.output_bias)
    return transformed, tuple(int(index) for index in permutation)


def _identify_by_functional_probes(
    network: SmallReluNetwork,
    task_outputs: dict[str, NDArray[np.float64]],
) -> str:
    """Select the known task fingerprint nearest to exhaustive black-box outputs."""

    observed = network.forward(_INPUTS)
    errors = {
        name: float(np.mean((observed - expected) ** 2)) for name, expected in task_outputs.items()
    }
    closest = min(errors, key=errors.__getitem__)
    if errors[closest] > 1e-20:
        raise WeightUnderstandingStudyError("functional probes do not match a supported task")
    return closest


def _causal_effects(network: SmallReluNetwork) -> NDArray[np.float64]:
    baseline = network.forward(_INPUTS)
    return np.asarray(
        [
            np.sqrt(np.mean((baseline - network.without_hidden_unit(index).forward(_INPUTS)) ** 2))
            for index in range(network.hidden_units)
        ],
        dtype=np.float64,
    )


def run_weight_understanding_study(
    *, seed: int = 42, repetitions: int = 32
) -> WeightUnderstandingStudyResult:
    """Verify exact task and causal probes under function-preserving weight symmetries.

    The exhaustive binary truth table is the independent functional measurement. The
    circuit target is the output change produced by removing each hidden unit's outgoing
    edge. The study checks that these labels survive hidden-unit permutations and
    positive ReLU rescalings. It is intentionally a synthetic protocol fixture.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 1 << 32:
        raise WeightUnderstandingStudyError("seed must be an unsigned 32-bit integer")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or not 1 <= repetitions <= MAX_TRANSFORM_REPETITIONS
    ):
        raise WeightUnderstandingStudyError(
            f"repetitions must be within [1, {MAX_TRANSFORM_REPETITIONS}]"
        )

    tasks = _task_networks()
    rng = np.random.default_rng(seed)
    correct = 0
    examples = 0
    max_function_error = 0.0
    max_causal_effect_error = 0.0
    for name, network in tasks.items():
        expected_outputs = _TASK_OUTPUTS[name]
        baseline_outputs = network.forward(_INPUTS)
        if not np.allclose(baseline_outputs, expected_outputs, rtol=0.0, atol=1e-12):
            raise WeightUnderstandingStudyError(f"analytic {name} circuit failed its truth table")
        baseline_effects = _causal_effects(network)
        for _ in range(repetitions):
            transformed, new_to_old = _symmetry_transform(network, rng)
            transformed_outputs = transformed.forward(_INPUTS)
            max_function_error = max(
                max_function_error,
                float(np.max(np.abs(transformed_outputs - baseline_outputs))),
            )
            correct += _identify_by_functional_probes(transformed, _TASK_OUTPUTS) == name
            examples += 1
            transformed_effects = _causal_effects(transformed)
            for new_index, old_index in enumerate(new_to_old):
                max_causal_effect_error = max(
                    max_causal_effect_error,
                    abs(float(transformed_effects[new_index] - baseline_effects[old_index])),
                )

    return WeightUnderstandingStudyResult(
        seed=seed,
        repetitions=repetitions,
        examples=examples,
        task_accuracy=correct / examples,
        max_function_error=max_function_error,
        max_causal_effect_error=max_causal_effect_error,
        task_names=tuple(tasks),
    )

"""Bounded feasibility study for globally coordinated hidden-dimension surgery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import AxisSemantic, GGUFDiscovery, TensorRole
from modelsurgeon.adapters.gguf.quantization import QUANT_LAYOUTS

HIDDEN_DIMENSION_STUDY_SCHEMA_VERSION: Final[int] = 1
_STUDIED_FAMILIES = frozenset({ModelFamily.LLAMA, ModelFamily.QWEN})
_GLOBAL_AXES = frozenset(
    {
        AxisSemantic.INPUT_FEATURE,
        AxisSemantic.OUTPUT_FEATURE,
        AxisSemantic.HIDDEN_FEATURE,
    }
)
_NORM_ROLES = frozenset(
    {TensorRole.INPUT_NORM, TensorRole.POST_ATTENTION_NORM, TensorRole.OUTPUT_NORM}
)


class HiddenDimensionStudyError(ValueError):
    """Raised when the bounded two-family study inputs are incomplete or unsupported."""


@dataclass(frozen=True, slots=True)
class HiddenDimensionFamilyAssessment:
    family: ModelFamily
    architecture: str
    hidden_dimension: int
    attention_heads: int
    kv_heads: int
    key_head_dimension: int
    value_head_dimension: int
    globally_coupled_tensors: tuple[str, ...]
    normalization_tensors: tuple[str, ...]
    quantized_axis0_granularity: int
    feasible_operation_classes: tuple[str, ...]
    rejection_reasons: tuple[str, ...]

    @property
    def physically_feasible(self) -> bool:
        return bool(self.feasible_operation_classes) and not self.rejection_reasons


@dataclass(frozen=True, slots=True)
class HiddenDimensionStudy:
    assessments: tuple[HiddenDimensionFamilyAssessment, ...]
    physical_mutation_implemented: bool = False
    schema_version: int = HIDDEN_DIMENSION_STUDY_SCHEMA_VERSION

    @property
    def physically_feasible(self) -> bool:
        return all(item.physically_feasible for item in self.assessments)


def evaluate_coordinated_hidden_dimension_surgery(
    discoveries: tuple[GGUFDiscovery, ...],
    *,
    max_tensors_per_family: int = 100_000,
) -> HiddenDimensionStudy:
    """Map proven consumers for Llama and Qwen and reject unproven physical mutation."""

    if max_tensors_per_family <= 0:
        raise HiddenDimensionStudyError("hidden-dimension study tensor limit must be positive")
    families = tuple(item.family for item in discoveries)
    if len(discoveries) != 2 or len(set(families)) != 2 or set(families) != _STUDIED_FAMILIES:
        raise HiddenDimensionStudyError("study requires exactly one Llama and one Qwen discovery")
    assessments: list[HiddenDimensionFamilyAssessment] = []
    for discovery in sorted(discoveries, key=lambda item: item.family.value):
        if len(discovery.tensors) > max_tensors_per_family:
            raise HiddenDimensionStudyError(
                f"{discovery.family.value} tensor count exceeds bounded study limit"
            )
        coupled: list[str] = []
        norms: list[str] = []
        granularities: list[int] = [1]
        for tensor in discovery.tensors:
            relevant_axes = tuple(
                axis for axis in tensor.mapping.axes if axis.semantic in _GLOBAL_AXES
            )
            if relevant_axes:
                coupled.append(tensor.descriptor.name)
            if tensor.mapping.role in _NORM_ROLES:
                norms.append(tensor.descriptor.name)
            for axis in relevant_axes:
                if axis.index == 0:
                    granularities.append(QUANT_LAYOUTS[tensor.descriptor.quant_type].block_size)
        if not coupled or not norms:
            raise HiddenDimensionStudyError(
                f"{discovery.family.value} discovery lacks global or normalization consumers"
            )
        reasons = (
            "rotary-dimension and rotary-frequency configuration consumers are not represented "
            "in the physical component graph",
            "token-embedding/output-weight tying provenance is unavailable from GGUF tensor "
            "descriptors",
            "tokenizer and runtime configuration consumers of embedding width are outside the "
            "validated physical plan",
        )
        assessments.append(
            HiddenDimensionFamilyAssessment(
                discovery.family,
                discovery.architecture,
                discovery.shape.embedding_length,
                discovery.shape.attention_heads,
                discovery.shape.kv_heads,
                discovery.shape.key_length,
                discovery.shape.value_length,
                tuple(sorted(coupled)),
                tuple(sorted(norms)),
                math.lcm(*granularities),
                (),
                reasons,
            )
        )
    return HiddenDimensionStudy(tuple(assessments))

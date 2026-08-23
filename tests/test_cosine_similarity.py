"""Tests for candidate-restricted blockwise cosine similarities."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from modelsurgeon.features.cosine_similarity import (
    CosineCandidate,
    CosineCandidateMode,
    CosineSimilarityConfig,
    CosineSimilarityError,
    CosineVector,
    CosineVectorKind,
    extract_pairwise_cosine_similarities,
    generate_cosine_candidates,
)
from modelsurgeon.graph import ComponentId


class TrackingSequence(Sequence[float]):
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.max_slice = 0

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int | slice) -> float | list[float]:
        if isinstance(index, slice):
            selected = self.values[index]
            self.max_slice = max(self.max_slice, len(selected))
            return selected
        return self.values[index]


def _ids() -> tuple[ComponentId, ComponentId, ComponentId]:
    return (
        ComponentId.parse("model.layers.0.mlp.down_proj.weight"),
        ComponentId.parse("model.layers.0.mlp.gate_proj.weight"),
        ComponentId.parse("model.layers.0.mlp.up_proj.weight"),
    )


def test_explicit_tensor_cosine_matches_reference_and_respects_block_size() -> None:
    first_id, second_id, _ = _ids()
    first_values = TrackingSequence([1.0, 0.0, 1.0, 0.0, 1.0])
    second_values = TrackingSequence([1.0, 1.0, 1.0, 1.0, 1.0])
    vectors = {
        first_id: CosineVector(first_id, first_values, CosineVectorKind.TENSOR, "float16"),
        second_id: CosineVector(second_id, second_values, CosineVectorKind.TENSOR, "float32"),
    }
    config = CosineSimilarityConfig(
        candidate_mode=CosineCandidateMode.EXPLICIT,
        block_size=2,
        max_candidates=4,
    )

    result = extract_pairwise_cosine_similarities(
        vectors,
        config,
        explicit_candidates=(CosineCandidate(second_id, first_id),),
    )

    assert len(result) == 1
    similarity = result[0]
    assert similarity.value == pytest.approx(3.0 / math.sqrt(15.0))
    assert similarity.left == first_id
    assert similarity.right == second_id
    assert similarity.left_kind is CosineVectorKind.TENSOR
    assert similarity.zero_vector is False
    assert first_values.max_slice <= 2
    assert second_values.max_slice <= 2
    assert similarity.feature_record().value == pytest.approx(similarity.value)
    assert similarity.to_record()["zero_vector_policy"] == "return_zero_and_flag"


def test_output_cosine_and_zero_vector_policy_are_explicit() -> None:
    first_id, second_id, _ = _ids()
    vectors = {
        first_id: CosineVector(first_id, [0.0, 0.0], CosineVectorKind.OUTPUT),
        second_id: CosineVector(second_id, [5.0, -2.0], CosineVectorKind.OUTPUT),
    }

    result = extract_pairwise_cosine_similarities(
        vectors,
        CosineSimilarityConfig(candidate_mode=CosineCandidateMode.ADJACENT, block_size=1),
    )

    assert result[0].value == 0.0
    assert result[0].zero_vector is True
    assert result[0].left_kind is CosineVectorKind.OUTPUT
    assert result[0].right_kind is CosineVectorKind.OUTPUT


def test_adjacent_and_explicit_candidate_generation_are_deterministic_and_restricted() -> None:
    first_id, second_id, third_id = _ids()
    config = CosineSimilarityConfig(candidate_mode=CosineCandidateMode.ADJACENT)

    adjacent = tuple(generate_cosine_candidates((third_id, first_id, second_id), config))

    assert adjacent == (
        CosineCandidate(first_id, second_id),
        CosineCandidate(second_id, third_id),
    )

    explicit_config = CosineSimilarityConfig(candidate_mode=CosineCandidateMode.EXPLICIT)
    explicit = tuple(
        generate_cosine_candidates(
            (first_id, second_id, third_id),
            explicit_config,
            explicit_candidates=(
                CosineCandidate(third_id, first_id),
                CosineCandidate(first_id, third_id),
            ),
        )
    )
    assert explicit == (CosineCandidate(first_id, third_id),)


def test_all_pairs_declines_before_generating_more_than_candidate_ceiling() -> None:
    first_id, second_id, third_id = _ids()
    config = CosineSimilarityConfig(
        candidate_mode=CosineCandidateMode.ALL,
        max_candidates=2,
    )

    with pytest.raises(CosineSimilarityError, match="all-pairs candidate count"):
        tuple(generate_cosine_candidates((first_id, second_id, third_id), config))


def test_mismatched_lengths_unknown_candidates_and_self_pairs_fail_closed() -> None:
    first_id, second_id, third_id = _ids()
    vectors = {
        first_id: CosineVector(first_id, [1.0, 2.0], CosineVectorKind.TENSOR),
        second_id: CosineVector(second_id, [1.0], CosineVectorKind.TENSOR),
    }
    config = CosineSimilarityConfig(candidate_mode=CosineCandidateMode.EXPLICIT)

    with pytest.raises(CosineSimilarityError, match="equal non-zero lengths"):
        extract_pairwise_cosine_similarities(
            vectors,
            config,
            explicit_candidates=(CosineCandidate(first_id, second_id),),
        )

    with pytest.raises(CosineSimilarityError, match="unknown vector"):
        tuple(
            generate_cosine_candidates(
                vectors,
                config,
                explicit_candidates=(CosineCandidate(first_id, third_id),),
            )
        )

    with pytest.raises(CosineSimilarityError, match="distinct"):
        CosineCandidate(first_id, first_id)

"""Tests for bounded transformer layer similarity features."""

from __future__ import annotations

import pytest

from modelsurgeon.features.layer_similarity import (
    LayerPair,
    LayerSimilarityConfig,
    LayerSimilarityError,
    LayerSummary,
    compare_layers,
    extract_layer_similarities,
)
from modelsurgeon.graph import ComponentId


def _ids() -> tuple[ComponentId, ComponentId, ComponentId]:
    return (
        ComponentId.parse("model.layers.0"),
        ComponentId.parse("model.layers.1"),
        ComponentId.parse("model.layers.2"),
    )


def test_identical_layer_fixture_has_maximal_residual_and_output_similarity() -> None:
    first_id, second_id, _ = _ids()
    first = LayerSummary(first_id, (2, 4), [1.0, 2.0, 3.0, 4.0], (2, 4), [4.0, 3.0, 2.0, 1.0])
    second = LayerSummary(second_id, (2, 4), [1.0, 2.0, 3.0, 4.0], (2, 4), [4.0, 3.0, 2.0, 1.0])

    result = compare_layers(first, second, LayerSimilarityConfig(block_size=2))

    assert result.residual_similarity == pytest.approx(1.0)
    assert result.output_similarity == pytest.approx(1.0)
    assert result.skipped == ()
    assert result.comparable is True
    assert result.to_record()["comparable"] is True


def test_non_comparable_shapes_are_skipped_with_exact_provenance() -> None:
    first_id, second_id, _ = _ids()
    first = LayerSummary(first_id, (2, 4), [1.0, 2.0], (2, 4), [1.0, 0.0])
    second = LayerSummary(second_id, (3, 4), [1.0, 2.0], (2, 5), [1.0, 0.0])

    result = compare_layers(first, second)

    assert result.residual_similarity is None
    assert result.output_similarity is None
    assert result.comparable is False
    assert result.skipped == (
        "residual_shape_mismatch:(2, 4)!=(3, 4)",
        "output_shape_mismatch:(2, 4)!=(2, 5)",
    )


def test_one_shape_can_skip_while_other_similarity_remains_available() -> None:
    first_id, second_id, _ = _ids()
    first = LayerSummary(first_id, (2, 4), [1.0, 2.0], (2, 4), [1.0, 1.0])
    second = LayerSummary(second_id, (3, 4), [9.0, 9.0], (2, 4), [1.0, 1.0])

    result = compare_layers(first, second)

    assert result.residual_similarity is None
    assert result.output_similarity == pytest.approx(1.0)
    assert result.comparable is True
    assert len(result.skipped) == 1


def test_pair_limit_and_unknown_layer_fail_closed() -> None:
    first_id, second_id, third_id = _ids()
    layers = {
        first_id: LayerSummary(first_id, (1, 2), [1.0], (1, 2), [1.0]),
        second_id: LayerSummary(second_id, (1, 2), [1.0], (1, 2), [1.0]),
    }

    with pytest.raises(LayerSimilarityError, match="pair limit exceeded"):
        extract_layer_similarities(
            layers,
            (LayerPair(first_id, second_id), LayerPair(first_id, second_id)),
            LayerSimilarityConfig(max_pairs=1),
        )

    with pytest.raises(LayerSimilarityError, match="unknown layer"):
        extract_layer_similarities(
            layers,
            (LayerPair(first_id, third_id),),
        )

"""Tests for bounded MLP duplicate-channel screening and confirmation."""

from __future__ import annotations

import pytest

from modelsurgeon.features.cosine_similarity import CosineCandidate
from modelsurgeon.features.neuron_correlation import (
    ChannelRedundancyInput,
    NeuronCorrelationConfig,
    NeuronCorrelationError,
    rank_duplicate_channels,
)
from modelsurgeon.graph import ComponentId


def _ids() -> tuple[ComponentId, ComponentId, ComponentId, ComponentId]:
    return tuple(
        ComponentId.parse(f"model.layers.0.mlp.channel.{index}") for index in range(4)
    )  # type: ignore[return-value]


def test_known_duplicate_fixture_ranks_together_first() -> None:
    first, duplicate, third, fourth = _ids()
    channels = {
        first: ChannelRedundancyInput(
            first,
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ),
        duplicate: ChannelRedundancyInput(
            duplicate,
            [2.0, 4.0, 6.0, 8.0],
            [0.0, 2.0, 4.0, 6.0, 8.0],
        ),
        third: ChannelRedundancyInput(
            third,
            [1.0, -1.0, 1.0, -1.0],
            [0.0, 1.0, 0.0, -1.0, 0.0],
        ),
        fourth: ChannelRedundancyInput(
            fourth,
            [-2.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, -1.0, 0.0, 1.0],
        ),
    }
    candidates = (
        CosineCandidate(first, third),
        CosineCandidate(first, duplicate),
        CosineCandidate(duplicate, fourth),
        CosineCandidate(third, fourth),
    )

    ranked = rank_duplicate_channels(
        channels,
        candidates,
        NeuronCorrelationConfig(candidate_budget=4, confirmation_budget=3, block_size=2),
    )

    assert ranked[0].left == first
    assert ranked[0].right == duplicate
    assert ranked[0].weight_cosine == pytest.approx(1.0)
    assert ranked[0].activation_correlation == pytest.approx(1.0)
    assert ranked[0].duplicate_score == pytest.approx(1.0)
    assert ranked[0].confirmation_rank == 1
    assert ranked[0].screening_rank == 1
    assert len(ranked) == 3


def test_confirmation_budget_limits_expensive_activation_checks() -> None:
    first, second, third, _ = _ids()
    channels = {
        first: ChannelRedundancyInput(first, [1.0, 0.0], [1.0, 2.0, 3.0]),
        second: ChannelRedundancyInput(second, [1.0, 0.0], [1.0, 2.0, 3.0]),
        third: ChannelRedundancyInput(third, [0.0, 1.0], [3.0, 2.0, 1.0]),
    }

    ranked = rank_duplicate_channels(
        channels,
        (
            CosineCandidate(first, second),
            CosineCandidate(first, third),
            CosineCandidate(second, third),
        ),
        NeuronCorrelationConfig(candidate_budget=3, confirmation_budget=1, block_size=1),
    )

    assert len(ranked) == 1
    assert ranked[0].candidate_budget == 3
    assert ranked[0].confirmation_budget == 1
    assert ranked[0].block_size == 1


def test_zero_variance_activation_confirmation_returns_zero() -> None:
    first, second, _, _ = _ids()
    channels = {
        first: ChannelRedundancyInput(first, [1.0, 2.0], [4.0, 4.0, 4.0]),
        second: ChannelRedundancyInput(second, [1.0, 2.0], [1.0, 2.0, 3.0]),
    }

    ranked = rank_duplicate_channels(
        channels,
        (CosineCandidate(first, second),),
        NeuronCorrelationConfig(candidate_budget=1, confirmation_budget=1),
    )

    assert ranked[0].weight_cosine == pytest.approx(1.0)
    assert ranked[0].activation_correlation == 0.0
    assert ranked[0].duplicate_score == 0.0
    assert ranked[0].to_record()["activation_zero_variance_policy"] == "return_zero"


def test_candidate_budget_and_unknown_channels_fail_closed() -> None:
    first, second, third, _ = _ids()
    channels = {
        first: ChannelRedundancyInput(first, [1.0], [1.0, 2.0]),
        second: ChannelRedundancyInput(second, [1.0], [1.0, 2.0]),
    }

    with pytest.raises(NeuronCorrelationError, match="candidate budget exceeded"):
        rank_duplicate_channels(
            channels,
            (CosineCandidate(first, second), CosineCandidate(first, second)),
            NeuronCorrelationConfig(candidate_budget=1, confirmation_budget=1),
        )

    with pytest.raises(NeuronCorrelationError, match="unknown channel"):
        rank_duplicate_channels(
            channels,
            (CosineCandidate(first, third),),
            NeuronCorrelationConfig(candidate_budget=1, confirmation_budget=1),
        )

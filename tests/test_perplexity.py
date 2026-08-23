"""Tests for masked token-weighted causal-LM loss and perplexity."""

from __future__ import annotations

import math

import pytest

from modelsurgeon.evaluation.baseline_cache import BaselineArtifact
from modelsurgeon.evaluation.perplexity import (
    CausalLMBatch,
    PerplexityEvaluationError,
    evaluate_perplexity,
)


def _row(target: int, width: int = 4) -> tuple[float, ...]:
    return tuple(3.0 if index == target else 0.0 for index in range(width))


def test_shifted_labels_and_padding_are_applied_to_target_positions() -> None:
    batch = CausalLMBatch(
        ((0, 1, 2, 3),),
        ((True, True, True, False),),
        (((_row(1), _row(2), _row(0), _row(0))),),
    )
    result = evaluate_perplexity((batch,))
    expected_nll = -math.log(math.exp(3.0) / (math.exp(3.0) + 3.0))
    assert result.token_count == 2
    assert result.mean_loss == pytest.approx(expected_nll)
    assert result.perplexity == pytest.approx(math.exp(expected_nll))


def test_token_weighting_agrees_when_sequences_are_rebatched() -> None:
    long_batch = CausalLMBatch(
        ((0, 1, 2, 3),),
        ((True, True, True, True),),
        (((_row(1), _row(2), _row(3), _row(0))),),
    )
    short_batch = CausalLMBatch(
        ((0, 2),),
        ((True, True),),
        (((_row(0), _row(0))),),
    )
    combined = CausalLMBatch(
        ((0, 1, 2, 3), (0, 2, 0, 0)),
        ((True, True, True, True), (True, True, False, False)),
        (
            (_row(1), _row(2), _row(3), _row(0)),
            (_row(0), _row(0), _row(0), _row(0)),
        ),
    )
    split = evaluate_perplexity((long_batch, short_batch))
    one_batch = evaluate_perplexity((combined,))
    assert split.token_count == one_batch.token_count == 4
    assert split.mean_loss == pytest.approx(one_batch.mean_loss)
    assert split.perplexity == pytest.approx(one_batch.perplexity)


def test_baseline_loss_and_perplexity_deltas_are_reported() -> None:
    batch = CausalLMBatch(
        ((0, 1),),
        ((True, True),),
        (((_row(1), _row(0))),),
    )
    baseline = BaselineArtifact(((1.0, 2.0),), 0.5, 1)
    result = evaluate_perplexity((batch,), baseline=baseline)
    assert result.baseline_mean_loss == 0.5
    assert result.baseline_perplexity == pytest.approx(math.exp(0.5))
    assert result.loss_delta == pytest.approx(result.mean_loss - 0.5)
    assert result.perplexity_delta == pytest.approx(result.perplexity - math.exp(0.5))


def test_invalid_target_nonfinite_logits_and_all_padding_fail() -> None:
    invalid_target = CausalLMBatch(
        ((0, 9),), ((True, True),), (((_row(0), _row(0))),)
    )
    with pytest.raises(PerplexityEvaluationError, match="outside vocabulary"):
        evaluate_perplexity((invalid_target,))

    nonfinite = CausalLMBatch(
        ((0, 1),),
        ((True, True),),
        ((((0.0, math.nan, 0.0, 0.0), _row(0))),),
    )
    with pytest.raises(PerplexityEvaluationError, match="finite"):
        evaluate_perplexity((nonfinite,))

    padded = CausalLMBatch(
        ((0, 1),), ((True, False),), (((_row(1), _row(0))),)
    )
    with pytest.raises(PerplexityEvaluationError, match="no valid"):
        evaluate_perplexity((padded,))


def test_batch_shape_contracts_fail_early() -> None:
    with pytest.raises(PerplexityEvaluationError, match="aligned"):
        CausalLMBatch(((0, 1),), ((True,),), (((_row(1), _row(0))),))
    with pytest.raises(PerplexityEvaluationError, match="at least one"):
        evaluate_perplexity(())

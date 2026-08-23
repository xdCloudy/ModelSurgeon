"""Tests for teacher-to-candidate KL and logit similarity metrics."""

from __future__ import annotations

import math

import pytest

from modelsurgeon.evaluation.logit_metrics import (
    LogitMetricConfig,
    LogitMetricError,
    LogitPairBatch,
    evaluate_logit_similarity,
)


def test_identical_logits_have_zero_kl_and_perfect_similarity() -> None:
    rows = ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))
    result = evaluate_logit_similarity((LogitPairBatch(rows, rows),), LogitMetricConfig(top_k=2))
    assert result.mean_teacher_to_candidate_kl == pytest.approx(0.0, abs=1e-15)
    assert result.mean_cosine_similarity == pytest.approx(1.0)
    assert result.mean_top_k_agreement == pytest.approx(1.0)
    assert result.temperature == 1.0
    assert result.reduction == "mean_over_token_rows"


def test_teacher_to_candidate_kl_is_directional_and_temperature_recorded() -> None:
    teacher = ((4.0, 0.0, -1.0),)
    candidate = ((0.0, 4.0, -1.0),)
    config = LogitMetricConfig(temperature=2.0, top_k=1)
    forward = evaluate_logit_similarity((LogitPairBatch(teacher, candidate),), config)
    reverse = evaluate_logit_similarity((LogitPairBatch(candidate, teacher),), config)
    assert forward.mean_teacher_to_candidate_kl > 0
    assert reverse.mean_teacher_to_candidate_kl > 0
    assert forward.temperature == 2.0
    assert forward.top_k == 1
    assert forward.mean_top_k_agreement == 0.0


def test_streamed_batches_reduce_by_token_row_count() -> None:
    perfect = LogitPairBatch(((3.0, 1.0),), ((3.0, 1.0),))
    different = LogitPairBatch(
        ((3.0, 1.0), (2.0, 0.0)),
        ((1.0, 3.0), (0.0, 2.0)),
    )
    combined = LogitPairBatch(
        ((3.0, 1.0), (3.0, 1.0), (2.0, 0.0)),
        ((3.0, 1.0), (1.0, 3.0), (0.0, 2.0)),
    )
    split = evaluate_logit_similarity((perfect, different), LogitMetricConfig(top_k=1))
    one = evaluate_logit_similarity((combined,), LogitMetricConfig(top_k=1))
    assert split.token_rows == one.token_rows == 3
    assert split.mean_teacher_to_candidate_kl == pytest.approx(one.mean_teacher_to_candidate_kl)
    assert split.mean_cosine_similarity == pytest.approx(one.mean_cosine_similarity)
    assert split.mean_top_k_agreement == pytest.approx(one.mean_top_k_agreement)


def test_deterministic_top_k_ties_use_lowest_indices() -> None:
    result = evaluate_logit_similarity(
        (LogitPairBatch(((1.0, 1.0, 0.0),), ((1.0, 0.0, 1.0),)),),
        LogitMetricConfig(top_k=2),
    )
    assert result.mean_top_k_agreement == pytest.approx(0.5)


def test_invalid_shapes_nonfinite_values_and_config_fail() -> None:
    with pytest.raises(LogitMetricError, match="align"):
        LogitPairBatch(((1.0, 2.0),), ())
    with pytest.raises(LogitMetricError, match="vocabulary"):
        LogitPairBatch(((1.0, 2.0),), ((1.0,),))
    with pytest.raises(LogitMetricError, match="finite"):
        LogitPairBatch(((math.nan, 2.0),), ((1.0, 2.0),))
    with pytest.raises(LogitMetricError, match="temperature"):
        LogitMetricConfig(temperature=0.0)
    with pytest.raises(LogitMetricError, match="at least one"):
        evaluate_logit_similarity(())

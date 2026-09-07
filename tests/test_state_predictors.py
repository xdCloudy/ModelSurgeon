"""Tests for calibrated state-dependent predictor contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.surgeon import (
    StateEmbedding,
    StateEmbeddingConfig,
    StatePredictionDataset,
    StatePredictionSample,
    StatePredictionSplit,
    StatePredictionStatus,
    StatePredictorConfig,
    StatePredictorError,
    StateTargetName,
    StateTargetObservation,
    StateTargetStatus,
    fit_state_predictors,
)


def _embedding(index: int, first: float, second: float) -> StateEmbedding:
    state_id = f"state_{index:064x}"
    config = StateEmbeddingConfig(
        max_history=2,
        max_set_items=2,
        mutation_buckets=2,
        history_buckets=2,
        constraint_buckets=2,
        histogram_bins=2,
        metric_names=(),
    )
    return StateEmbedding(
        state_id,
        (first, second),
        (1.0, 1.0),
        ("feature_a", "feature_b"),
        config,
        0,
        0,
        None,
    )


def _sample(
    index: int,
    group: str,
    model: str,
    candidate: str,
    value: float | None,
    status: StateTargetStatus = StateTargetStatus.MEASURED,
) -> StatePredictionSample:
    embedding = _embedding(index, float(index) / 10.0, float(index * index) / 100.0)
    observation = StateTargetObservation(
        StateTargetName.QUALITY_DELTA,
        "quality_loss",
        status,
        value,
        None if value is None else value + 0.4,
    )
    return StatePredictionSample(
        f"sample-{index}",
        embedding.state_id,
        model,
        candidate,
        group,
        embedding,
        (observation,),
    )


def _dataset() -> StatePredictionDataset:
    samples = (
        _sample(1, "train", "model-train", "candidate-1", 0.1),
        _sample(2, "train", "model-train", "candidate-2", 0.2),
        _sample(3, "train", "model-train", "candidate-3", 0.3),
        _sample(4, "train", "model-train", "candidate-4", None, StateTargetStatus.FAILED),
        _sample(5, "validation", "model-validation", "candidate-5", 0.5),
        _sample(6, "validation", "model-validation", "candidate-6", 0.6),
        _sample(7, "test", "model-test", "candidate-7", 0.7),
        _sample(8, "test", "model-test", "candidate-8", 0.8),
    )
    return StatePredictionDataset(
        samples,
        StatePredictionSplit(("train",), ("validation",), ("test",)),
        "state-predictor-fixture-v1",
    )


def test_state_predictor_calibrates_intervals_and_failure_probability() -> None:
    study = fit_state_predictors(
        _dataset(),
        StatePredictorConfig(
            target_names=(StateTargetName.QUALITY_DELTA,), minimum_train_examples=3
        ),
    )
    model = study.model_for(StateTargetName.QUALITY_DELTA)
    assert model is not None
    assert 0.0 < model.failure_probability < 1.0
    assert model.interval_radius >= 0.0
    score = study.scores[0]
    assert score.target is StateTargetName.QUALITY_DELTA
    assert score.test_count == 2
    assert score.stateless_mae is not None
    assert score.additive_mae is not None
    assert score.failure_brier is not None

    first = _dataset().samples[0].embedding
    second = _dataset().samples[1].embedding
    first_prediction = study.predict(first, StateTargetName.QUALITY_DELTA)
    second_prediction = study.predict(second, StateTargetName.QUALITY_DELTA)
    assert first_prediction.status is StatePredictionStatus.PREDICTED
    assert second_prediction.status is StatePredictionStatus.PREDICTED
    assert first_prediction.value != second_prediction.value
    assert first_prediction.failure_probability == model.failure_probability


def test_unseen_state_is_separate_from_numeric_prediction() -> None:
    study = fit_state_predictors(
        _dataset(),
        StatePredictorConfig(
            target_names=(StateTargetName.QUALITY_DELTA,), minimum_train_examples=3
        ),
    )
    unseen = _embedding(99, 0.2, 0.1)
    prediction = study.predict(unseen, StateTargetName.QUALITY_DELTA)

    assert prediction.status is StatePredictionStatus.OUT_OF_DISTRIBUTION
    assert prediction.value is None
    assert prediction.failure_probability > 0.0


def test_parent_model_and_candidate_leakage_is_rejected() -> None:
    samples = _dataset().samples
    leaking = replace(samples[-1], model_revision="model-train")
    with pytest.raises(StatePredictorError, match="leakage"):
        StatePredictionDataset(
            (*samples[:-1], leaking),
            StatePredictionSplit(("train",), ("validation",), ("test",)),
            "fixture",
        )

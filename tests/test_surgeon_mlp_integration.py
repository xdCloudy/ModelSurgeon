"""Optional PyTorch integration smoke for the bounded surgeon MLP baseline."""

from __future__ import annotations

import pytest

from modelsurgeon.datasets.grouped_splits import SplitPartition
from modelsurgeon.surgeon.matrix import SurgeonMatrix
from modelsurgeon.surgeon.models import (
    MLPConfig,
    MLPSurgeonModel,
    ModelTask,
    model_from_json,
    model_to_json,
    train_mlp,
)

pytest.importorskip("torch")


def _matrix(
    partition: SplitPartition,
    target: str,
    rows: tuple[tuple[float, float], ...],
    labels: tuple[float, ...],
) -> SurgeonMatrix:
    count = len(rows)
    return SurgeonMatrix(
        partition,
        tuple(f"{partition.value}-{index}" for index in range(count)),
        ("num:x", "num:y"),
        rows,
        target,
        labels,
        (True,) * count,
        (1.0,) * count,
        tuple(f"group-{partition.value}-{index}" for index in range(count)),
    )


def test_real_pytorch_mlp_regressor_round_trips() -> None:
    train = _matrix(
        SplitPartition.TRAIN,
        "perplexity",
        ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)),
        (0.0, 0.5, 0.5, 1.0),
    )
    validation = _matrix(
        SplitPartition.VALIDATION,
        "perplexity",
        ((0.25, 0.25), (0.75, 0.75)),
        (0.25, 0.75),
    )
    model = train_mlp(
        train,
        validation,
        config=MLPConfig(
            task=ModelTask.REGRESSION,
            hidden_sizes=(8,),
            learning_rate=0.01,
            batch_size=4,
            max_epochs=30,
            patience=8,
            seed=7,
        ),
    )

    assert isinstance(model, MLPSurgeonModel)
    predictions = model.predict(validation.values)
    assert len(predictions) == 2
    restored = model_from_json(model_to_json(model))
    assert restored.to_record() == model.to_record()
    assert restored.predict(validation.values) == pytest.approx(predictions)


def test_real_pytorch_mlp_classifier_returns_probabilities() -> None:
    train = _matrix(
        SplitPartition.TRAIN,
        "safe_mutation",
        ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)),
        (0.0, 0.0, 1.0, 1.0),
    )
    validation = _matrix(
        SplitPartition.VALIDATION,
        "safe_mutation",
        ((0.1, 0.2), (0.9, 0.8)),
        (0.0, 1.0),
    )
    model = train_mlp(
        train,
        validation,
        config=MLPConfig(
            task=ModelTask.CLASSIFICATION,
            hidden_sizes=(8,),
            learning_rate=0.01,
            batch_size=4,
            max_epochs=30,
            patience=8,
            seed=11,
        ),
    )

    probabilities = model.predict(validation.values)
    assert len(probabilities) == 2
    assert all(0.0 <= value <= 1.0 for value in probabilities)

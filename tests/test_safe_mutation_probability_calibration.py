from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.datasets.grouped_splits import SplitPartition
from modelsurgeon.surgeon.matrix import SurgeonMatrix
from modelsurgeon.surgeon.models import LogisticSurgeonModel, train_logistic


def _matrix(
    partition: SplitPartition,
    labels: tuple[float, ...],
    *,
    weights: tuple[float, ...] | None = None,
    target_name: str = "safe_mutation",
) -> SurgeonMatrix:
    return SurgeonMatrix(
        partition=partition,
        example_ids=tuple(f"{partition.value}-{index}" for index in range(len(labels))),
        feature_names=("importance",),
        values=tuple((float(index),) for index in range(len(labels))),
        target_name=target_name,
        target_values=labels,
        target_mask=(True,) * len(labels),
        sample_weights=weights or (1.0,) * len(labels),
        group_ids=tuple(f"group-{index}" for index in range(len(labels))),
    )


@pytest.fixture
def training_matrices() -> tuple[SurgeonMatrix, SurgeonMatrix]:
    train = _matrix(SplitPartition.TRAIN, (0.0, 1.0) * 8)
    validation = _matrix(SplitPartition.VALIDATION, (0.0, 1.0) * 4)
    return train, validation


def test_safe_mutation_logistic_fits_and_persists_validation_calibration(
    training_matrices: tuple[SurgeonMatrix, SurgeonMatrix],
) -> None:
    train, validation = training_matrices
    model = train_logistic(train, validation=validation)

    assert model.calibration_status == "fitted"
    assert model.calibration_selection is not None
    raw_model = replace(model, calibration_selection=None, calibration_status="not_requested")
    raw = raw_model.predict_proba(validation.values)
    expected = model.calibration_selection.calibrator.calibrate(raw)
    assert model.predict_proba(validation.values) == expected

    restored = LogisticSurgeonModel.from_record(model.to_record())
    assert restored == model
    assert restored.predict_proba(validation.values) == expected

    legacy_record = model.to_record()
    legacy_record["schema_version"] = 1
    legacy_record.pop("calibration_selection")
    legacy_record.pop("calibration_status")
    legacy = LogisticSurgeonModel.from_record(legacy_record)
    assert legacy.calibration_selection is None
    assert legacy.predict_proba(validation.values) == raw


@pytest.mark.parametrize(
    ("labels", "weights", "expected_status"),
    [
        ((0.0, 0.0, 0.0, 1.0), None, "skipped_insufficient_validation_classes"),
        ((0.0, 1.0, 0.0, 1.0), (1.0, 2.0, 1.0, 2.0), "skipped_nonuniform_validation_weights"),
    ],
)
def test_calibration_abstains_when_validation_evidence_is_unsuitable(
    training_matrices: tuple[SurgeonMatrix, SurgeonMatrix],
    labels: tuple[float, ...],
    weights: tuple[float, ...] | None,
    expected_status: str,
) -> None:
    train, _ = training_matrices
    validation = _matrix(SplitPartition.VALIDATION, labels, weights=weights)

    model = train_logistic(train, validation=validation)

    assert model.calibration_status == expected_status
    assert model.calibration_selection is None
    raw_model = replace(model, calibration_status="not_requested")
    assert model.predict_proba(validation.values) == raw_model.predict_proba(validation.values)


def test_other_logistic_targets_do_not_use_safe_mutation_calibration(
    training_matrices: tuple[SurgeonMatrix, SurgeonMatrix],
) -> None:
    train, validation = training_matrices
    other_train = replace(train, target_name="quality_retained")
    other_validation = replace(validation, target_name="quality_retained")

    model = train_logistic(other_train, validation=other_validation)

    assert model.calibration_status == "not_safe_mutation_target"
    assert model.calibration_selection is None

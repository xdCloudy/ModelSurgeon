from __future__ import annotations

import pytest

from modelsurgeon.surgeon import (
    RecoverabilitySplit,
    RepairCostConfig,
    RepairCostDataset,
    RepairCostLimits,
    RepairCostObservation,
    RepairCostObservationStatus,
    RepairCostPredictionStatus,
    RepairCostPreflightStatus,
    RepairCostSample,
    RepairCostTarget,
    fit_repair_cost_predictors,
)
from modelsurgeon.surgeon.recoverability import RecoverabilityAction
from modelsurgeon.surgery import RepairMethod

NO_REPAIR = RecoverabilityAction(RepairMethod.NO_REPAIR, "zero")
LORA = RecoverabilityAction(RepairMethod.LORA, "short")


def _values(index: int, action: RecoverabilityAction) -> tuple[tuple[RepairCostTarget, float], ...]:
    multiplier = 0.0 if action is NO_REPAIR else float(index + 1)
    values = {
        RepairCostTarget.TOKENS: 100.0 * multiplier,
        RepairCostTarget.WALL_SECONDS: 2.0 * multiplier,
        RepairCostTarget.GPU_SECONDS: 1.0 * multiplier,
        RepairCostTarget.CPU_SECONDS: 3.0 * multiplier,
        RepairCostTarget.PEAK_RAM_BYTES: 1_000.0 * multiplier,
        RepairCostTarget.PEAK_VRAM_BYTES: 2_000.0 * multiplier,
        RepairCostTarget.ARTIFACT_BYTES: 300.0 * multiplier,
        RepairCostTarget.CACHE_BYTES: 400.0 * multiplier,
    }
    return tuple(sorted(values.items(), key=lambda item: item[0].value))


def _sample(index: int, action: RecoverabilityAction, partition: str) -> RepairCostSample:
    group = f"{partition}-{index}"
    return RepairCostSample(
        f"sample-{partition}-{index}-{action.method.value}",
        f"model-{partition}-{index}",
        f"hardware-{partition}-{index}",
        f"state-{partition}-{index}",
        f"candidate-{partition}-{index}",
        group,
        (float(index), 1.0 + index),
        action,
        RepairCostObservation(RepairCostObservationStatus.MEASURED, _values(index, action)),
    )


def _dataset() -> RepairCostDataset:
    samples = tuple(
        sample
        for partition, count in (("train", 4), ("validation", 2), ("test", 2))
        for index in range(count)
        for sample in (_sample(index, NO_REPAIR, partition), _sample(index, LORA, partition))
    )
    return RepairCostDataset(
        samples,
        RecoverabilitySplit(
            tuple(f"train-{index}" for index in range(4)),
            tuple(f"validation-{index}" for index in range(2)),
            tuple(f"test-{index}" for index in range(2)),
        ),
        ("damage", "parameters"),
        "repair-dataset-v1",
    )


def test_cost_study_retains_optional_energy_gap_and_is_deterministic() -> None:
    config = RepairCostConfig(minimum_train_examples=3)
    first = fit_repair_cost_predictors(_dataset(), config)
    second = fit_repair_cost_predictors(_dataset(), config)

    assert first.to_record() == second.to_record()
    energy_scores = [item for item in first.scores if item.target is RepairCostTarget.ENERGY_JOULES]
    assert energy_scores
    assert all(item.outcome is RepairCostObservationStatus.UNSUPPORTED for item in energy_scores)
    assert any(item.target is RepairCostTarget.WALL_SECONDS for item in first.models)


def test_preflight_uses_conservative_bounds_and_optional_energy_does_not_block() -> None:
    study = fit_repair_cost_predictors(_dataset(), RepairCostConfig(minimum_train_examples=3))
    limits = RepairCostLimits(
        10_000,
        10_000,
        10_000,
        10_000,
        10_000_000,
        10_000_000,
        10_000_000,
        10_000_000,
    )
    allowed = study.preflight((1.0, 2.0), "state-train-0", LORA, limits)
    assert allowed.status is RepairCostPreflightStatus.ALLOWED
    assert any(item.target is RepairCostTarget.ENERGY_JOULES for item in allowed.predictions)

    over = study.preflight(
        (1.0, 2.0),
        "state-train-0",
        LORA,
        RepairCostLimits(
            10_000, 0.1, 10_000, 10_000, 10_000_000, 10_000_000, 10_000_000, 10_000_000
        ),
    )
    assert over.status is RepairCostPreflightStatus.OVER_BUDGET
    assert RepairCostTarget.WALL_SECONDS in over.exceeded_targets


def test_cost_prediction_rejects_unseen_state_and_incompatible_action() -> None:
    study = fit_repair_cost_predictors(_dataset())
    predictions = study.predict((1.0, 2.0), "state-new", (LORA,))
    assert all(
        item.status is RepairCostPredictionStatus.OUT_OF_DISTRIBUTION
        for item in predictions
        if item.target is not RepairCostTarget.ENERGY_JOULES
    )

    incompatible = RecoverabilityAction(
        RepairMethod.LORA,
        "short",
        compatible=False,
        compatibility_reason="unsupported codec",
    )
    incompatible_predictions = study.predict((1.0, 2.0), "state-train-0", (incompatible,))
    assert all(
        item.status is RepairCostPredictionStatus.UNSUPPORTED for item in incompatible_predictions
    )


def test_cost_dataset_rejects_terminal_values() -> None:
    with pytest.raises(ValueError, match="terminal"):
        RepairCostObservation(
            RepairCostObservationStatus.FAILED,
            ((RepairCostTarget.TOKENS, 1.0),),
            "out of memory",
        )

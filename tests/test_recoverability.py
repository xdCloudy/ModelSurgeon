from __future__ import annotations

import pytest

from modelsurgeon.surgeon import (
    RecoverabilityAction,
    RecoverabilityConfig,
    RecoverabilityDataset,
    RecoverabilityDecisionStatus,
    RecoverabilityError,
    RecoverabilityObservation,
    RecoverabilityOutcome,
    RecoverabilityPredictionStatus,
    RecoverabilitySample,
    RecoverabilitySplit,
    RecoverabilityTargetStatus,
    fit_recoverability_predictors,
)
from modelsurgeon.surgery import RepairMethod

NO_REPAIR = RecoverabilityAction(RepairMethod.NO_REPAIR, "zero")
LORA = RecoverabilityAction(RepairMethod.LORA, "short")


def _sample(index: int, action: RecoverabilityAction, partition: str) -> RecoverabilitySample:
    group = f"{partition}-{index}"
    gain = 0.0 if action is NO_REPAIR else 0.2 + index * 0.03
    return RecoverabilitySample(
        f"sample-{partition}-{index}-{action.method.value}",
        f"model-{partition}-{index}",
        f"state-{partition}-{index}",
        f"candidate-{partition}-{index}",
        "gemma" if index % 2 == 0 else "llama",
        group,
        (float(index), 1.0 + index * 0.1),
        action,
        RecoverabilityObservation(
            RecoverabilityTargetStatus.MEASURED,
            gain,
            action is NO_REPAIR or index != 1,
        ),
    )


def _dataset() -> RecoverabilityDataset:
    samples = tuple(
        sample
        for partition, count in (("train", 4), ("validation", 2), ("test", 2))
        for index in range(count)
        for sample in (_sample(index, NO_REPAIR, partition), _sample(index, LORA, partition))
    )
    return RecoverabilityDataset(
        samples,
        RecoverabilitySplit(
            tuple(f"train-{index}" for index in range(4)),
            tuple(f"validation-{index}" for index in range(2)),
            tuple(f"test-{index}" for index in range(2)),
        ),
        ("damage", "width"),
        "repair-dataset-v1",
    )


def test_recoverability_fit_is_deterministic_and_retains_scores() -> None:
    dataset = _dataset()
    config = RecoverabilityConfig(minimum_train_examples=3, minimum_gain=0.0)
    first = fit_recoverability_predictors(dataset, config)
    second = fit_recoverability_predictors(dataset, config)

    assert first.to_record() == second.to_record()
    assert {item.action.key for item in first.models} == {
        (RepairMethod.NO_REPAIR.value, "zero"),
        (RepairMethod.LORA.value, "short"),
    }
    assert all(
        item.outcome in {RecoverabilityOutcome.MEASURED, RecoverabilityOutcome.NEGATIVE_RESULT}
        for item in first.scores
    )


def test_no_repair_is_first_class_and_unseen_states_abstain() -> None:
    study = fit_recoverability_predictors(
        _dataset(),
        RecoverabilityConfig(minimum_train_examples=3, minimum_gain=1.0),
    )
    decision = study.decide(
        (1.0, 1.1),
        "state-train-0",
        (NO_REPAIR, LORA),
    )
    assert decision.status is RecoverabilityDecisionStatus.NO_REPAIR
    assert decision.selected_action == NO_REPAIR

    unseen = study.predict_actions((1.0, 1.1), "state-new", (NO_REPAIR, LORA))
    assert all(item.status is RecoverabilityPredictionStatus.OUT_OF_DISTRIBUTION for item in unseen)
    assert (
        study.decide((1.0, 1.1), "state-new", (NO_REPAIR, LORA)).status
        is RecoverabilityDecisionStatus.OUT_OF_DISTRIBUTION
    )


def test_recoverability_rejects_split_leakage_and_terminal_values() -> None:
    with pytest.raises(RecoverabilityError, match="terminal"):
        RecoverabilityObservation(RecoverabilityTargetStatus.FAILED, 0.1, None, "oom")

    dataset = _dataset()
    leaking = RecoverabilitySample(
        "sample-leak",
        "model-train-0",
        "state-train-0",
        "candidate-train-0",
        "gemma",
        "test-0",
        (0.0, 1.0),
        NO_REPAIR,
        RecoverabilityObservation(RecoverabilityTargetStatus.MEASURED, 0.0, True),
    )
    with pytest.raises(RecoverabilityError, match="leakage"):
        RecoverabilityDataset(
            (*dataset.samples, leaking),
            dataset.split,
            dataset.feature_names,
            dataset.dataset_revision,
        )


def test_incompatible_repair_action_is_not_recommended() -> None:
    study = fit_recoverability_predictors(_dataset())
    incompatible = RecoverabilityAction(
        RepairMethod.LORA,
        "short",
        compatible=False,
        compatibility_reason="codec does not support adapters",
    )
    decision = study.decide((1.0, 1.1), "state-train-0", (NO_REPAIR, incompatible))
    assert decision.status is RecoverabilityDecisionStatus.NO_REPAIR

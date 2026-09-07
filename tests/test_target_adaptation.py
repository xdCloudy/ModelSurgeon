from __future__ import annotations

import pytest

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.surgeon import (
    AdaptationLearningPoint,
    AdaptationOutcomeStatus,
    AdaptationTransition,
    CompatibilityContext,
    CompatibilityOperation,
    ModelLineageGraph,
    ModelLineageNode,
    TargetAdaptationConfig,
    TargetAdaptationError,
    TargetExamplePartition,
    TargetMutationExample,
    decide_compatibility,
    plan_target_adaptation,
    record_target_adaptation_result,
    select_target_support,
)


def _compatibility():
    source = CompatibilityContext(
        "source",
        "rev-source",
        ModelFamily.LLAMA,
        "arch-v1",
        "features-v1",
        1,
        "targets-v1",
        1,
        "q4_k",
        "cpu",
        1,
        (CompatibilityOperation.INFER,),
    )
    target = CompatibilityContext(
        "target",
        "rev-target",
        ModelFamily.QWEN,
        "arch-v1",
        "features-v1",
        1,
        "targets-v1",
        1,
        "q4_k",
        "cpu",
        1,
        (CompatibilityOperation.INFER,),
    )
    return decide_compatibility(
        source,
        target,
        operation=CompatibilityOperation.INFER,
        lineage=ModelLineageGraph(
            (
                ModelLineageNode("source", "rev-source", ModelFamily.LLAMA, "arch-v1"),
                ModelLineageNode("target", "rev-target", ModelFamily.QWEN, "arch-v1"),
            )
        ),
        allow_adaptation=True,
    )


def _examples() -> tuple[TargetMutationExample, ...]:
    return tuple(
        TargetMutationExample(
            f"support-{index}",
            "target",
            TargetExamplePartition.SUPPORT_POOL,
            f"feature-{index}",
            float(index),
        )
        for index in range(4)
    ) + tuple(
        TargetMutationExample(
            f"test-{index}",
            "target",
            TargetExamplePartition.TEST,
            f"test-feature-{index}",
        )
        for index in range(2)
    )


def test_zero_and_few_shot_requests_are_deterministic_and_leakage_safe() -> None:
    config = TargetAdaptationConfig(budgets=(0, 2, 4))
    first = plan_target_adaptation(
        "sha256:parent", "target", _compatibility(), _examples(), config=config
    )
    second = plan_target_adaptation(
        "sha256:parent", "target", _compatibility(), _examples(), config=config
    )
    assert first.to_record() == second.to_record()
    for request in first.requests:
        assert not set(request.support_example_ids) & set(request.test_example_ids)
        if request.target_example_budget == 0:
            assert request.support_example_ids == ()
            assert request.target_statistics_digest is None
            assert request.mode.value == "zero_shot"
        else:
            assert len(request.support_example_ids) == request.target_example_budget


def test_result_recording_retains_commit_and_rollback_outcomes() -> None:
    study = plan_target_adaptation(
        "sha256:parent",
        "target",
        _compatibility(),
        _examples(),
        config=TargetAdaptationConfig(budgets=(0, 2)),
    )
    committed = record_target_adaptation_result(
        study,
        study.requests[0].request_id,
        status=AdaptationOutcomeStatus.MEASURED,
        transition=AdaptationTransition.COMMITTED,
        child_bundle_digest="sha256:child",
        learning_curve=(AdaptationLearningPoint(0, 0.4, 0.1, 1.0, 2.0),),
    )
    rolled_back = record_target_adaptation_result(
        committed,
        committed.requests[1].request_id,
        status=AdaptationOutcomeStatus.NEGATIVE_RESULT,
        transition=AdaptationTransition.ROLLED_BACK,
        child_bundle_digest="sha256:discarded-child",
        learning_curve=(AdaptationLearningPoint(2, 0.2, -0.1, 1.0, 2.0),),
        failures=("equal-budget frontier did not improve",),
    )
    assert len(rolled_back.results) == 2
    assert rolled_back.results[1].transition is AdaptationTransition.ROLLED_BACK


def test_support_selection_rejects_budget_overrun_and_test_labels() -> None:
    with pytest.raises(TargetAdaptationError, match="smaller"):
        select_target_support(_examples(), 5, 0)
    with pytest.raises(TargetAdaptationError, match="cannot carry test outcomes"):
        TargetMutationExample(
            "test-bad",
            "target",
            TargetExamplePartition.TEST,
            "feature",
            1.0,
        )

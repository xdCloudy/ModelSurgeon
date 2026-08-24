from __future__ import annotations

from modelsurgeon.active_learning import (
    MetricDirection,
    PromotionCriterion,
    RetrainingTriggerConfig,
    evaluate_challenger_promotion,
    evaluate_retraining_trigger,
)


def test_retraining_triggers_record_count_elapsed_and_drift_reasons() -> None:
    config = RetrainingTriggerConfig(100, 3600.0, 0.2)

    decision = evaluate_retraining_trigger(100, 4000.0, 0.1, config=config)

    assert decision.triggered
    assert decision.reasons == ("example-count", "elapsed-budget")


def test_explicit_promotion_criteria_promote_only_when_all_pass() -> None:
    criteria = (
        PromotionCriterion("auc", MetricDirection.MAXIMIZE, 0.01),
        PromotionCriterion("brier", MetricDirection.MINIMIZE, 0.005),
    )
    promoted = evaluate_challenger_promotion(
        "surgeon-v1",
        "surgeon-v2",
        {"auc": 0.70, "brier": 0.20},
        {"auc": 0.72, "brier": 0.19},
        criteria,
        challenger_succeeded=True,
    )

    assert promoted.promoted and promoted.active_version == "surgeon-v2"
    assert all(item.passed for item in promoted.metrics)


def test_failed_challenger_never_replaces_incumbent() -> None:
    decision = evaluate_challenger_promotion(
        "surgeon-v1",
        "surgeon-v2",
        {"auc": 0.70},
        {"auc": 0.99},
        (PromotionCriterion("auc", MetricDirection.MAXIMIZE),),
        challenger_succeeded=False,
    )

    assert not decision.promoted
    assert decision.active_version == "surgeon-v1"
    assert decision.reason == "challenger-training-failed"

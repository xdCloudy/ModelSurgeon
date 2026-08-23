"""Tests for sequential Tier 0-3 evaluation escalation."""

from __future__ import annotations

import math

import pytest

from modelsurgeon.evaluation.latency import LatencyComparison
from modelsurgeon.evaluation.logit_metrics import LogitMetricResult
from modelsurgeon.evaluation.numerics import Tier0NumericsResult
from modelsurgeon.evaluation.perplexity import PerplexityResult
from modelsurgeon.evaluation.tier0 import Tier0Stage, Tier0ValidationResult
from modelsurgeon.evaluation.tiered import (
    EscalationAction,
    EvaluationTier,
    ThresholdComparator,
    TieredEvaluationConfig,
    TieredEvaluationError,
    TierThreshold,
    run_tiered_evaluation,
)


def _load_pass() -> Tier0ValidationResult:
    return Tier0ValidationResult(
        True,
        tuple(Tier0Stage),
        None,
        None,
        None,
        "cpu",
        32,
    )


def _load_fail() -> Tier0ValidationResult:
    return Tier0ValidationResult(
        False,
        (),
        Tier0Stage.LOAD,
        "ValueError",
        "broken checkpoint",
        "cpu",
        32,
    )


def _numerics_pass() -> Tier0NumericsResult:
    return Tier0NumericsResult(True, 2, 4, None, 4096)


def _perplexity(*, loss_delta: float | None = 0.05) -> PerplexityResult:
    baseline_loss = None if loss_delta is None else 0.95
    baseline_perplexity = None if baseline_loss is None else math.exp(baseline_loss)
    return PerplexityResult(
        1.0,
        math.e,
        10,
        10.0,
        baseline_loss,
        baseline_perplexity,
        loss_delta,
        None if baseline_perplexity is None else math.e - baseline_perplexity,
    )


def _logits() -> LogitMetricResult:
    return LogitMetricResult(
        0.02,
        0.98,
        0.9,
        5,
        1.0,
        5,
        "mean_over_token_rows",
    )


class Backend:
    def __init__(
        self,
        *,
        load: Tier0ValidationResult | None = None,
        numerics: Tier0NumericsResult | None = None,
        perplexity: PerplexityResult | None = None,
        logits: LogitMetricResult | None = None,
        latency: LatencyComparison | None = None,
    ) -> None:
        self.load = load or _load_pass()
        self.numerics = numerics or _numerics_pass()
        self.perplexity = perplexity or _perplexity()
        self.logits = logits or _logits()
        self.latency = latency or LatencyComparison(True, None, 1.1, 1.2)
        self.calls: list[str] = []

    def run_tier0_load(self) -> Tier0ValidationResult:
        self.calls.append("tier0_load")
        return self.load

    def run_tier0_numerics(self) -> Tier0NumericsResult:
        self.calls.append("tier0_numerics")
        return self.numerics

    def run_tier1_perplexity(self) -> PerplexityResult:
        self.calls.append("tier1")
        return self.perplexity

    def run_tier2_logit_metrics(self) -> LogitMetricResult:
        self.calls.append("tier2")
        return self.logits

    def run_tier3_latency(self) -> LatencyComparison:
        self.calls.append("tier3")
        return self.latency


def test_tier0_load_failure_prevents_all_higher_tiers_and_numerics() -> None:
    backend = Backend(load=_load_fail())
    report = run_tiered_evaluation(backend)

    assert backend.calls == ["tier0_load"]
    assert not report.accepted
    assert report.highest_executed_tier is EvaluationTier.TIER0
    assert report.decisions[0].action is EscalationAction.REJECT
    assert report.decisions[0].metrics[1].reason is not None
    assert all(
        decision.action is EscalationAction.SKIP for decision in report.decisions[1:]
    )


def test_tier1_threshold_failure_stops_logit_and_latency_work() -> None:
    backend = Backend(perplexity=_perplexity(loss_delta=0.5))
    config = TieredEvaluationConfig(
        thresholds=(
            TierThreshold(
                EvaluationTier.TIER1,
                "loss_delta",
                ThresholdComparator.MAXIMUM,
                0.1,
            ),
        )
    )
    report = run_tiered_evaluation(backend, config)

    assert backend.calls == ["tier0_load", "tier0_numerics", "tier1"]
    assert not report.accepted
    tier1 = report.decisions[1]
    assert tier1.action is EscalationAction.REJECT
    decision = next(item for item in tier1.metrics if item.metric == "loss_delta")
    assert decision.value == 0.5
    assert decision.passed is False
    assert report.decisions[2].reason == "candidate rejected by Tier 1"
    assert report.decisions[3].reason == "candidate rejected by Tier 1"


def test_full_pass_records_thresholds_and_explicit_escalation_decisions() -> None:
    backend = Backend()
    config = TieredEvaluationConfig(
        thresholds=(
            TierThreshold(
                EvaluationTier.TIER1,
                "loss_delta",
                ThresholdComparator.MAXIMUM,
                0.1,
            ),
            TierThreshold(
                EvaluationTier.TIER2,
                "teacher_to_candidate_kl",
                ThresholdComparator.MAXIMUM,
                0.05,
            ),
            TierThreshold(
                EvaluationTier.TIER2,
                "cosine_similarity",
                ThresholdComparator.MINIMUM,
                0.95,
            ),
            TierThreshold(
                EvaluationTier.TIER3,
                "decode_speedup_ratio",
                ThresholdComparator.MINIMUM,
                1.0,
            ),
        )
    )
    report = run_tiered_evaluation(backend, config)

    assert backend.calls == ["tier0_load", "tier0_numerics", "tier1", "tier2", "tier3"]
    assert report.accepted
    assert [item.action for item in report.decisions] == [
        EscalationAction.ESCALATE,
        EscalationAction.ESCALATE,
        EscalationAction.ESCALATE,
        EscalationAction.COMPLETE,
    ]
    thresholded = [
        metric
        for decision in report.decisions
        for metric in decision.metrics
        if metric.threshold is not None
    ]
    assert thresholded and all(metric.passed is True for metric in thresholded)


def test_configured_max_tier_records_higher_tiers_as_skipped() -> None:
    backend = Backend()
    report = run_tiered_evaluation(
        backend,
        TieredEvaluationConfig(max_tier=EvaluationTier.TIER1),
    )

    assert backend.calls == ["tier0_load", "tier0_numerics", "tier1"]
    assert report.accepted
    assert report.highest_executed_tier is EvaluationTier.TIER1
    assert report.decisions[1].action is EscalationAction.COMPLETE
    assert all(item.reason == "tier not configured" for item in report.decisions[2:])
    assert all(
        metric.value is None
        for decision in report.decisions[2:]
        for metric in decision.metrics
    )


def test_requested_unavailable_metric_fails_closed_and_records_reason() -> None:
    backend = Backend(perplexity=_perplexity(loss_delta=None))
    config = TieredEvaluationConfig(
        max_tier=EvaluationTier.TIER1,
        thresholds=(
            TierThreshold(
                EvaluationTier.TIER1,
                "loss_delta",
                ThresholdComparator.MAXIMUM,
                0.1,
            ),
        ),
    )
    report = run_tiered_evaluation(backend, config)

    assert not report.accepted
    metric = next(
        item for item in report.decisions[1].metrics if item.metric == "loss_delta"
    )
    assert metric.value is None
    assert metric.passed is False
    assert metric.reason == "metric unavailable"


def test_noncomparable_latency_is_rejected_with_skipped_ratio_metrics() -> None:
    backend = Backend(
        latency=LatencyComparison(False, "device contexts differ", None, None)
    )
    report = run_tiered_evaluation(backend)

    assert not report.accepted
    tier3 = report.decisions[3]
    assert tier3.action is EscalationAction.REJECT
    assert tier3.reason == "device contexts differ"
    assert all(metric.value is None for metric in tier3.metrics)
    assert all(metric.reason == "device contexts differ" for metric in tier3.metrics)


def test_threshold_contract_rejects_unknown_duplicate_and_unconfigured_targets() -> None:
    with pytest.raises(TieredEvaluationError, match="not defined"):
        TierThreshold(
            EvaluationTier.TIER2,
            "loss_delta",
            ThresholdComparator.MAXIMUM,
            1.0,
        )
    threshold = TierThreshold(
        EvaluationTier.TIER1,
        "loss_delta",
        ThresholdComparator.MAXIMUM,
        0.1,
    )
    with pytest.raises(TieredEvaluationError, match="unique"):
        TieredEvaluationConfig(thresholds=(threshold, threshold))
    with pytest.raises(TieredEvaluationError, match="above"):
        TieredEvaluationConfig(
            max_tier=EvaluationTier.TIER0,
            thresholds=(threshold,),
        )

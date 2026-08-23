"""Sequential Tier 0-3 evaluation with explicit threshold and escalation decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Protocol

from modelsurgeon.evaluation.latency import LatencyComparison
from modelsurgeon.evaluation.logit_metrics import LogitMetricResult
from modelsurgeon.evaluation.numerics import Tier0NumericsResult
from modelsurgeon.evaluation.perplexity import PerplexityResult
from modelsurgeon.evaluation.tier0 import Tier0ValidationResult

TIERED_EVALUATOR_VERSION = "1"


class TieredEvaluationError(ValueError):
    """Raised when tier configuration or evaluator outputs violate the escalation contract."""


class EvaluationTier(IntEnum):
    TIER0 = 0
    TIER1 = 1
    TIER2 = 2
    TIER3 = 3


class EscalationAction(StrEnum):
    REJECT = "reject"
    ESCALATE = "escalate"
    COMPLETE = "complete"
    SKIP = "skip"


class ThresholdComparator(StrEnum):
    MAXIMUM = "maximum"
    MINIMUM = "minimum"


_ALLOWED_METRICS: dict[EvaluationTier, frozenset[str]] = {
    EvaluationTier.TIER0: frozenset({"load_shape_forward_pass", "numerics_pass"}),
    EvaluationTier.TIER1: frozenset(
        {"mean_loss", "perplexity", "loss_delta", "perplexity_delta"}
    ),
    EvaluationTier.TIER2: frozenset(
        {"teacher_to_candidate_kl", "cosine_similarity", "top_k_agreement"}
    ),
    EvaluationTier.TIER3: frozenset(
        {"prefill_speedup_ratio", "decode_speedup_ratio"}
    ),
}


@dataclass(frozen=True, slots=True)
class TierThreshold:
    tier: EvaluationTier
    metric: str
    comparator: ThresholdComparator
    limit: float

    def __post_init__(self) -> None:
        if self.metric not in _ALLOWED_METRICS[self.tier]:
            raise TieredEvaluationError(
                f"metric {self.metric!r} is not defined for tier {int(self.tier)}"
            )
        if not math.isfinite(self.limit):
            raise TieredEvaluationError("tier thresholds must be finite")

    def to_record(self) -> dict[str, object]:
        return {
            "tier": int(self.tier),
            "metric": self.metric,
            "comparator": self.comparator.value,
            "limit": self.limit,
        }


@dataclass(frozen=True, slots=True)
class TieredEvaluationConfig:
    max_tier: EvaluationTier = EvaluationTier.TIER3
    thresholds: tuple[TierThreshold, ...] = ()

    def __post_init__(self) -> None:
        keys = tuple((item.tier, item.metric) for item in self.thresholds)
        if len(keys) != len(set(keys)):
            raise TieredEvaluationError("tier thresholds must target unique tier/metric pairs")
        if any(item.tier > self.max_tier for item in self.thresholds):
            raise TieredEvaluationError("threshold targets a tier above the configured maximum")


@dataclass(frozen=True, slots=True)
class MetricDecision:
    tier: EvaluationTier
    metric: str
    value: float | None
    threshold: TierThreshold | None
    passed: bool | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.metric not in _ALLOWED_METRICS[self.tier]:
            raise TieredEvaluationError("metric decision uses an unknown tier metric")
        if self.value is not None and not math.isfinite(self.value):
            raise TieredEvaluationError("metric decisions cannot contain non-finite values")
        if self.threshold is None:
            if self.passed is not None:
                raise TieredEvaluationError("unthresholded metrics cannot claim a threshold result")
        elif self.threshold.tier is not self.tier or self.threshold.metric != self.metric:
            raise TieredEvaluationError("metric decision threshold identity does not match")
        elif self.passed is None:
            raise TieredEvaluationError("thresholded metrics require an explicit pass state")
        if self.value is None and not self.reason:
            raise TieredEvaluationError("unavailable or skipped metrics require a reason")
        if self.value is not None and self.reason is not None:
            raise TieredEvaluationError("measured metrics cannot also be marked unavailable")

    def to_record(self) -> dict[str, object]:
        return {
            "tier": int(self.tier),
            "metric": self.metric,
            "value": self.value,
            "threshold": None if self.threshold is None else self.threshold.to_record(),
            "passed": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TierDecision:
    tier: EvaluationTier
    executed: bool
    passed: bool | None
    action: EscalationAction
    reason: str | None
    metrics: tuple[MetricDecision, ...]

    def __post_init__(self) -> None:
        if self.executed:
            if self.passed is None or self.action is EscalationAction.SKIP:
                raise TieredEvaluationError("executed tiers require pass state and non-skip action")
            if self.reason is not None and self.passed:
                raise TieredEvaluationError("passing tiers cannot carry a failure reason")
        else:
            if self.passed is not None or self.action is not EscalationAction.SKIP:
                raise TieredEvaluationError("skipped tiers require skip action and no pass state")
            if not self.reason:
                raise TieredEvaluationError("skipped tiers require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "tier": int(self.tier),
            "executed": self.executed,
            "passed": self.passed,
            "action": self.action.value,
            "reason": self.reason,
            "metrics": [item.to_record() for item in self.metrics],
        }


@dataclass(frozen=True, slots=True)
class TieredEvaluationReport:
    decisions: tuple[TierDecision, ...]
    accepted: bool
    highest_executed_tier: EvaluationTier
    version: str = TIERED_EVALUATOR_VERSION

    def __post_init__(self) -> None:
        if tuple(item.tier for item in self.decisions) != tuple(EvaluationTier):
            raise TieredEvaluationError("tiered reports must contain canonical decisions for Tier 0-3")
        executed = tuple(item.tier for item in self.decisions if item.executed)
        if not executed or executed[-1] is not self.highest_executed_tier:
            raise TieredEvaluationError("highest executed tier disagrees with decision history")
        rejected = any(item.action is EscalationAction.REJECT for item in self.decisions)
        if self.accepted == rejected:
            raise TieredEvaluationError("report acceptance disagrees with rejection history")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "accepted": self.accepted,
            "highest_executed_tier": int(self.highest_executed_tier),
            "decisions": [item.to_record() for item in self.decisions],
        }


class TieredEvaluationBackend(Protocol):
    def run_tier0_load(self) -> Tier0ValidationResult: ...

    def run_tier0_numerics(self) -> Tier0NumericsResult: ...

    def run_tier1_perplexity(self) -> PerplexityResult: ...

    def run_tier2_logit_metrics(self) -> LogitMetricResult: ...

    def run_tier3_latency(self) -> LatencyComparison: ...


def _threshold_map(
    config: TieredEvaluationConfig,
) -> dict[tuple[EvaluationTier, str], TierThreshold]:
    return {(item.tier, item.metric): item for item in config.thresholds}


def _apply_threshold(value: float, threshold: TierThreshold) -> bool:
    if threshold.comparator is ThresholdComparator.MAXIMUM:
        return value <= threshold.limit
    return value >= threshold.limit


def _metric_decision(
    tier: EvaluationTier,
    metric: str,
    value: float | None,
    thresholds: dict[tuple[EvaluationTier, str], TierThreshold],
    *,
    unavailable_reason: str | None = None,
) -> MetricDecision:
    threshold = thresholds.get((tier, metric))
    if value is None:
        return MetricDecision(
            tier,
            metric,
            None,
            threshold,
            False if threshold is not None else None,
            unavailable_reason or "metric unavailable",
        )
    passed = None if threshold is None else _apply_threshold(value, threshold)
    return MetricDecision(tier, metric, value, threshold, passed)


def _threshold_failure(metrics: tuple[MetricDecision, ...]) -> MetricDecision | None:
    return next((item for item in metrics if item.threshold is not None and not item.passed), None)


def _skip_metrics(tier: EvaluationTier, reason: str) -> tuple[MetricDecision, ...]:
    return tuple(
        MetricDecision(tier, metric, None, None, None, reason)
        for metric in sorted(_ALLOWED_METRICS[tier])
    )


def _next_action(tier: EvaluationTier, max_tier: EvaluationTier) -> EscalationAction:
    return EscalationAction.COMPLETE if tier is max_tier else EscalationAction.ESCALATE


def _skipped_tail(
    start: EvaluationTier,
    reason: str,
) -> list[TierDecision]:
    return [
        TierDecision(
            tier,
            False,
            None,
            EscalationAction.SKIP,
            reason,
            _skip_metrics(tier, reason),
        )
        for tier in EvaluationTier
        if tier >= start
    ]


def _tier_failure_reason(metric: MetricDecision) -> str:
    if metric.value is None:
        return f"required threshold metric {metric.metric} is unavailable"
    assert metric.threshold is not None
    return (
        f"metric {metric.metric}={metric.value} failed "
        f"{metric.threshold.comparator.value} threshold {metric.threshold.limit}"
    )


def run_tiered_evaluation(
    backend: TieredEvaluationBackend,
    config: TieredEvaluationConfig | None = None,
) -> TieredEvaluationReport:
    """Run configured tiers sequentially, rejecting before any unnecessary higher-tier work."""

    resolved = config or TieredEvaluationConfig()
    thresholds = _threshold_map(resolved)
    decisions: list[TierDecision] = []

    load = backend.run_tier0_load()
    load_metric = _metric_decision(
        EvaluationTier.TIER0,
        "load_shape_forward_pass",
        1.0 if load.passed else 0.0,
        thresholds,
    )
    if not load.passed:
        numerics_reason = "skipped because Tier 0 load/shape/forward validation failed"
        tier0_metrics = (
            load_metric,
            MetricDecision(
                EvaluationTier.TIER0,
                "numerics_pass",
                None,
                thresholds.get((EvaluationTier.TIER0, "numerics_pass")),
                False
                if (EvaluationTier.TIER0, "numerics_pass") in thresholds
                else None,
                numerics_reason,
            ),
        )
        decisions.append(
            TierDecision(
                EvaluationTier.TIER0,
                True,
                False,
                EscalationAction.REJECT,
                "Tier 0 load/shape/forward validation failed",
                tier0_metrics,
            )
        )
        decisions.extend(
            _skipped_tail(EvaluationTier.TIER1, "candidate rejected by Tier 0")
        )
        return TieredEvaluationReport(tuple(decisions), False, EvaluationTier.TIER0)

    numerics = backend.run_tier0_numerics()
    numerics_metric = _metric_decision(
        EvaluationTier.TIER0,
        "numerics_pass",
        1.0 if numerics.passed else 0.0,
        thresholds,
    )
    tier0_metrics = (load_metric, numerics_metric)
    tier0_threshold_failure = _threshold_failure(tier0_metrics)
    if not numerics.passed or tier0_threshold_failure is not None:
        reason = "Tier 0 numerical validation failed"
        if tier0_threshold_failure is not None:
            reason = _tier_failure_reason(tier0_threshold_failure)
        decisions.append(
            TierDecision(
                EvaluationTier.TIER0,
                True,
                False,
                EscalationAction.REJECT,
                reason,
                tier0_metrics,
            )
        )
        decisions.extend(
            _skipped_tail(EvaluationTier.TIER1, "candidate rejected by Tier 0")
        )
        return TieredEvaluationReport(tuple(decisions), False, EvaluationTier.TIER0)

    decisions.append(
        TierDecision(
            EvaluationTier.TIER0,
            True,
            True,
            _next_action(EvaluationTier.TIER0, resolved.max_tier),
            None,
            tier0_metrics,
        )
    )
    if resolved.max_tier is EvaluationTier.TIER0:
        decisions.extend(_skipped_tail(EvaluationTier.TIER1, "tier not configured"))
        return TieredEvaluationReport(tuple(decisions), True, EvaluationTier.TIER0)

    perplexity = backend.run_tier1_perplexity()
    tier1_metrics = tuple(
        _metric_decision(EvaluationTier.TIER1, name, value, thresholds)
        for name, value in (
            ("mean_loss", perplexity.mean_loss),
            ("perplexity", perplexity.perplexity),
            ("loss_delta", perplexity.loss_delta),
            ("perplexity_delta", perplexity.perplexity_delta),
        )
    )
    failure = _threshold_failure(tier1_metrics)
    if failure is not None:
        decisions.append(
            TierDecision(
                EvaluationTier.TIER1,
                True,
                False,
                EscalationAction.REJECT,
                _tier_failure_reason(failure),
                tier1_metrics,
            )
        )
        decisions.extend(
            _skipped_tail(EvaluationTier.TIER2, "candidate rejected by Tier 1")
        )
        return TieredEvaluationReport(tuple(decisions), False, EvaluationTier.TIER1)
    decisions.append(
        TierDecision(
            EvaluationTier.TIER1,
            True,
            True,
            _next_action(EvaluationTier.TIER1, resolved.max_tier),
            None,
            tier1_metrics,
        )
    )
    if resolved.max_tier is EvaluationTier.TIER1:
        decisions.extend(_skipped_tail(EvaluationTier.TIER2, "tier not configured"))
        return TieredEvaluationReport(tuple(decisions), True, EvaluationTier.TIER1)

    logits = backend.run_tier2_logit_metrics()
    tier2_metrics = tuple(
        _metric_decision(EvaluationTier.TIER2, name, value, thresholds)
        for name, value in (
            ("teacher_to_candidate_kl", logits.mean_teacher_to_candidate_kl),
            ("cosine_similarity", logits.mean_cosine_similarity),
            ("top_k_agreement", logits.mean_top_k_agreement),
        )
    )
    failure = _threshold_failure(tier2_metrics)
    if failure is not None:
        decisions.append(
            TierDecision(
                EvaluationTier.TIER2,
                True,
                False,
                EscalationAction.REJECT,
                _tier_failure_reason(failure),
                tier2_metrics,
            )
        )
        decisions.extend(
            _skipped_tail(EvaluationTier.TIER3, "candidate rejected by Tier 2")
        )
        return TieredEvaluationReport(tuple(decisions), False, EvaluationTier.TIER2)
    decisions.append(
        TierDecision(
            EvaluationTier.TIER2,
            True,
            True,
            _next_action(EvaluationTier.TIER2, resolved.max_tier),
            None,
            tier2_metrics,
        )
    )
    if resolved.max_tier is EvaluationTier.TIER2:
        decisions.extend(_skipped_tail(EvaluationTier.TIER3, "tier not configured"))
        return TieredEvaluationReport(tuple(decisions), True, EvaluationTier.TIER2)

    latency = backend.run_tier3_latency()
    tier3_reason = None if latency.comparable else latency.reason or "latency comparison invalid"
    tier3_metrics = (
        _metric_decision(
            EvaluationTier.TIER3,
            "prefill_speedup_ratio",
            latency.prefill_speedup_ratio,
            thresholds,
            unavailable_reason=tier3_reason,
        ),
        _metric_decision(
            EvaluationTier.TIER3,
            "decode_speedup_ratio",
            latency.decode_speedup_ratio,
            thresholds,
            unavailable_reason=tier3_reason,
        ),
    )
    failure = _threshold_failure(tier3_metrics)
    if not latency.comparable or failure is not None:
        reason = tier3_reason or _tier_failure_reason(failure)  # type: ignore[arg-type]
        decisions.append(
            TierDecision(
                EvaluationTier.TIER3,
                True,
                False,
                EscalationAction.REJECT,
                reason,
                tier3_metrics,
            )
        )
        return TieredEvaluationReport(tuple(decisions), False, EvaluationTier.TIER3)

    decisions.append(
        TierDecision(
            EvaluationTier.TIER3,
            True,
            True,
            EscalationAction.COMPLETE,
            None,
            tier3_metrics,
        )
    )
    return TieredEvaluationReport(tuple(decisions), True, EvaluationTier.TIER3)

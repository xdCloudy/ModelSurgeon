"""Deterministic held-out studies for ranking objective choices."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

RANKING_OBJECTIVE_SCHEMA_VERSION: Final[int] = 1
RANKING_OBJECTIVE_PROTOCOL_REVISION: Final[str] = "ranking-objectives-v1"


class RankingObjectiveError(ValueError):
    """Raised when objective study evidence or comparison inputs are unsafe."""


class RankingObjective(StrEnum):
    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    LISTWISE = "listwise"
    SEQUENCE = "sequence"


class CandidateEvidenceStatus(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RankingObjectiveError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RankingObjectiveError(f"{label} must be finite")
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RankingObjectiveError(f"{label} is required")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class RankingCandidateEvidence:
    candidate_id: str
    predicted_score: float
    utility: float | None
    violation_probability: float
    violated: bool | None
    cumulative_utility: float | None
    status: CandidateEvidenceStatus = CandidateEvidenceStatus.MEASURED

    def __post_init__(self) -> None:
        _text(self.candidate_id, "candidate ID")
        _finite(self.predicted_score, "predicted score")
        if self.utility is not None:
            _finite(self.utility, "candidate utility")
        if not 0.0 <= _finite(self.violation_probability, "violation probability") <= 1.0:
            raise RankingObjectiveError("violation probability must be within [0, 1]")
        if self.status is CandidateEvidenceStatus.MEASURED:
            if self.utility is None or self.violated is None or self.cumulative_utility is None:
                raise RankingObjectiveError("measured candidates require complete outcomes")
        elif (
            self.utility is not None
            or self.violated is not None
            or self.cumulative_utility is not None
        ):
            raise RankingObjectiveError("censored candidates cannot carry partial outcomes")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "predicted_score": self.predicted_score,
            "utility": self.utility,
            "violation_probability": self.violation_probability,
            "violated": self.violated,
            "cumulative_utility": self.cumulative_utility,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class RankingCandidateList:
    list_id: str
    parent_state_id: str
    model_revision: str
    split: str
    lineage_group_id: str
    candidates: tuple[RankingCandidateEvidence, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("list ID", self.list_id),
            ("parent state ID", self.parent_state_id),
            ("model revision", self.model_revision),
            ("split", self.split),
            ("lineage group ID", self.lineage_group_id),
        ):
            _text(value, label)
        if len(self.candidates) < 2:
            raise RankingObjectiveError("candidate lists require at least two candidates")
        ids = tuple(item.candidate_id for item in self.candidates)
        if len(ids) != len(set(ids)):
            raise RankingObjectiveError("candidate IDs must be unique within a list")

    def to_record(self) -> dict[str, object]:
        return {
            "list_id": self.list_id,
            "parent_state_id": self.parent_state_id,
            "model_revision": self.model_revision,
            "split": self.split,
            "lineage_group_id": self.lineage_group_id,
            "candidates": [item.to_record() for item in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class RankingObjectiveConfig:
    top_k: int = 3
    bootstrap_repetitions: int = 200
    confidence: float = 0.95
    seed: int = 0
    equal_training_budget_steps: int = 1_000
    inference_cost_microseconds: tuple[tuple[RankingObjective, float], ...] = (
        (RankingObjective.LISTWISE, 8.0),
        (RankingObjective.PAIRWISE, 6.0),
        (RankingObjective.POINTWISE, 4.0),
        (RankingObjective.SEQUENCE, 10.0),
    )

    def __post_init__(self) -> None:
        if (
            self.top_k <= 0
            or self.bootstrap_repetitions <= 0
            or self.equal_training_budget_steps <= 0
        ):
            raise RankingObjectiveError("ranking objective budgets must be positive")
        if not 0.5 < self.confidence < 1.0 or not math.isfinite(self.confidence):
            raise RankingObjectiveError("ranking confidence must be in (0.5, 1)")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise RankingObjectiveError("ranking objective seed must be unsigned")
        objectives = tuple(item[0] for item in self.inference_cost_microseconds)
        if len(objectives) != len(set(objectives)):
            raise RankingObjectiveError("objective costs must be unique and canonical")
        if any(
            not math.isfinite(cost) or cost <= 0.0 for _, cost in self.inference_cost_microseconds
        ):
            raise RankingObjectiveError("objective inference costs must be positive")
        object.__setattr__(
            self,
            "inference_cost_microseconds",
            tuple(sorted(self.inference_cost_microseconds, key=lambda item: item[0].value)),
        )

    def cost_for(self, objective: RankingObjective) -> float:
        for candidate, cost in self.inference_cost_microseconds:
            if candidate is objective:
                return cost
        raise RankingObjectiveError(f"missing resource cost for {objective.value}")

    def to_record(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "confidence": self.confidence,
            "seed": self.seed,
            "equal_training_budget_steps": self.equal_training_budget_steps,
            "inference_cost_microseconds": {
                objective.value: cost for objective, cost in self.inference_cost_microseconds
            },
        }


@dataclass(frozen=True, slots=True)
class RankingMetric:
    name: str
    value: float | None
    interval_low: float | None
    interval_high: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise RankingObjectiveError("ranking metrics require names")
        values = (self.value, self.interval_low, self.interval_high)
        if any(item is not None and not math.isfinite(item) for item in values):
            raise RankingObjectiveError("ranking metric values must be finite")
        if self.value is None and not self.reason:
            raise RankingObjectiveError("undefined ranking metrics require a reason")
        if (
            self.value is not None
            and self.interval_low is not None
            and self.interval_high is not None
            and (self.interval_low > self.value or self.value > self.interval_high)
        ):
            raise RankingObjectiveError("ranking metric intervals must contain the value")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "interval_low": self.interval_low,
            "interval_high": self.interval_high,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RankingObjectiveResult:
    objective: RankingObjective
    rankings: tuple[tuple[str, tuple[str, ...]], ...]
    metrics: tuple[RankingMetric, ...]
    training_budget_steps: int
    inference_cost_microseconds: float

    def __post_init__(self) -> None:
        if self.training_budget_steps <= 0 or self.inference_cost_microseconds <= 0.0:
            raise RankingObjectiveError("ranking resource costs must be positive")

    def metric(self, name: str) -> RankingMetric:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise RankingObjectiveError(f"unknown ranking metric {name}")

    def to_record(self) -> dict[str, object]:
        return {
            "objective": self.objective.value,
            "rankings": [
                {"list_id": list_id, "candidate_ids": list(candidate_ids)}
                for list_id, candidate_ids in self.rankings
            ],
            "metrics": [item.to_record() for item in self.metrics],
            "training_budget_steps": self.training_budget_steps,
            "inference_cost_microseconds": self.inference_cost_microseconds,
        }


@dataclass(frozen=True, slots=True)
class RankingObjectiveStudy:
    results: tuple[RankingObjectiveResult, ...]
    recommended_objective: RankingObjective | None
    recommendation_reason: str
    protocol_revision: str = RANKING_OBJECTIVE_PROTOCOL_REVISION
    schema_version: int = RANKING_OBJECTIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RANKING_OBJECTIVE_SCHEMA_VERSION:
            raise RankingObjectiveError("unsupported ranking objective schema")
        objectives = tuple(item.objective for item in self.results)
        if objectives != tuple(sorted(set(objectives), key=lambda item: item.value)):
            raise RankingObjectiveError("objective results must be unique and canonical")
        if not self.recommendation_reason:
            raise RankingObjectiveError("objective recommendations require a reason")

    def result_for(self, objective: RankingObjective) -> RankingObjectiveResult:
        for result in self.results:
            if result.objective is objective:
                return result
        raise RankingObjectiveError(f"objective result unavailable for {objective.value}")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": self.protocol_revision,
            "results": [item.to_record() for item in self.results],
            "recommended_objective": (
                None if self.recommended_objective is None else self.recommended_objective.value
            ),
            "recommendation_reason": self.recommendation_reason,
        }


def _rank_candidates(
    candidate_list: RankingCandidateList, objective: RankingObjective
) -> tuple[RankingCandidateEvidence, ...]:
    measured = tuple(
        item
        for item in candidate_list.candidates
        if item.status is CandidateEvidenceStatus.MEASURED
    )
    if not measured:
        return ()
    if objective is RankingObjective.SEQUENCE:
        return tuple(sorted(measured, key=lambda item: (-_cumulative(item), item.candidate_id)))
    if objective is RankingObjective.PAIRWISE:
        return tuple(sorted(measured, key=lambda item: (-item.predicted_score, item.candidate_id)))
    if objective is RankingObjective.LISTWISE:
        # Softmax normalization is list-aware but preserves deterministic utility order.
        scores = tuple(math.exp(max(-40.0, min(40.0, item.predicted_score))) for item in measured)
        return tuple(
            item
            for _, item in sorted(
                zip(scores, measured, strict=True),
                key=lambda pair: (-pair[0], pair[1].candidate_id),
            )
        )
    return tuple(sorted(measured, key=lambda item: (-item.predicted_score, item.candidate_id)))


def _utility(item: RankingCandidateEvidence) -> float:
    if item.utility is None:
        raise RankingObjectiveError("measured ranking candidates require utility")
    return item.utility


def _cumulative(item: RankingCandidateEvidence) -> float:
    if item.cumulative_utility is None:
        raise RankingObjectiveError("measured ranking candidates require cumulative utility")
    return item.cumulative_utility


def _ndcg(ranked: Sequence[RankingCandidateEvidence], top_k: int) -> float:
    if not ranked:
        return 0.0
    selected = ranked[:top_k]
    ideal = sorted(ranked, key=lambda item: (-_utility(item), item.candidate_id))[:top_k]

    def dcg(items: Sequence[RankingCandidateEvidence]) -> float:
        return math.fsum(
            (2.0 ** max(0.0, _utility(item)) - 1.0) / math.log2(index + 2)
            for index, item in enumerate(items)
        )

    ideal_score = dcg(ideal)
    return 0.0 if ideal_score == 0.0 else dcg(selected) / ideal_score


def _list_metrics(
    ranked: Sequence[RankingCandidateEvidence], top_k: int
) -> tuple[float, float, float, float, float]:
    selected = ranked[:top_k]
    if not selected:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    best = max(_utility(item) for item in ranked)
    selected_utility = math.fsum(_utility(item) for item in selected) / len(selected)
    regret = max(0.0, best - selected_utility)
    precision = math.fsum(float(not bool(item.violated)) for item in selected) / len(selected)
    violation_rate = 1.0 - precision
    brier = math.fsum(
        (item.violation_probability - float(bool(item.violated))) ** 2 for item in selected
    ) / len(selected)
    frontier = max(_cumulative(item) for item in selected)
    return regret, precision, violation_rate, brier, frontier


def _bootstrap_interval(
    values: Sequence[float], config: RankingObjectiveConfig, seed_offset: int
) -> tuple[float, float, float]:
    if not values:
        raise RankingObjectiveError("ranking metrics require measured held-out lists")
    rng = random.Random(config.seed + seed_offset)
    estimates = [math.fsum(values) / len(values)]
    for _ in range(config.bootstrap_repetitions):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(math.fsum(sample) / len(sample))
    estimates.sort()
    low_index = int((1.0 - config.confidence) / 2.0 * len(estimates))
    high_index = min(len(estimates) - 1, int((1.0 + config.confidence) / 2.0 * len(estimates)))
    return estimates[0], estimates[low_index], estimates[high_index]


def evaluate_ranking_objectives(
    candidate_lists: Sequence[RankingCandidateList],
    config: RankingObjectiveConfig | None = None,
) -> RankingObjectiveStudy:
    """Compare all objective forms on identical held-out candidate lists."""

    resolved = config or RankingObjectiveConfig()
    if not candidate_lists:
        raise RankingObjectiveError("ranking objective studies require candidate lists")
    list_ids = tuple(item.list_id for item in candidate_lists)
    if len(list_ids) != len(set(list_ids)):
        raise RankingObjectiveError("candidate list IDs must be unique")
    if any(item.split not in {"validation", "test"} for item in candidate_lists):
        raise RankingObjectiveError("objective evaluation requires held-out validation/test lists")
    results: list[RankingObjectiveResult] = []
    for objective in RankingObjective:
        ranked_lists: list[tuple[str, tuple[str, ...]]] = []
        metric_rows: list[tuple[float, float, float, float, float]] = []
        for candidate_list in candidate_lists:
            ranked = _rank_candidates(candidate_list, objective)
            if not ranked:
                continue
            ranked_lists.append(
                (candidate_list.list_id, tuple(item.candidate_id for item in ranked))
            )
            metric_rows.append(_list_metrics(ranked, min(resolved.top_k, len(ranked))))
        if not metric_rows:
            raise RankingObjectiveError(
                f"objective {objective.value} has no measured held-out lists"
            )
        ndcg_values = [
            _ndcg(_rank_candidates(candidate_list, objective), resolved.top_k)
            for candidate_list in candidate_lists
            if _rank_candidates(candidate_list, objective)
        ]
        names = (
            "ndcg",
            "regret",
            "precision_at_k",
            "violation_rate_at_k",
            "calibration_brier",
            "cumulative_frontier_utility",
        )
        columns = (
            ndcg_values,
            [row[0] for row in metric_rows],
            [row[1] for row in metric_rows],
            [row[2] for row in metric_rows],
            [row[3] for row in metric_rows],
            [row[4] for row in metric_rows],
        )
        metrics: list[RankingMetric] = []
        for index, (name, values) in enumerate(zip(names, columns, strict=True), start=1):
            mean, low, high = _bootstrap_interval(values, resolved, index * 100)
            metrics.append(RankingMetric(name, mean, low, high))
        results.append(
            RankingObjectiveResult(
                objective,
                tuple(ranked_lists),
                tuple(metrics),
                resolved.equal_training_budget_steps,
                resolved.cost_for(objective),
            )
        )
    results.sort(key=lambda item: item.objective.value)
    pointwise = next(item for item in results if item.objective is RankingObjective.POINTWISE)
    pointwise_regret = pointwise.metric("regret")
    recommendation: RankingObjective | None = None
    reason = (
        "pointwise remains the preregistered baseline; no confidence-bounded "
        "complex-objective benefit"
    )
    for result in results:
        if result.objective is RankingObjective.POINTWISE:
            continue
        regret = result.metric("regret")
        if (
            regret.interval_high is not None
            and pointwise_regret.interval_low is not None
            and regret.interval_high < pointwise_regret.interval_low
        ):
            recommendation = result.objective
            reason = (
                f"{result.objective.value} has a confidence-bounded lower held-out "
                "regret than pointwise"
            )
            break
    return RankingObjectiveStudy(tuple(results), recommendation, reason)

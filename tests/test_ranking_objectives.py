"""Tests for matched held-out ranking objective studies."""

from __future__ import annotations

import pytest

from modelsurgeon.surgeon import (
    CandidateEvidenceStatus,
    RankingCandidateEvidence,
    RankingCandidateList,
    RankingObjective,
    RankingObjectiveConfig,
    RankingObjectiveError,
    evaluate_ranking_objectives,
)


def _list(index: int) -> RankingCandidateList:
    return RankingCandidateList(
        f"list-{index}",
        f"state-{index}",
        f"model-{index}",
        "test",
        f"lineage-{index}",
        (
            RankingCandidateEvidence("safe", 0.8, 1.0, 0.1, False, 1.2),
            RankingCandidateEvidence("risky", 0.9, 1.4, 0.8, True, 1.0),
            RankingCandidateEvidence("steady", 0.7, 0.9, 0.05, False, 1.1),
            RankingCandidateEvidence(
                "unsupported", 1.0, None, 0.5, None, None, CandidateEvidenceStatus.UNSUPPORTED
            ),
        ),
    )


def test_all_objectives_share_lists_and_report_metrics_and_cost() -> None:
    study = evaluate_ranking_objectives(
        (_list(1), _list(2)),
        RankingObjectiveConfig(top_k=2, bootstrap_repetitions=20),
    )

    assert tuple(item.objective for item in study.results) == tuple(
        sorted(RankingObjective, key=lambda item: item.value)
    )
    for result in study.results:
        assert result.training_budget_steps == 1_000
        assert result.inference_cost_microseconds > 0.0
        assert result.metric("ndcg").value is not None
        assert result.metric("regret").interval_low is not None
        assert result.metric("precision_at_k").value is not None
        assert result.metric("violation_rate_at_k").value is not None
        assert result.metric("calibration_brier").value is not None
        assert result.metric("cumulative_frontier_utility").value is not None
        assert all("unsupported" not in ids for _, ids in result.rankings)
    assert study.recommended_objective is None or isinstance(
        study.recommended_objective, RankingObjective
    )


def test_invalid_objective_inputs_fail_closed() -> None:
    with pytest.raises(RankingObjectiveError, match="held-out"):
        evaluate_ranking_objectives(
            (
                RankingCandidateList(
                    "list-validation",
                    "state",
                    "model",
                    "train",
                    "lineage",
                    _list(1).candidates,
                ),
            ),
            RankingObjectiveConfig(top_k=2, bootstrap_repetitions=2),
        )

    with pytest.raises(RankingObjectiveError, match="complete outcomes"):
        RankingCandidateEvidence("bad", 0.1, 1.0, 0.5, None, 1.0)

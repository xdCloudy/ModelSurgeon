"""Tests for matched held-out ranking objective studies."""

from __future__ import annotations

from dataclasses import replace

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
        assert result.training_budget_steps is None
        assert result.inference_cost_microseconds is None
        assert result.to_record()["training_budget_steps"] is None
        assert result.to_record()["inference_cost_microseconds"] is None
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


def test_study_records_only_explicit_resource_measurements() -> None:
    costs = tuple((objective, 12.5 + index) for index, objective in enumerate(RankingObjective))
    measured_config = RankingObjectiveConfig(
        top_k=2,
        bootstrap_repetitions=4,
        equal_training_budget_steps=2_500,
        inference_cost_microseconds=costs,
    )
    study = evaluate_ranking_objectives((_list(1),), measured_config)

    for objective in RankingObjective:
        result = study.result_for(objective)
        assert result.training_budget_steps == 2_500
        assert result.inference_cost_microseconds == measured_config.cost_for(objective)

    absent = evaluate_ranking_objectives(
        (_list(2),), RankingObjectiveConfig(top_k=2, bootstrap_repetitions=4)
    )
    assert all(result.training_budget_steps is None for result in absent.results)
    assert all(result.inference_cost_microseconds is None for result in absent.results)


def test_sequence_ranking_uses_prediction_instead_of_held_out_cumulative_utility() -> None:
    study = evaluate_ranking_objectives(
        (_list(1),), RankingObjectiveConfig(top_k=2, bootstrap_repetitions=4)
    )

    # The measured cumulative utilities favor "safe", but the held-out model
    # prediction ranks "risky" first. Measured outcomes are scoring targets only.
    assert study.result_for(RankingObjective.SEQUENCE).rankings == (
        ("list-1", ("risky", "safe", "steady")),
    )


def test_objective_specific_predictions_drive_each_ranking() -> None:
    def all_scores(
        values: dict[RankingObjective, float],
    ) -> tuple[tuple[RankingObjective, float], ...]:
        return tuple((objective, values[objective]) for objective in RankingObjective)

    rows = (
        RankingCandidateEvidence(
            "candidate-a",
            0.0,
            1.0,
            0.1,
            False,
            1.0,
            objective_scores=all_scores(
                {
                    RankingObjective.POINTWISE: 0.9,
                    RankingObjective.PAIRWISE: 0.1,
                    RankingObjective.LISTWISE: 0.3,
                    RankingObjective.SEQUENCE: 0.2,
                }
            ),
        ),
        RankingCandidateEvidence(
            "candidate-b",
            0.0,
            0.8,
            0.2,
            False,
            0.8,
            objective_scores=all_scores(
                {
                    RankingObjective.POINTWISE: 0.1,
                    RankingObjective.PAIRWISE: 0.9,
                    RankingObjective.LISTWISE: 0.2,
                    RankingObjective.SEQUENCE: 0.3,
                }
            ),
        ),
        RankingCandidateEvidence(
            "candidate-c",
            0.0,
            0.7,
            0.3,
            False,
            0.7,
            objective_scores=all_scores(
                {
                    RankingObjective.POINTWISE: 0.2,
                    RankingObjective.PAIRWISE: 0.3,
                    RankingObjective.LISTWISE: 0.9,
                    RankingObjective.SEQUENCE: 0.8,
                }
            ),
        ),
    )
    study = evaluate_ranking_objectives(
        (
            RankingCandidateList(
                "list-objective-scores",
                "state",
                "model",
                "test",
                "lineage",
                rows,
            ),
        ),
        RankingObjectiveConfig(top_k=1, bootstrap_repetitions=4),
    )

    expected_first = {
        RankingObjective.POINTWISE: "candidate-a",
        RankingObjective.PAIRWISE: "candidate-b",
        RankingObjective.LISTWISE: "candidate-c",
        RankingObjective.SEQUENCE: "candidate-c",
    }
    for objective, candidate_id in expected_first.items():
        ranking = study.result_for(objective).rankings[0]
        assert ranking[0] == "list-objective-scores"
        assert ranking[1][0] == candidate_id


def test_bootstrap_resamples_lineage_groups_as_independent_units() -> None:
    first = _list(1)
    second = replace(
        _list(2),
        lineage_group_id=first.lineage_group_id,
        candidates=tuple(
            replace(candidate, utility=0.2) if candidate.candidate_id == "risky" else candidate
            for candidate in _list(2).candidates
        ),
    )
    study = evaluate_ranking_objectives(
        (first, second), RankingObjectiveConfig(top_k=1, bootstrap_repetitions=50)
    )
    regret = study.result_for(RankingObjective.POINTWISE).metric("regret")

    assert regret.value == pytest.approx(0.4)
    assert regret.interval_low == pytest.approx(regret.value)
    assert regret.interval_high == pytest.approx(regret.value)


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

    with pytest.raises(RankingObjectiveError, match="cover every ranking objective"):
        RankingCandidateEvidence(
            "incomplete-objective-scores",
            0.1,
            1.0,
            0.5,
            False,
            1.0,
            objective_scores=((RankingObjective.POINTWISE, 0.1),),
        )

    with pytest.raises(RankingObjectiveError, match="measured objective inference costs"):
        RankingObjectiveConfig(inference_cost_microseconds=((RankingObjective.POINTWISE, 0.0),))

    with pytest.raises(
        RankingObjectiveError, match="measured objective inference cost must be finite"
    ):
        RankingObjectiveConfig(inference_cost_microseconds=((RankingObjective.POINTWISE, 10**400),))

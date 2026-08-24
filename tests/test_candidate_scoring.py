from __future__ import annotations

from dataclasses import dataclass

from modelsurgeon.active_learning import ScoringCandidate, score_candidate_pool
from modelsurgeon.surgeon.calibration import PlattCalibrator


@dataclass
class _Predictor:
    column: int

    def predict(self, rows):
        return tuple(row[self.column] for row in rows)


def _candidates():
    return tuple(
        ScoringCandidate(f"cand_{index}", 1, ("a", "b"), (index / 10.0, 0.2)) for index in range(8)
    )


def _score(batch_size: int):
    return score_candidate_pool(
        _candidates(),
        utility_predictor=_Predictor(0),
        outcome_predictors={"perplexity": _Predictor(1)},
        safe_predictor=_Predictor(0),
        uncertainty_predictor=_Predictor(1),
        calibrator=PlattCalibrator(1.0, 0.0),
        expected_feature_schema_version=1,
        expected_feature_names=("a", "b"),
        model_revision="model-v1",
        tool_revision="tool-v1",
        batch_size=batch_size,
    )


def test_scores_are_stable_across_batch_sizes() -> None:
    assert _score(1).scores == _score(3).scores == _score(8).scores


def test_schema_incompatible_candidates_are_quarantined() -> None:
    candidates = (
        *_candidates(),
        ScoringCandidate("cand_bad", 2, ("a", "b"), (0.1, 0.2)),
    )

    report = score_candidate_pool(
        candidates,
        utility_predictor=_Predictor(0),
        outcome_predictors={"perplexity": _Predictor(1)},
        safe_predictor=_Predictor(0),
        uncertainty_predictor=_Predictor(1),
        calibrator=PlattCalibrator(1.0, 0.0),
        expected_feature_schema_version=1,
        expected_feature_names=("a", "b"),
        model_revision="model-v1",
        tool_revision="tool-v1",
        batch_size=4,
    )

    assert len(report.scores) == 8
    assert report.quarantined[0].candidate_id == "cand_bad"
    assert report.quarantined[0].reason == "feature-schema-version-incompatible"
    assert report.to_record()["quarantine_count"] == 1

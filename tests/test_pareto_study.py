from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.evaluation import (
    DEFAULT_PARETO_STUDY,
    ParetoStudyError,
    StudyArtifact,
    StudyMetric,
    StudyOutcome,
    render_pareto_study,
)


def test_default_study_retains_all_negative_cells_without_claims() -> None:
    study = DEFAULT_PARETO_STUDY

    assert len(study.models) == 2
    assert {model.family for model in study.models} == {"llama", "qwen"}
    assert {model.size_parameters for model in study.models} == {135_000_000, 500_000_000}
    assert len(study.points) == 48
    assert {point.outcome for point in study.points} == {StudyOutcome.UNSUPPORTED}
    assert study.frontier_point_ids == ()
    assert study.superiority_claim is None


def test_study_render_and_evidence_id_are_deterministic() -> None:
    study = DEFAULT_PARETO_STUDY
    assert render_pareto_study(study) == render_pareto_study(study)
    payload = json.loads(render_pareto_study())
    assert payload["study_id"] == study.study_id
    evidence_path = (
        Path(__file__).parents[1]
        / "docs"
        / "research"
        / "v1.1-competitor-pareto-study-v1.json"
    )
    assert json.loads(evidence_path.read_text(encoding="utf-8"))["study_id"] == study.study_id


def test_measured_metrics_require_intervals_and_deployable_artifact() -> None:
    with pytest.raises(ParetoStudyError, match="confidence"):
        StudyMetric("quality", "nats/token", "lower", value=1.0)
    with pytest.raises(ParetoStudyError, match="generation"):
        StudyArtifact("a" * 64, 100, True, False)

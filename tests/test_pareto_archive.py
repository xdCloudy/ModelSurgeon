from pathlib import Path

import pytest

from modelsurgeon.config import ObjectiveDirection, ObjectiveNormalization, OptimizeMetric
from modelsurgeon.search.objectives import ObjectiveSet, ObjectiveTerm
from modelsurgeon.search.pareto import (
    ParetoArchive,
    ParetoArchiveError,
    ParetoCandidate,
    ParetoObjectiveValue,
    conservatively_dominates,
)

OBJECTIVES = ObjectiveSet(
    (
        ObjectiveTerm(
            OptimizeMetric.QUALITY,
            ObjectiveDirection.MAXIMIZE,
            normalization=ObjectiveNormalization.IDENTITY,
        ),
        ObjectiveTerm(
            OptimizeMetric.PARAMETER_COUNT,
            ObjectiveDirection.MINIMIZE,
            normalization=ObjectiveNormalization.IDENTITY,
        ),
    )
)


def _candidate(
    candidate_id: str,
    quality: tuple[float, float, float] | None,
    parameters: tuple[float, float, float] | None,
) -> ParetoCandidate:
    values = []
    if quality is not None:
        values.append(ParetoObjectiveValue(OptimizeMetric.QUALITY, *quality))
    if parameters is not None:
        values.append(ParetoObjectiveValue(OptimizeMetric.PARAMETER_COUNT, *parameters))
    return ParetoCandidate(candidate_id, tuple(values), {"lineage": "root"})


def test_conservative_interval_dominance_requires_worst_case_separation() -> None:
    strong = _candidate("strong", (0.99, 0.985, 0.995), (70, 68, 72))
    weak = _candidate("weak", (0.97, 0.96, 0.98), (80, 78, 82))
    overlap = _candidate("overlap", (0.985, 0.975, 0.995), (71, 69, 73))

    assert conservatively_dominates(strong, weak, OBJECTIVES) is True
    assert conservatively_dominates(strong, overlap, OBJECTIVES) is False
    assert conservatively_dominates(strong, _candidate("pending", None, None), OBJECTIVES) is False


def test_archive_persists_frontier_and_completes_missing_objectives(tmp_path: Path) -> None:
    path = tmp_path / "pareto.sqlite3"
    weak = _candidate("weak", (0.97, 0.96, 0.98), (80, 78, 82))
    pending = _candidate("pending", None, (75, 74, 76))
    strong = _candidate("strong", (0.99, 0.985, 0.995), (70, 68, 72))
    with ParetoArchive(path, OBJECTIVES) as archive:
        assert archive.put(weak).entry.on_frontier is True
        assert archive.put(pending).entry.on_frontier is True
        result = archive.put(strong)
        assert result.frontier_candidate_ids == ("pending", "strong")
        completed = _candidate("pending", (0.95, 0.94, 0.96), (75, 74, 76))
        assert archive.put(completed).frontier_candidate_ids == ("strong",)

    with ParetoArchive(path, OBJECTIVES) as resumed:
        assert [entry.candidate.candidate_id for entry in resumed.entries(frontier_only=True)] == [
            "strong"
        ]
        assert len(resumed.entries()) == 3


def test_archive_rejects_conflicts_and_objective_identity_changes(tmp_path: Path) -> None:
    path = tmp_path / "pareto.sqlite3"
    candidate = _candidate("same", (0.9, 0.9, 0.9), None)
    with ParetoArchive(path, OBJECTIVES) as archive:
        archive.put(candidate)
        with pytest.raises(ParetoArchiveError, match="only fill"):
            archive.put(_candidate("same", (0.8, 0.8, 0.8), None))
    changed = ObjectiveSet((OBJECTIVES.terms[0],))
    with pytest.raises(ParetoArchiveError, match="does not match"):
        ParetoArchive(path, changed)

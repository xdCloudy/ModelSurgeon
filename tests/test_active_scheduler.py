from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.active_learning import (
    AcquisitionCandidate,
    ActiveScheduleError,
    ActiveScheduleStore,
    ScheduledEvaluationState,
    acquire_candidates,
)


def _acquisition(count: int = 3):
    candidates = tuple(
        AcquisitionCandidate(f"cand_{index}", float(index), 0.8, float(index), float(index))
        for index in range(count)
    )
    return acquire_candidates(candidates, count)


def test_partial_schedule_resumes_without_reselection_and_links_results(tmp_path: Path) -> None:
    store = ActiveScheduleStore(tmp_path / "schedule.json")
    created = store.create_or_resume(_acquisition(), tool_revision="tool-v1")
    first = created.entries[0]
    store.mark_scheduled(first.candidate_id, "exp_" + "a" * 64)
    store.mark_completed(first.candidate_id, "example-1")

    resumed = store.create_or_resume(_acquisition(), tool_revision="tool-v1")

    assert resumed.selection_digest == created.selection_digest
    assert resumed.entries[0].state is ScheduledEvaluationState.COMPLETED
    assert resumed.entries[0].experiment_id == "exp_" + "a" * 64
    assert resumed.entries[0].example_id == "example-1"
    assert len(resumed.pending) == 2
    assert resumed.entries[0].reason and resumed.entries[0].propensity == 1.0


def test_resume_rejects_reselection(tmp_path: Path) -> None:
    store = ActiveScheduleStore(tmp_path / "schedule.json")
    store.create_or_resume(_acquisition(3), tool_revision="tool-v1")

    with pytest.raises(ActiveScheduleError, match="cannot reselect"):
        store.create_or_resume(_acquisition(2), tool_revision="tool-v1")

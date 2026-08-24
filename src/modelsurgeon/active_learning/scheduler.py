"""Durable active-learning schedules that resume without reselection."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Final

from modelsurgeon.experiments.identity import canonical_identity_json

from .acquisition import AcquisitionReport

ACTIVE_SCHEDULE_SCHEMA_VERSION: Final[int] = 1


class ActiveScheduleError(ValueError):
    """Raised when a persisted selection or experiment link is incompatible."""


class ScheduledEvaluationState(StrEnum):
    SELECTED = "selected"
    SCHEDULED = "scheduled"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ScheduledEvaluation:
    candidate_id: str
    rank: int
    reason: str
    propensity: float
    policy_score: float
    state: ScheduledEvaluationState = ScheduledEvaluationState.SELECTED
    experiment_id: str | None = None
    example_id: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "reason": self.reason,
            "propensity": self.propensity,
            "policy_score": self.policy_score,
            "state": self.state.value,
            "experiment_id": self.experiment_id,
            "example_id": self.example_id,
        }


@dataclass(frozen=True, slots=True)
class ActiveEvaluationSchedule:
    selection_digest: str
    tool_revision: str
    entries: tuple[ScheduledEvaluation, ...]
    schema_version: int = ACTIVE_SCHEDULE_SCHEMA_VERSION

    @property
    def pending(self) -> tuple[ScheduledEvaluation, ...]:
        return tuple(
            item for item in self.entries if item.state is not ScheduledEvaluationState.COMPLETED
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selection_digest": self.selection_digest,
            "tool_revision": self.tool_revision,
            "entries": [item.to_record() for item in self.entries],
        }


class ActiveScheduleStore:
    """Persist selection once and append experiment/example links via atomic replacement."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def create_or_resume(
        self, acquisition: AcquisitionReport, *, tool_revision: str
    ) -> ActiveEvaluationSchedule:
        if not tool_revision:
            raise ActiveScheduleError("active schedule tool revision is required")
        selection_record = acquisition.to_record()
        digest = (
            "sha256:"
            + hashlib.sha256(canonical_identity_json(selection_record).encode("utf-8")).hexdigest()
        )
        if self.path.exists():
            schedule = self.load()
            if schedule.selection_digest != digest:
                raise ActiveScheduleError(
                    "persisted active schedule selection differs; resume cannot reselect"
                )
            if schedule.tool_revision != tool_revision:
                raise ActiveScheduleError("persisted active schedule tool revision changed")
            return schedule
        entries = tuple(
            ScheduledEvaluation(
                item.candidate_id,
                item.rank,
                item.reason.value,
                item.propensity,
                item.policy_score,
            )
            for item in acquisition.selections
        )
        schedule = ActiveEvaluationSchedule(digest, tool_revision, entries)
        self._publish(schedule)
        return schedule

    def mark_scheduled(self, candidate_id: str, experiment_id: str) -> ActiveEvaluationSchedule:
        if not experiment_id.startswith("exp_"):
            raise ActiveScheduleError("scheduled evaluations require canonical experiment IDs")
        schedule = self.load()
        entry = _entry(schedule, candidate_id)
        if entry.state is ScheduledEvaluationState.COMPLETED:
            raise ActiveScheduleError("completed active evaluation cannot be rescheduled")
        if entry.experiment_id is not None and entry.experiment_id != experiment_id:
            raise ActiveScheduleError("active evaluation is already linked to another experiment")
        updated = replace(
            entry,
            state=ScheduledEvaluationState.SCHEDULED,
            experiment_id=experiment_id,
        )
        return self._replace(schedule, updated)

    def mark_completed(self, candidate_id: str, example_id: str) -> ActiveEvaluationSchedule:
        if not example_id:
            raise ActiveScheduleError("completed active evaluations require example IDs")
        schedule = self.load()
        entry = _entry(schedule, candidate_id)
        if entry.state is not ScheduledEvaluationState.SCHEDULED or entry.experiment_id is None:
            raise ActiveScheduleError("active evaluation must be scheduled before completion")
        updated = replace(
            entry,
            state=ScheduledEvaluationState.COMPLETED,
            example_id=example_id,
        )
        return self._replace(schedule, updated)

    def load(self) -> ActiveEvaluationSchedule:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ActiveScheduleError("active schedule is unreadable") from error
        if (
            not isinstance(raw, Mapping)
            or raw.get("schema_version") != ACTIVE_SCHEDULE_SCHEMA_VERSION
        ):
            raise ActiveScheduleError("active schedule schema is incompatible")
        digest = raw.get("selection_digest")
        tool_revision = raw.get("tool_revision")
        entries_raw = raw.get("entries")
        if (
            not isinstance(digest, str)
            or not isinstance(tool_revision, str)
            or not isinstance(entries_raw, list)
        ):
            raise ActiveScheduleError("active schedule fields are malformed")
        entries: list[ScheduledEvaluation] = []
        try:
            for item in entries_raw:
                if not isinstance(item, Mapping):
                    raise ActiveScheduleError("active schedule entry is malformed")
                entries.append(
                    ScheduledEvaluation(
                        str(item["candidate_id"]),
                        int(item["rank"]),
                        str(item["reason"]),
                        float(item["propensity"]),
                        float(item["policy_score"]),
                        ScheduledEvaluationState(str(item["state"])),
                        None if item.get("experiment_id") is None else str(item["experiment_id"]),
                        None if item.get("example_id") is None else str(item["example_id"]),
                    )
                )
        except (KeyError, TypeError, ValueError) as error:
            raise ActiveScheduleError("active schedule entry fields are malformed") from error
        return ActiveEvaluationSchedule(digest, tool_revision, tuple(entries))

    def _replace(
        self, schedule: ActiveEvaluationSchedule, updated: ScheduledEvaluation
    ) -> ActiveEvaluationSchedule:
        entries = tuple(
            updated if item.candidate_id == updated.candidate_id else item
            for item in schedule.entries
        )
        result = replace(schedule, entries=entries)
        self._publish(result)
        return result

    def _publish(self, schedule: ActiveEvaluationSchedule) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(schedule.to_record(), indent=2, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)


def _entry(schedule: ActiveEvaluationSchedule, candidate_id: str) -> ScheduledEvaluation:
    matches = tuple(item for item in schedule.entries if item.candidate_id == candidate_id)
    if len(matches) != 1:
        raise ActiveScheduleError(f"active schedule has no unique candidate {candidate_id!r}")
    return matches[0]

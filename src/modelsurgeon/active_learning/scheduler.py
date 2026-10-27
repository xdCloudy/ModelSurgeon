"""Durable active-learning schedules that resume without reselection."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Final

from modelsurgeon.experiments.identity import canonical_identity_json

from .acquisition import AcquisitionReason, AcquisitionReport

ACTIVE_SCHEDULE_SCHEMA_VERSION: Final[int] = 1
MAX_ACTIVE_SCHEDULE_ENTRIES: Final[int] = 100_000
MAX_ACTIVE_SCHEDULE_BYTES: Final[int] = 64 * 1024 * 1024


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

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.startswith("cand_"):
            raise ActiveScheduleError("scheduled evaluations require canonical candidate IDs")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ActiveScheduleError("active schedule ranks must be positive integers")
        if not isinstance(self.reason, str) or self.reason not in {
            reason.value for reason in AcquisitionReason
        }:
            raise ActiveScheduleError("active schedule selection reason is unsupported")
        if not isinstance(self.state, ScheduledEvaluationState):
            raise ActiveScheduleError("active schedule state is unsupported")
        if (
            isinstance(self.propensity, bool)
            or not isinstance(self.propensity, (int, float))
            or not math.isfinite(self.propensity)
            or not 0.0 < self.propensity <= 1.0
        ):
            raise ActiveScheduleError("active schedule propensity must be finite and in (0, 1]")
        if (
            isinstance(self.policy_score, bool)
            or not isinstance(self.policy_score, (int, float))
            or not math.isfinite(self.policy_score)
        ):
            raise ActiveScheduleError("active schedule policy score must be finite")
        if self.state is ScheduledEvaluationState.SELECTED:
            if self.experiment_id is not None or self.example_id is not None:
                raise ActiveScheduleError("selected evaluations cannot have experiment links")
        elif self.state is ScheduledEvaluationState.SCHEDULED:
            if (
                not isinstance(self.experiment_id, str)
                or not self.experiment_id.startswith("exp_")
                or self.example_id is not None
            ):
                raise ActiveScheduleError("scheduled evaluations require one experiment link")
        elif self.state is ScheduledEvaluationState.COMPLETED and (
            not isinstance(self.experiment_id, str)
            or not self.experiment_id.startswith("exp_")
            or not isinstance(self.example_id, str)
            or not self.example_id
        ):
            raise ActiveScheduleError("completed evaluations require experiment and example links")

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

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != ACTIVE_SCHEDULE_SCHEMA_VERSION
        ):
            raise ActiveScheduleError("active schedule schema is incompatible")
        if (
            not isinstance(self.selection_digest, str)
            or len(self.selection_digest) != 71
            or not self.selection_digest.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in self.selection_digest[7:])
        ):
            raise ActiveScheduleError("active schedule selection digest is malformed")
        if not isinstance(self.tool_revision, str) or not self.tool_revision.strip():
            raise ActiveScheduleError("active schedule tool revision is required")
        if not isinstance(self.entries, tuple) or not all(
            isinstance(item, ScheduledEvaluation) for item in self.entries
        ):
            raise ActiveScheduleError("active schedule entries are malformed")
        candidate_ids = tuple(item.candidate_id for item in self.entries)
        ranks = tuple(item.rank for item in self.entries)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ActiveScheduleError("active schedule candidate IDs must be unique")
        if ranks != tuple(range(1, len(self.entries) + 1)):
            raise ActiveScheduleError("active schedule ranks must be contiguous and ordered")

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
        if len(acquisition.selections) > MAX_ACTIVE_SCHEDULE_ENTRIES:
            raise ActiveScheduleError(
                f"active schedule exceeds the {MAX_ACTIVE_SCHEDULE_ENTRIES}-entry limit"
            )
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
            with self.path.open("rb") as stream:
                payload = stream.read(MAX_ACTIVE_SCHEDULE_BYTES + 1)
        except OSError as error:
            raise ActiveScheduleError("active schedule is unreadable") from error
        if len(payload) > MAX_ACTIVE_SCHEDULE_BYTES:
            raise ActiveScheduleError(
                f"active schedule exceeds the {MAX_ACTIVE_SCHEDULE_BYTES}-byte limit"
            )
        try:
            raw = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
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
        if len(entries_raw) > MAX_ACTIVE_SCHEDULE_ENTRIES:
            raise ActiveScheduleError(
                f"active schedule exceeds the {MAX_ACTIVE_SCHEDULE_ENTRIES}-entry limit"
            )
        entries: list[ScheduledEvaluation] = []
        try:
            for item in entries_raw:
                if not isinstance(item, Mapping):
                    raise ActiveScheduleError("active schedule entry is malformed")
                candidate_id = item.get("candidate_id")
                rank = item.get("rank")
                reason = item.get("reason")
                propensity = item.get("propensity")
                policy_score = item.get("policy_score")
                state = item.get("state")
                experiment_id = item.get("experiment_id")
                example_id = item.get("example_id")
                if (
                    not isinstance(candidate_id, str)
                    or isinstance(rank, bool)
                    or not isinstance(rank, int)
                    or not isinstance(reason, str)
                    or isinstance(propensity, bool)
                    or not isinstance(propensity, (int, float))
                    or isinstance(policy_score, bool)
                    or not isinstance(policy_score, (int, float))
                    or not isinstance(state, str)
                    or (experiment_id is not None and not isinstance(experiment_id, str))
                    or (example_id is not None and not isinstance(example_id, str))
                ):
                    raise ActiveScheduleError("active schedule entry fields are malformed")
                entries.append(
                    ScheduledEvaluation(
                        candidate_id,
                        rank,
                        reason,
                        float(propensity),
                        float(policy_score),
                        ScheduledEvaluationState(state),
                        experiment_id,
                        example_id,
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
        if len(schedule.entries) > MAX_ACTIVE_SCHEDULE_ENTRIES:
            raise ActiveScheduleError(
                f"active schedule exceeds the {MAX_ACTIVE_SCHEDULE_ENTRIES}-entry limit"
            )
        payload = (json.dumps(schedule.to_record(), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
        if len(payload) > MAX_ACTIVE_SCHEDULE_BYTES:
            raise ActiveScheduleError(
                f"active schedule exceeds the {MAX_ACTIVE_SCHEDULE_BYTES}-byte limit"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)


def _entry(schedule: ActiveEvaluationSchedule, candidate_id: str) -> ScheduledEvaluation:
    matches = tuple(item for item in schedule.entries if item.candidate_id == candidate_id)
    if len(matches) != 1:
        raise ActiveScheduleError(f"active schedule has no unique candidate {candidate_id!r}")
    return matches[0]

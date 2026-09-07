"""Crash-consistent, privacy-safe progress and resume contracts."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from modelsurgeon.experiments.identity import canonical_identity_json

PROGRESS_SCHEMA_VERSION = 1
_SECRET_KEY = re.compile(r"(token|secret|password|api[_-]?key|credential)", re.IGNORECASE)
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class ProgressError(ValueError):
    """Raised when progress state cannot safely advance or resume."""


class ProgressStage(StrEnum):
    DOWNLOAD = "download"
    CALIBRATION = "calibration"
    SEARCH = "search"
    SURGERY = "surgery"
    REPAIR = "repair"
    EVALUATION = "evaluation"
    PUBLICATION = "publication"


class ProgressStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ProgressOutcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """One ordered machine-readable event."""

    campaign_id: str
    sequence: int
    event_type: str
    stage: ProgressStage | None
    status: ProgressStatus
    completed: int
    total: int | None
    elapsed_seconds: float
    next_work: tuple[str, ...] = ()
    detail: str | None = None
    emitted_at: float | None = None

    def __post_init__(self) -> None:
        if not self.campaign_id.strip() or not self.event_type.strip():
            raise ProgressError("progress events require campaign and event identifiers")
        if self.sequence < 0 or self.completed < 0:
            raise ProgressError("progress sequence and completed work cannot be negative")
        if self.total is not None and (self.total <= 0 or self.completed > self.total):
            raise ProgressError("progress totals must be positive and bound completed work")
        if self.elapsed_seconds < 0:
            raise ProgressError("elapsed progress cannot be negative")
        if self.next_work != tuple(dict.fromkeys(self.next_work)):
            raise ProgressError("next work must be unique and ordered")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "stage": None if self.stage is None else self.stage.value,
            "status": self.status.value,
            "completed": self.completed,
            "total": self.total,
            "elapsed_seconds": self.elapsed_seconds,
            "next_work": list(self.next_work),
            "detail": self.detail,
            "emitted_at": self.emitted_at,
        }


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """Durable state used to resume a campaign after interruption."""

    campaign_id: str
    status: ProgressStatus
    outcome: ProgressOutcome
    stages: tuple[ProgressStage, ...]
    completed_stages: tuple[ProgressStage, ...]
    next_work: tuple[str, ...]
    completed: int
    total: int | None
    elapsed_seconds: float
    eta_seconds: float | None
    eta_uncertainty: str
    accepted_artifact_digest: str | None
    source_artifact_digest: str | None
    event_sequence: int

    def __post_init__(self) -> None:
        if not self.campaign_id.strip() or not self.stages:
            raise ProgressError("progress snapshots require a campaign and stages")
        if self.stages != tuple(dict.fromkeys(self.stages)):
            raise ProgressError("progress stages must be unique and ordered")
        if not set(self.completed_stages).issubset(self.stages):
            raise ProgressError("completed stages must belong to the campaign")
        if self.completed < 0 or (self.total is not None and self.completed > self.total):
            raise ProgressError("snapshot work bounds are invalid")
        if self.total is not None and self.total <= 0:
            raise ProgressError("snapshot total must be positive when present")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "status": self.status.value,
            "outcome": self.outcome.value,
            "stages": [item.value for item in self.stages],
            "completed_stages": [item.value for item in self.completed_stages],
            "next_work": list(self.next_work),
            "completed": self.completed,
            "total": self.total,
            "elapsed_seconds": self.elapsed_seconds,
            "eta_seconds": self.eta_seconds,
            "eta_uncertainty": self.eta_uncertainty,
            "accepted_artifact_digest": self.accepted_artifact_digest,
            "source_artifact_digest": self.source_artifact_digest,
            "event_sequence": self.event_sequence,
        }


@dataclass(frozen=True, slots=True)
class ProgressDiagnosticBundle:
    """Privacy-safe troubleshooting record."""

    campaign_id: str
    snapshot: ProgressSnapshot
    events: tuple[ProgressEvent, ...]
    environment: Mapping[str, object]
    redactions: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "progress_diagnostic_bundle",
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "snapshot": self.snapshot.to_record(),
            "events": [item.to_record() for item in self.events],
            "environment": dict(self.environment),
            "redactions": list(self.redactions),
        }


def redact_diagnostics(value: object, *, _path: str = "root") -> tuple[object, tuple[str, ...]]:
    """Redact secrets and machine-local paths recursively."""

    redactions: list[str] = []
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            if _SECRET_KEY.search(name):
                result[name] = "<redacted-secret>"
                redactions.append(f"{_path}.{name}")
            else:
                child, child_redactions = redact_diagnostics(item, _path=f"{_path}.{name}")
                result[name] = child
                redactions.extend(child_redactions)
        return result, tuple(redactions)
    if isinstance(value, (list, tuple)):
        values: list[object] = []
        for index, item in enumerate(value):
            child, child_redactions = redact_diagnostics(item, _path=f"{_path}[{index}]")
            values.append(child)
            redactions.extend(child_redactions)
        return values, tuple(redactions)
    if isinstance(value, str) and (_WINDOWS_PATH.match(value) or value.startswith("/")):
        return "<redacted-path>", (_path,)
    return value, ()


def _stage(value: object) -> ProgressStage:
    try:
        return ProgressStage(str(value))
    except ValueError as error:
        raise ProgressError(f"unknown progress stage: {value!r}") from error


def _status(value: object) -> ProgressStatus:
    try:
        return ProgressStatus(str(value))
    except ValueError as error:
        raise ProgressError(f"unknown progress status: {value!r}") from error


class ProgressStore:
    """Atomic JSON state store with idempotent event replay."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().absolute().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, campaign_id: str) -> Path:
        if not campaign_id or any(character in campaign_id for character in "\\/.."):
            raise ProgressError("campaign ID contains unsafe path characters")
        return self.root / f"{campaign_id}.json"

    def _load(self, campaign_id: str) -> tuple[ProgressSnapshot, tuple[ProgressEvent, ...]]:
        try:
            raw = json.loads(self._path(campaign_id).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProgressError("progress state is missing or corrupt") from error
        if not isinstance(raw, Mapping) or raw.get("schema_version") != PROGRESS_SCHEMA_VERSION:
            raise ProgressError("progress state schema version is unsupported")
        snapshot_raw = raw.get("snapshot")
        events_raw = raw.get("events")
        if not isinstance(snapshot_raw, Mapping) or not isinstance(events_raw, list):
            raise ProgressError("progress state sections are malformed")
        snapshot = self._snapshot_from_record(snapshot_raw)
        events: list[ProgressEvent] = []
        for item in events_raw:
            if not isinstance(item, Mapping):
                raise ProgressError("progress event is malformed")
            stage_raw = item.get("stage")
            next_work_raw = item.get("next_work", [])
            if not isinstance(next_work_raw, list) or not all(
                isinstance(value, str) for value in next_work_raw
            ):
                raise ProgressError("progress next work is malformed")
            events.append(
                ProgressEvent(
                    str(item.get("campaign_id")),
                    int(item.get("sequence", -1)),
                    str(item.get("event_type")),
                    None if stage_raw is None else _stage(stage_raw),
                    _status(item.get("status")),
                    int(item.get("completed", -1)),
                    None if item.get("total") is None else int(item["total"]),
                    float(item.get("elapsed_seconds", -1)),
                    tuple(next_work_raw),
                    None if item.get("detail") is None else str(item["detail"]),
                    None if item.get("emitted_at") is None else float(item["emitted_at"]),
                )
            )
        if tuple(event.sequence for event in events) != tuple(range(len(events))):
            raise ProgressError("progress events must have contiguous sequence numbers")
        return snapshot, tuple(events)

    def _snapshot_from_record(self, raw: Mapping[str, object]) -> ProgressSnapshot:
        stages_raw = raw.get("stages")
        completed_raw = raw.get("completed_stages")
        next_work_raw = raw.get("next_work", [])
        if not isinstance(stages_raw, list) or not isinstance(completed_raw, list):
            raise ProgressError("progress snapshot stages are malformed")
        if not isinstance(next_work_raw, list) or not all(
            isinstance(value, str) for value in next_work_raw
        ):
            raise ProgressError("progress snapshot next work is malformed")
        completed = raw.get("completed", -1)
        total = raw.get("total")
        elapsed = raw.get("elapsed_seconds", -1)
        eta = raw.get("eta_seconds")
        event_sequence = raw.get("event_sequence", -1)
        if (
            not isinstance(completed, int)
            or isinstance(completed, bool)
            or (total is not None and (not isinstance(total, int) or isinstance(total, bool)))
            or not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or (eta is not None and (not isinstance(eta, (int, float)) or isinstance(eta, bool)))
            or not isinstance(event_sequence, int)
            or isinstance(event_sequence, bool)
        ):
            raise ProgressError("progress snapshot numeric fields are malformed")
        return ProgressSnapshot(
            str(raw.get("campaign_id")),
            _status(raw.get("status")),
            ProgressOutcome(str(raw.get("outcome"))),
            tuple(_stage(value) for value in stages_raw),
            tuple(_stage(value) for value in completed_raw),
            tuple(next_work_raw),
            completed,
            total,
            float(elapsed),
            None if eta is None else float(eta),
            str(raw.get("eta_uncertainty")),
            None
            if raw.get("accepted_artifact_digest") is None
            else str(raw["accepted_artifact_digest"]),
            None
            if raw.get("source_artifact_digest") is None
            else str(raw["source_artifact_digest"]),
            event_sequence,
        )

    def _save(self, snapshot: ProgressSnapshot, events: Sequence[ProgressEvent]) -> None:
        destination = self._path(snapshot.campaign_id)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
        payload = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "snapshot": snapshot.to_record(),
            "events": [item.to_record() for item in events],
        }
        temporary.write_text(canonical_identity_json(payload), encoding="utf-8")
        try:
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def start(
        self,
        campaign_id: str,
        stages: Sequence[ProgressStage],
        *,
        source_artifact_digest: str | None = None,
        total: int | None = None,
    ) -> ProgressSnapshot:
        if not stages or len(stages) != len(set(stages)):
            raise ProgressError("campaign requires unique ordered stages")
        if self._path(campaign_id).exists():
            raise ProgressError("campaign already exists; use resume or inspect")
        snapshot = ProgressSnapshot(
            campaign_id,
            ProgressStatus.RUNNING,
            ProgressOutcome.UNKNOWN,
            tuple(stages),
            (),
            (stages[0].value,),
            0,
            total,
            0.0,
            None,
            "no completed work has been observed",
            None,
            source_artifact_digest,
            0,
        )
        event = ProgressEvent(
            campaign_id,
            0,
            "campaign_started",
            stages[0],
            ProgressStatus.RUNNING,
            0,
            total,
            0.0,
            (stages[0].value,),
            emitted_at=time.time(),
        )
        self._save(snapshot, (event,))
        return snapshot

    def snapshot(self, campaign_id: str) -> ProgressSnapshot:
        return self._load(campaign_id)[0]

    def events(self, campaign_id: str) -> tuple[ProgressEvent, ...]:
        return self._load(campaign_id)[1]

    def record(
        self,
        campaign_id: str,
        *,
        event_type: str,
        stage: ProgressStage | None,
        status: ProgressStatus,
        completed: int,
        total: int | None,
        elapsed_seconds: float,
        next_work: Sequence[str] = (),
        detail: str | None = None,
        outcome: ProgressOutcome | None = None,
        accepted_artifact_digest: str | None = None,
    ) -> ProgressSnapshot:
        current, events = self._load(campaign_id)
        if current.status is ProgressStatus.COMPLETED:
            raise ProgressError("terminal campaign state cannot accept new progress")
        if stage is not None and stage not in current.stages:
            raise ProgressError("event stage is not part of the campaign")
        completed_stages = list(current.completed_stages)
        if status is ProgressStatus.COMPLETED and stage is not None:
            if stage in completed_stages:
                return current
            completed_stages.append(stage)
        next_stage = tuple(next_work)
        remaining = [item.value for item in current.stages if item not in completed_stages]
        if status is ProgressStatus.COMPLETED and not next_stage:
            next_stage = tuple(remaining)
        snapshot_status = status
        if status is ProgressStatus.COMPLETED and remaining:
            snapshot_status = ProgressStatus.RUNNING
        eta = None
        uncertainty = "insufficient observations for ETA"
        if total is not None and completed < total and elapsed_seconds > 0 and completed > 0:
            eta = (total - completed) * elapsed_seconds / completed
            uncertainty = "linear estimate; stage duration and worker rate may change"
        elif total is not None and completed >= total:
            eta = 0.0
            uncertainty = "bounded by completed total"
        next_outcome = outcome or current.outcome
        if status is ProgressStatus.FAILED:
            next_outcome = ProgressOutcome.FAILED
        elif status is ProgressStatus.COMPLETED and not remaining:
            next_outcome = outcome or ProgressOutcome.SUPPORTED
        snapshot = ProgressSnapshot(
            campaign_id,
            snapshot_status,
            next_outcome,
            current.stages,
            tuple(completed_stages),
            next_stage,
            completed,
            total,
            elapsed_seconds,
            eta,
            uncertainty,
            accepted_artifact_digest or current.accepted_artifact_digest,
            current.source_artifact_digest,
            len(events),
        )
        event = ProgressEvent(
            campaign_id,
            len(events),
            event_type,
            stage,
            status,
            completed,
            total,
            elapsed_seconds,
            next_stage,
            detail,
            time.time(),
        )
        self._save(snapshot, (*events, event))
        return snapshot

    def pause(self, campaign_id: str, *, detail: str = "paused by operator") -> ProgressSnapshot:
        return self.record(
            campaign_id,
            event_type="campaign_paused",
            stage=None,
            status=ProgressStatus.PAUSED,
            completed=self.snapshot(campaign_id).completed,
            total=self.snapshot(campaign_id).total,
            elapsed_seconds=self.snapshot(campaign_id).elapsed_seconds,
            next_work=self.snapshot(campaign_id).next_work,
            detail=detail,
        )

    def cancel(
        self, campaign_id: str, *, detail: str = "cancelled by operator"
    ) -> ProgressSnapshot:
        current = self.snapshot(campaign_id)
        return self.record(
            campaign_id,
            event_type="campaign_cancelled",
            stage=None,
            status=ProgressStatus.CANCELLED,
            completed=current.completed,
            total=current.total,
            elapsed_seconds=current.elapsed_seconds,
            next_work=current.next_work,
            detail=detail,
        )

    def resume(self, campaign_id: str) -> ProgressSnapshot:
        current = self.snapshot(campaign_id)
        if current.status is ProgressStatus.COMPLETED:
            return current
        return self.record(
            campaign_id,
            event_type="campaign_resumed",
            stage=None,
            status=ProgressStatus.RUNNING,
            completed=current.completed,
            total=current.total,
            elapsed_seconds=current.elapsed_seconds,
            next_work=current.next_work,
            detail="resumed from last atomically accepted state",
        )

    def diagnostics(
        self,
        campaign_id: str,
        *,
        environment: Mapping[str, object] | None = None,
    ) -> ProgressDiagnosticBundle:
        snapshot, events = self._load(campaign_id)
        redacted, redactions = redact_diagnostics({} if environment is None else environment)
        if not isinstance(redacted, Mapping):
            raise ProgressError("redaction produced an invalid environment record")
        return ProgressDiagnosticBundle(campaign_id, snapshot, events, redacted, redactions)

    def write_diagnostics(
        self,
        campaign_id: str,
        destination: str | Path,
        *,
        environment: Mapping[str, object] | None = None,
    ) -> Path:
        bundle = self.diagnostics(campaign_id, environment=environment)
        path = Path(destination)
        if path.exists():
            raise ProgressError(f"refusing to overwrite diagnostic bundle: {path}")
        path.write_text(
            canonical_identity_json(bundle.to_record()) + "\n", encoding="utf-8", newline="\n"
        )
        return path


__all__ = [
    "PROGRESS_SCHEMA_VERSION",
    "ProgressDiagnosticBundle",
    "ProgressError",
    "ProgressEvent",
    "ProgressOutcome",
    "ProgressSnapshot",
    "ProgressStage",
    "ProgressStatus",
    "ProgressStore",
    "redact_diagnostics",
]

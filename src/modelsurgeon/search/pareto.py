"""Persistent conservative Pareto frontier with noisy and missing objectives."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from modelsurgeon.config import ObjectiveDirection, OptimizeMetric
from modelsurgeon.search.objectives import ObjectiveSet

PARETO_ARCHIVE_SCHEMA_VERSION = 1


class ParetoArchiveError(ValueError):
    """Raised when archive identity or candidate evidence is inconsistent."""


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ParetoArchiveError("Pareto records must be canonical JSON values") from error


@dataclass(frozen=True, slots=True)
class ParetoObjectiveValue:
    metric: OptimizeMetric
    estimate: float
    confidence_low: float | None = None
    confidence_high: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.estimate):
            raise ParetoArchiveError("objective estimates must be finite")
        has_low = self.confidence_low is not None
        has_high = self.confidence_high is not None
        if has_low != has_high:
            raise ParetoArchiveError("objective confidence bounds must be both present or absent")
        if has_low:
            assert self.confidence_low is not None and self.confidence_high is not None
            if (
                not math.isfinite(self.confidence_low)
                or not math.isfinite(self.confidence_high)
                or not self.confidence_low <= self.estimate <= self.confidence_high
            ):
                raise ParetoArchiveError("objective confidence bounds must contain the estimate")

    @property
    def bounds(self) -> tuple[float, float]:
        return (
            self.estimate if self.confidence_low is None else self.confidence_low,
            self.estimate if self.confidence_high is None else self.confidence_high,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric.value,
            "estimate": self.estimate,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
        }


@dataclass(frozen=True, slots=True)
class ParetoCandidate:
    candidate_id: str
    objectives: tuple[ParetoObjectiveValue, ...]
    payload: dict[str, object]

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ParetoArchiveError("Pareto candidates require a non-empty ID")
        metrics = [value.metric for value in self.objectives]
        if len(metrics) != len(set(metrics)):
            raise ParetoArchiveError("candidate objective metrics must be unique")
        object.__setattr__(
            self,
            "objectives",
            tuple(sorted(self.objectives, key=lambda value: value.metric.value)),
        )
        _canonical(self.payload)

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "objectives": [value.to_record() for value in self.objectives],
            "payload": self.payload,
        }


@dataclass(frozen=True, slots=True)
class ParetoArchiveEntry:
    candidate: ParetoCandidate
    on_frontier: bool
    insertion_sequence: int

    def to_record(self) -> dict[str, object]:
        return {
            **self.candidate.to_record(),
            "on_frontier": self.on_frontier,
            "insertion_sequence": self.insertion_sequence,
        }


@dataclass(frozen=True, slots=True)
class ParetoInsertResult:
    entry: ParetoArchiveEntry
    frontier_candidate_ids: tuple[str, ...]


def _objectives_by_metric(
    candidate: ParetoCandidate,
) -> dict[OptimizeMetric, ParetoObjectiveValue]:
    return {value.metric: value for value in candidate.objectives}


def conservatively_dominates(
    left: ParetoCandidate,
    right: ParetoCandidate,
    objectives: ObjectiveSet,
) -> bool:
    """Return whether left is no worse at worst-case bounds and strictly better once."""

    left_values = _objectives_by_metric(left)
    right_values = _objectives_by_metric(right)
    strictly_better = False
    for term in objectives.terms:
        left_value = left_values.get(term.metric)
        right_value = right_values.get(term.metric)
        if left_value is None or right_value is None:
            return False
        left_low, left_high = left_value.bounds
        right_low, right_high = right_value.bounds
        if term.direction is ObjectiveDirection.MAXIMIZE:
            if left_low < right_high:
                return False
            strictly_better = strictly_better or left_low > right_high
        else:
            if left_high > right_low:
                return False
            strictly_better = strictly_better or left_high < right_low
    return strictly_better


class ParetoArchive:
    """SQLite-backed frontier whose objective definition is immutable."""

    def __init__(self, path: Path, objectives: ObjectiveSet) -> None:
        self.path = path
        self.objectives = objectives
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS pareto_metadata (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                schema_version INTEGER NOT NULL,
                objective_set_id TEXT NOT NULL,
                objective_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pareto_candidates (
                candidate_id TEXT PRIMARY KEY,
                objective_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                on_frontier INTEGER NOT NULL CHECK(on_frontier IN (0, 1)),
                insertion_sequence INTEGER NOT NULL UNIQUE
            );
            """
        )
        existing = self._connection.execute(
            "SELECT schema_version, objective_set_id, objective_json FROM pareto_metadata "
            "WHERE singleton = 1"
        ).fetchone()
        objective_json = _canonical(objectives.to_record())
        if existing is None:
            self._connection.execute(
                "INSERT INTO pareto_metadata VALUES (1, ?, ?, ?)",
                (PARETO_ARCHIVE_SCHEMA_VERSION, objectives.objective_set_id, objective_json),
            )
            self._connection.commit()
        elif (
            int(existing["schema_version"]) != PARETO_ARCHIVE_SCHEMA_VERSION
            or str(existing["objective_set_id"]) != objectives.objective_set_id
            or str(existing["objective_json"]) != objective_json
        ):
            self._connection.close()
            raise ParetoArchiveError("archive objective definition or schema does not match")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ParetoArchive:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _candidate_from_row(self, row: sqlite3.Row) -> ParetoCandidate:
        raw_objectives = json.loads(str(row["objective_json"]))
        values = tuple(
            ParetoObjectiveValue(
                OptimizeMetric(item["metric"]),
                float(item["estimate"]),
                None if item["confidence_low"] is None else float(item["confidence_low"]),
                None if item["confidence_high"] is None else float(item["confidence_high"]),
            )
            for item in raw_objectives
        )
        payload = json.loads(str(row["payload_json"]))
        if not isinstance(payload, dict):
            raise ParetoArchiveError("stored Pareto payload is not an object")
        return ParetoCandidate(str(row["candidate_id"]), values, payload)

    def entries(self, *, frontier_only: bool = False) -> tuple[ParetoArchiveEntry, ...]:
        where = "WHERE on_frontier = 1" if frontier_only else ""
        rows = self._connection.execute(
            f"SELECT * FROM pareto_candidates {where} ORDER BY candidate_id"
        ).fetchall()
        return tuple(
            ParetoArchiveEntry(
                self._candidate_from_row(row),
                bool(row["on_frontier"]),
                int(row["insertion_sequence"]),
            )
            for row in rows
        )

    def _validate_metrics(self, candidate: ParetoCandidate) -> None:
        allowed = {term.metric for term in self.objectives.terms}
        unknown = {value.metric for value in candidate.objectives} - allowed
        if unknown:
            raise ParetoArchiveError(
                f"candidate contains objectives outside archive definition: "
                f"{sorted(metric.value for metric in unknown)}"
            )

    def _recompute_frontier(self) -> None:
        entries = self.entries()
        for entry in entries:
            dominated = any(
                other.candidate.candidate_id != entry.candidate.candidate_id
                and conservatively_dominates(
                    other.candidate,
                    entry.candidate,
                    self.objectives,
                )
                for other in entries
            )
            self._connection.execute(
                "UPDATE pareto_candidates SET on_frontier = ? WHERE candidate_id = ?",
                (0 if dominated else 1, entry.candidate.candidate_id),
            )

    def put(self, candidate: ParetoCandidate) -> ParetoInsertResult:
        """Insert or complete a candidate and atomically recompute the frontier."""

        self._validate_metrics(candidate)
        objective_json = _canonical([value.to_record() for value in candidate.objectives])
        payload_json = _canonical(candidate.payload)
        with self._connection:
            existing = self._connection.execute(
                "SELECT * FROM pareto_candidates WHERE candidate_id = ?",
                (candidate.candidate_id,),
            ).fetchone()
            if existing is None:
                sequence = int(
                    self._connection.execute(
                        "SELECT COALESCE(MAX(insertion_sequence), 0) + 1 FROM pareto_candidates"
                    ).fetchone()[0]
                )
                self._connection.execute(
                    "INSERT INTO pareto_candidates VALUES (?, ?, ?, 1, ?)",
                    (candidate.candidate_id, objective_json, payload_json, sequence),
                )
            else:
                previous = self._candidate_from_row(existing)
                previous_values = _objectives_by_metric(previous)
                current_values = _objectives_by_metric(candidate)
                if (
                    previous.payload != candidate.payload
                    or not previous_values.keys() <= current_values.keys()
                    or any(
                        current_values[metric] != value for metric, value in previous_values.items()
                    )
                ):
                    raise ParetoArchiveError(
                        "candidate updates may only fill previously missing objectives"
                    )
                self._connection.execute(
                    "UPDATE pareto_candidates SET objective_json = ? WHERE candidate_id = ?",
                    (objective_json, candidate.candidate_id),
                )
            self._recompute_frontier()
        matching = next(
            entry
            for entry in self.entries()
            if entry.candidate.candidate_id == candidate.candidate_id
        )
        frontier_ids = tuple(
            entry.candidate.candidate_id for entry in self.entries(frontier_only=True)
        )
        return ParetoInsertResult(matching, frontier_ids)

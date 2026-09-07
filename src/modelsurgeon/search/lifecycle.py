"""Resumable lifecycle for architecture-level search candidates."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from modelsurgeon.experiments.identity import canonical_identity_json

ARCHITECTURE_LIFECYCLE_SCHEMA_VERSION = 1


class ArchitectureLifecycleError(RuntimeError):
    """Raised when a lifecycle transition would lose search or lineage evidence."""


class LifecycleStage(StrEnum):
    PREDICTED = "predicted"
    RESERVED = "reserved"
    MATERIALIZED = "materialized"
    EVALUATED = "evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FAILED = "failed"


class EvaluationStatus(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    FAILED = "failed"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArchitectureLifecycleError(f"{label} is required")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class LifecycleEvidence:
    evaluation_id: str
    status: EvaluationStatus
    passed: bool
    reason: str
    arrival_cursor: int

    def __post_init__(self) -> None:
        if not self.evaluation_id.startswith("evaluation_"):
            raise ArchitectureLifecycleError("evidence requires a canonical evaluation ID")
        _text(self.reason, "evidence reason")
        if self.arrival_cursor < 0:
            raise ArchitectureLifecycleError("evidence arrival cursor cannot be negative")
        if self.status is not EvaluationStatus.MEASURED and self.passed:
            raise ArchitectureLifecycleError(
                "unsupported, unknown, and failed evidence cannot pass"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "evaluation_id": self.evaluation_id,
            "status": self.status.value,
            "passed": self.passed,
            "reason": self.reason,
            "arrival_cursor": self.arrival_cursor,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureLifecycleCandidate:
    candidate_id: str
    architecture_id: str
    equivalence_id: str
    state_id: str
    parent_checkpoint_id: str
    predicted_only: bool = True
    stage: LifecycleStage = LifecycleStage.PREDICTED
    evidence: LifecycleEvidence | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("architecture_candidate_"):
            raise ArchitectureLifecycleError(
                "lifecycle candidates require architecture candidate IDs"
            )
        if not self.architecture_id.startswith("architecture_"):
            raise ArchitectureLifecycleError("architecture identities must be canonical")
        if not self.equivalence_id.startswith("equivalence_"):
            raise ArchitectureLifecycleError("equivalence identities must be canonical")
        if not self.state_id.startswith("state_"):
            raise ArchitectureLifecycleError("lifecycle candidates require state IDs")
        if not self.parent_checkpoint_id.startswith("checkpoint_"):
            raise ArchitectureLifecycleError("lifecycle candidates require accepted parents")
        if self.stage is LifecycleStage.ACCEPTED and self.predicted_only:
            raise ArchitectureLifecycleError("predicted-only candidates cannot be accepted")
        if self.stage is LifecycleStage.ACCEPTED and (
            self.evidence is None
            or self.evidence.status is not EvaluationStatus.MEASURED
            or not self.evidence.passed
        ):
            raise ArchitectureLifecycleError(
                "accepted candidates require passing measured evidence"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "architecture_id": self.architecture_id,
            "equivalence_id": self.equivalence_id,
            "state_id": self.state_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "predicted_only": self.predicted_only,
            "stage": self.stage.value,
            "evidence": None if self.evidence is None else self.evidence.to_record(),
        }


@dataclass(frozen=True, slots=True)
class LifecycleBudget:
    evaluation_limit: int
    evaluations_reserved: int = 0
    charged_architecture_ids: tuple[str, ...] = ()
    charged_equivalence_ids: tuple[str, ...] = ()
    disk_bytes_used: int = 0

    def __post_init__(self) -> None:
        if self.evaluation_limit <= 0:
            raise ArchitectureLifecycleError("lifecycle evaluation limit must be positive")
        if not 0 <= self.evaluations_reserved <= self.evaluation_limit:
            raise ArchitectureLifecycleError("lifecycle reservation count exceeds budget")
        for label, values, prefix in (
            ("architecture", self.charged_architecture_ids, "architecture_"),
            ("equivalence", self.charged_equivalence_ids, "equivalence_"),
        ):
            if values != tuple(sorted(set(values))) or any(
                not value.startswith(prefix) for value in values
            ):
                raise ArchitectureLifecycleError(f"charged {label} identities must be canonical")
        if len(self.charged_architecture_ids) != self.evaluations_reserved:
            raise ArchitectureLifecycleError(
                "budget reservations and charged architectures disagree"
            )
        if self.disk_bytes_used < 0:
            raise ArchitectureLifecycleError("lifecycle disk usage cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "evaluation_limit": self.evaluation_limit,
            "evaluations_reserved": self.evaluations_reserved,
            "charged_architecture_ids": list(self.charged_architecture_ids),
            "charged_equivalence_ids": list(self.charged_equivalence_ids),
            "disk_bytes_used": self.disk_bytes_used,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureSearchSnapshot:
    search_id: str
    generation: int
    candidates: tuple[ArchitectureLifecycleCandidate, ...]
    frontier_candidate_ids: tuple[str, ...]
    lineage_checkpoint_ids: tuple[str, ...]
    budget: LifecycleBudget
    decision_cursor: int
    evidence_arrival_cursor: int
    schema_version: int = ARCHITECTURE_LIFECYCLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.search_id.startswith("search_") or self.generation < 0:
            raise ArchitectureLifecycleError("search identity or generation is invalid")
        if self.schema_version != ARCHITECTURE_LIFECYCLE_SCHEMA_VERSION:
            raise ArchitectureLifecycleError("unsupported architecture lifecycle schema")
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ArchitectureLifecycleError("lifecycle candidates must be unique and canonical")
        if self.frontier_candidate_ids != tuple(sorted(set(self.frontier_candidate_ids))):
            raise ArchitectureLifecycleError("frontier candidate IDs must be unique and canonical")
        if not set(self.frontier_candidate_ids) <= set(candidate_ids):
            raise ArchitectureLifecycleError("frontier references an unknown candidate")
        if any(
            self._candidate(candidate_id).stage is not LifecycleStage.ACCEPTED
            for candidate_id in self.frontier_candidate_ids
        ):
            raise ArchitectureLifecycleError("frontier can contain accepted candidates only")
        if self.lineage_checkpoint_ids != tuple(sorted(set(self.lineage_checkpoint_ids))):
            raise ArchitectureLifecycleError("lineage checkpoint IDs must be unique and canonical")
        if not self.lineage_checkpoint_ids or any(
            not value.startswith("checkpoint_") for value in self.lineage_checkpoint_ids
        ):
            raise ArchitectureLifecycleError("lifecycle requires accepted checkpoint lineage")
        if any(
            candidate.parent_checkpoint_id not in self.lineage_checkpoint_ids
            for candidate in self.candidates
        ):
            raise ArchitectureLifecycleError("candidate parent is outside accepted lineage")
        if self.decision_cursor < 0 or self.evidence_arrival_cursor < 0:
            raise ArchitectureLifecycleError("lifecycle cursors cannot be negative")

    def _candidate(self, candidate_id: str) -> ArchitectureLifecycleCandidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise ArchitectureLifecycleError("candidate is not in this search")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "search_id": self.search_id,
            "generation": self.generation,
            "candidates": [item.to_record() for item in self.candidates],
            "frontier_candidate_ids": list(self.frontier_candidate_ids),
            "lineage_checkpoint_ids": list(self.lineage_checkpoint_ids),
            "budget": self.budget.to_record(),
            "decision_cursor": self.decision_cursor,
            "evidence_arrival_cursor": self.evidence_arrival_cursor,
        }


class ArchitectureSearchLifecycle:
    """Immutable transition facade for predicted-to-accepted architecture states."""

    def __init__(self, snapshot: ArchitectureSearchSnapshot) -> None:
        self.snapshot = snapshot

    @classmethod
    def create(
        cls,
        search_id: str,
        candidates: tuple[ArchitectureLifecycleCandidate, ...],
        root_checkpoint_id: str,
        evaluation_limit: int,
    ) -> ArchitectureSearchLifecycle:
        canonical = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        return cls(
            ArchitectureSearchSnapshot(
                search_id,
                0,
                canonical,
                (),
                (root_checkpoint_id,),
                LifecycleBudget(evaluation_limit),
                0,
                0,
            )
        )

    def _replace_candidate(
        self,
        candidate_id: str,
        stage: LifecycleStage,
        evidence: LifecycleEvidence | None = None,
    ) -> ArchitectureSearchLifecycle:
        candidates = tuple(
            replace(
                candidate,
                stage=stage,
                evidence=candidate.evidence if evidence is None else evidence,
            )
            if candidate.candidate_id == candidate_id
            else candidate
            for candidate in self.snapshot.candidates
        )
        if all(candidate.candidate_id != candidate_id for candidate in self.snapshot.candidates):
            raise ArchitectureLifecycleError("candidate is not in this search")
        return ArchitectureSearchLifecycle(
            replace(self.snapshot, generation=self.snapshot.generation + 1, candidates=candidates)
        )

    def next_decision_ids(self, limit: int) -> tuple[str, ...]:
        if limit <= 0:
            raise ArchitectureLifecycleError("decision batch limit must be positive")
        ordered = tuple(
            candidate.candidate_id
            for candidate in self.snapshot.candidates[self.snapshot.decision_cursor :]
            if candidate.stage is LifecycleStage.PREDICTED
        )
        return ordered[:limit]

    def reserve(self, candidate_ids: tuple[str, ...]) -> ArchitectureSearchLifecycle:
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise ArchitectureLifecycleError("reservation IDs must be unique and canonical")
        budget = self.snapshot.budget
        candidates = list(self.snapshot.candidates)
        for candidate_id in candidate_ids:
            candidate = next(
                (item for item in candidates if item.candidate_id == candidate_id), None
            )
            if candidate is None:
                raise ArchitectureLifecycleError("candidate is not in this search")
            if candidate.stage is not LifecycleStage.PREDICTED:
                raise ArchitectureLifecycleError("only predicted candidates can be reserved")
            if candidate.architecture_id in budget.charged_architecture_ids:
                raise ArchitectureLifecycleError("duplicate architecture cannot be charged twice")
            if candidate.equivalence_id in budget.charged_equivalence_ids:
                raise ArchitectureLifecycleError("equivalent path cannot be charged twice")
            if budget.evaluations_reserved >= budget.evaluation_limit:
                raise ArchitectureLifecycleError(
                    "architecture search evaluation budget is exhausted"
                )
            budget = replace(
                budget,
                evaluations_reserved=budget.evaluations_reserved + 1,
                charged_architecture_ids=tuple(
                    sorted((*budget.charged_architecture_ids, candidate.architecture_id))
                ),
                charged_equivalence_ids=tuple(
                    sorted((*budget.charged_equivalence_ids, candidate.equivalence_id))
                ),
            )
            candidates[candidates.index(candidate)] = replace(
                candidate, stage=LifecycleStage.RESERVED
            )
        return ArchitectureSearchLifecycle(
            replace(
                self.snapshot,
                generation=self.snapshot.generation + 1,
                candidates=tuple(candidates),
                budget=budget,
                decision_cursor=self.snapshot.decision_cursor + len(candidate_ids),
            )
        )

    def materialize(self, candidate_id: str) -> ArchitectureSearchLifecycle:
        candidate = self.snapshot._candidate(candidate_id)
        if candidate.stage is not LifecycleStage.RESERVED:
            raise ArchitectureLifecycleError("only reserved candidates can materialize")
        return self._replace_candidate(candidate_id, LifecycleStage.MATERIALIZED)

    def evaluate(
        self, candidate_id: str, evidence: LifecycleEvidence
    ) -> ArchitectureSearchLifecycle:
        candidate = self.snapshot._candidate(candidate_id)
        if candidate.stage is not LifecycleStage.MATERIALIZED:
            raise ArchitectureLifecycleError("only materialized candidates can be evaluated")
        return self._replace_candidate(
            candidate_id,
            LifecycleStage.EVALUATED,
            evidence,
        )

    def accept(self, candidate_id: str, checkpoint_id: str) -> ArchitectureSearchLifecycle:
        candidate = self.snapshot._candidate(candidate_id)
        if candidate.stage is not LifecycleStage.EVALUATED or candidate.evidence is None:
            raise ArchitectureLifecycleError("only evaluated candidates can be accepted")
        if candidate.predicted_only:
            raise ArchitectureLifecycleError("predicted-only state cannot become accepted")
        if (
            candidate.evidence.status is not EvaluationStatus.MEASURED
            or not candidate.evidence.passed
        ):
            raise ArchitectureLifecycleError("only passing measured evidence can be accepted")
        if not checkpoint_id.startswith("checkpoint_"):
            raise ArchitectureLifecycleError("accepted checkpoints must be canonical")
        return ArchitectureSearchLifecycle(
            replace(
                self._replace_candidate(candidate_id, LifecycleStage.ACCEPTED).snapshot,
                frontier_candidate_ids=tuple(
                    sorted((*self.snapshot.frontier_candidate_ids, candidate_id))
                ),
                lineage_checkpoint_ids=tuple(
                    sorted((*self.snapshot.lineage_checkpoint_ids, checkpoint_id))
                ),
            )
        )

    def reject(self, candidate_id: str) -> ArchitectureSearchLifecycle:
        candidate = self.snapshot._candidate(candidate_id)
        if candidate.stage is not LifecycleStage.EVALUATED:
            raise ArchitectureLifecycleError("only evaluated candidates can be rejected")
        return self._replace_candidate(candidate_id, LifecycleStage.REJECTED)

    def fail(self, candidate_id: str, reason: str) -> ArchitectureSearchLifecycle:
        candidate = self.snapshot._candidate(candidate_id)
        if candidate.stage not in {
            LifecycleStage.RESERVED,
            LifecycleStage.MATERIALIZED,
            LifecycleStage.EVALUATED,
        }:
            raise ArchitectureLifecycleError("candidate stage cannot be marked failed")
        _text(reason, "failure reason")
        evidence = candidate.evidence or LifecycleEvidence(
            "evaluation_failure",
            EvaluationStatus.FAILED,
            False,
            reason,
            self.snapshot.evidence_arrival_cursor,
        )
        return self._replace_candidate(candidate_id, LifecycleStage.FAILED, evidence)

    def advance_evidence_cursor(self, cursor: int) -> ArchitectureSearchLifecycle:
        if cursor < self.snapshot.evidence_arrival_cursor:
            raise ArchitectureLifecycleError("evidence cursor cannot move backwards")
        return ArchitectureSearchLifecycle(
            replace(
                self.snapshot,
                generation=self.snapshot.generation + 1,
                evidence_arrival_cursor=cursor,
            )
        )


class ArchitectureSearchResumeStore:
    """Atomic SQLite persistence for architecture lifecycle snapshots."""

    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS lifecycle_snapshots "
            "(search_id TEXT NOT NULL, generation INTEGER NOT NULL, payload TEXT NOT NULL, "
            "digest TEXT NOT NULL, PRIMARY KEY(search_id, generation))"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS lifecycle_heads "
            "(search_id TEXT PRIMARY KEY, generation INTEGER NOT NULL)"
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ArchitectureSearchResumeStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def save(self, snapshot: ArchitectureSearchSnapshot, expected_generation: int | None) -> None:
        payload = canonical_identity_json(snapshot.to_record())
        digest = hashlib.sha256(payload.encode()).hexdigest()
        row = self._connection.execute(
            "SELECT generation FROM lifecycle_heads WHERE search_id = ?", (snapshot.search_id,)
        ).fetchone()
        current = None if row is None else int(row[0])
        if current != expected_generation:
            raise ArchitectureLifecycleError("stale lifecycle snapshot generation")
        required = 0 if current is None else current + 1
        if snapshot.generation != required:
            raise ArchitectureLifecycleError(f"next lifecycle generation must be {required}")
        with self._connection:
            self._connection.execute(
                "INSERT INTO lifecycle_snapshots VALUES (?, ?, ?, ?)",
                (snapshot.search_id, snapshot.generation, payload, digest),
            )
            if current is None:
                self._connection.execute(
                    "INSERT INTO lifecycle_heads VALUES (?, ?)",
                    (snapshot.search_id, snapshot.generation),
                )
            else:
                self._connection.execute(
                    "UPDATE lifecycle_heads SET generation = ? WHERE search_id = ?",
                    (snapshot.generation, snapshot.search_id),
                )

    def load_latest(self, search_id: str) -> ArchitectureSearchSnapshot:
        row = self._connection.execute(
            "SELECT s.payload, s.digest FROM lifecycle_heads h "
            "JOIN lifecycle_snapshots s ON s.search_id = h.search_id "
            "AND s.generation = h.generation "
            "WHERE h.search_id = ?",
            (search_id,),
        ).fetchone()
        if row is None:
            raise ArchitectureLifecycleError("search has no lifecycle snapshot")
        payload, digest = str(row[0]), str(row[1])
        if hashlib.sha256(payload.encode()).hexdigest() != digest:
            raise ArchitectureLifecycleError("lifecycle snapshot checksum mismatch")
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ArchitectureLifecycleError("lifecycle snapshot JSON is corrupt") from error
        return _snapshot_from_record(raw)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArchitectureLifecycleError(f"stored {label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ArchitectureLifecycleError(f"stored {label} must be text")
    return value


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArchitectureLifecycleError(f"stored {label} must be an integer")
    return value


def _exact(value: dict[str, object], keys: set[str], label: str) -> dict[str, object]:
    if set(value) != keys:
        raise ArchitectureLifecycleError(f"stored {label} has unknown or missing fields")
    return value


def _evidence_from_record(value: object) -> LifecycleEvidence | None:
    if value is None:
        return None
    record = _exact(
        _mapping(value, "evidence"),
        {"evaluation_id", "status", "passed", "reason", "arrival_cursor"},
        "evidence",
    )
    passed = record["passed"]
    if not isinstance(passed, bool):
        raise ArchitectureLifecycleError("stored evidence pass flag must be boolean")
    return LifecycleEvidence(
        _string(record["evaluation_id"], "evaluation ID"),
        EvaluationStatus(_string(record["status"], "evaluation status")),
        passed,
        _string(record["reason"], "evidence reason"),
        _integer(record["arrival_cursor"], "evidence arrival cursor"),
    )


def _candidate_from_record(value: object) -> ArchitectureLifecycleCandidate:
    record = _exact(
        _mapping(value, "candidate"),
        {
            "candidate_id",
            "architecture_id",
            "equivalence_id",
            "state_id",
            "parent_checkpoint_id",
            "predicted_only",
            "stage",
            "evidence",
        },
        "candidate",
    )
    predicted_only = record["predicted_only"]
    if not isinstance(predicted_only, bool):
        raise ArchitectureLifecycleError("stored predicted-only flag must be boolean")
    return ArchitectureLifecycleCandidate(
        _string(record["candidate_id"], "candidate ID"),
        _string(record["architecture_id"], "architecture ID"),
        _string(record["equivalence_id"], "equivalence ID"),
        _string(record["state_id"], "state ID"),
        _string(record["parent_checkpoint_id"], "parent checkpoint ID"),
        predicted_only,
        LifecycleStage(_string(record["stage"], "candidate stage")),
        _evidence_from_record(record["evidence"]),
    )


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ArchitectureLifecycleError(f"stored {label} must be a string list")
    return tuple(value)


def _budget_from_record(value: object) -> LifecycleBudget:
    record = _exact(
        _mapping(value, "budget"),
        {
            "evaluation_limit",
            "evaluations_reserved",
            "charged_architecture_ids",
            "charged_equivalence_ids",
            "disk_bytes_used",
        },
        "budget",
    )
    return LifecycleBudget(
        _integer(record["evaluation_limit"], "evaluation limit"),
        _integer(record["evaluations_reserved"], "reserved evaluations"),
        _string_tuple(record["charged_architecture_ids"], "charged architectures"),
        _string_tuple(record["charged_equivalence_ids"], "charged equivalences"),
        _integer(record["disk_bytes_used"], "disk bytes used"),
    )


def _snapshot_from_record(value: object) -> ArchitectureSearchSnapshot:
    record = _exact(
        _mapping(value, "lifecycle snapshot"),
        {
            "schema_version",
            "search_id",
            "generation",
            "candidates",
            "frontier_candidate_ids",
            "lineage_checkpoint_ids",
            "budget",
            "decision_cursor",
            "evidence_arrival_cursor",
        },
        "lifecycle snapshot",
    )
    raw_candidates = record["candidates"]
    if not isinstance(raw_candidates, list):
        raise ArchitectureLifecycleError("stored lifecycle candidates must be a list")
    return ArchitectureSearchSnapshot(
        _string(record["search_id"], "search ID"),
        _integer(record["generation"], "generation"),
        tuple(_candidate_from_record(item) for item in raw_candidates),
        _string_tuple(record["frontier_candidate_ids"], "frontier candidates"),
        _string_tuple(record["lineage_checkpoint_ids"], "lineage checkpoints"),
        _budget_from_record(record["budget"]),
        _integer(record["decision_cursor"], "decision cursor"),
        _integer(record["evidence_arrival_cursor"], "evidence cursor"),
        _integer(record["schema_version"], "schema version"),
    )

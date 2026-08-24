"""Atomic, integrity-checked iterative search resume snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.search.policies import SearchPolicyState

SEARCH_RESUME_SCHEMA_VERSION = 1


class SearchResumeError(RuntimeError):
    """Raised when persisted search state is stale, corrupt, or inconsistent."""


@dataclass(frozen=True, slots=True)
class SearchRngState:
    seed: int
    decision_index: int
    tie_namespace_version: int = 1

    def __post_init__(self) -> None:
        if self.seed < 0 or self.decision_index < 0 or self.tie_namespace_version != 1:
            raise SearchResumeError("search RNG state is invalid or unsupported")

    def to_record(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "decision_index": self.decision_index,
            "tie_namespace_version": self.tie_namespace_version,
        }


@dataclass(frozen=True, slots=True)
class SearchBudgetSnapshot:
    evaluation_limit: int
    evaluations_reserved: int
    gpu_seconds_used: float = 0.0
    disk_bytes_used: int = 0

    def __post_init__(self) -> None:
        if (
            self.evaluation_limit <= 0
            or self.evaluations_reserved < 0
            or self.evaluations_reserved > self.evaluation_limit
            or not math.isfinite(self.gpu_seconds_used)
            or self.gpu_seconds_used < 0
            or self.disk_bytes_used < 0
        ):
            raise SearchResumeError("search budget snapshot is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "evaluation_limit": self.evaluation_limit,
            "evaluations_reserved": self.evaluations_reserved,
            "gpu_seconds_used": self.gpu_seconds_used,
            "disk_bytes_used": self.disk_bytes_used,
        }


@dataclass(frozen=True, slots=True)
class PendingSearchEvaluation:
    candidate_id: str
    candidate_state_id: str
    parent_checkpoint_id: str

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("candidate_"):
            raise SearchResumeError("pending evaluation requires a candidate ID")
        if not self.candidate_state_id.startswith("state_"):
            raise SearchResumeError("pending evaluation requires a candidate state ID")
        if not self.parent_checkpoint_id.startswith("checkpoint_"):
            raise SearchResumeError("pending evaluation requires an accepted parent checkpoint")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_state_id": self.candidate_state_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
        }


@dataclass(frozen=True, slots=True)
class SearchResumeSnapshot:
    search_id: str
    generation: int
    policy_state: SearchPolicyState
    rng_state: SearchRngState
    frontier_checkpoint_ids: tuple[str, ...]
    lineage_checkpoint_ids: tuple[str, ...]
    budget: SearchBudgetSnapshot
    pending_evaluations: tuple[PendingSearchEvaluation, ...]
    evidence_arrival_cursor: int
    schema_version: int = SEARCH_RESUME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEARCH_RESUME_SCHEMA_VERSION:
            raise SearchResumeError("unsupported search resume schema")
        if not self.search_id.startswith("search_") or self.generation < 0:
            raise SearchResumeError("search snapshot identity or generation is invalid")
        for name, values in (
            ("frontier", self.frontier_checkpoint_ids),
            ("lineage", self.lineage_checkpoint_ids),
        ):
            if values != tuple(sorted(set(values))) or any(
                not value.startswith("checkpoint_") for value in values
            ):
                raise SearchResumeError(f"{name} checkpoint IDs must be canonical")
        if not set(self.frontier_checkpoint_ids) <= set(self.lineage_checkpoint_ids):
            raise SearchResumeError("search frontier must be a subset of accepted lineage")
        if not self.lineage_checkpoint_ids:
            raise SearchResumeError("search resume requires at least one accepted checkpoint")
        pending_ids = [item.candidate_id for item in self.pending_evaluations]
        if len(pending_ids) != len(set(pending_ids)):
            raise SearchResumeError("pending evaluation candidate IDs must be unique")
        if not set(pending_ids) <= set(self.policy_state.selected_candidate_ids):
            raise SearchResumeError("pending evaluations must have reserved policy selections")
        if any(
            item.parent_checkpoint_id not in self.lineage_checkpoint_ids
            for item in self.pending_evaluations
        ):
            raise SearchResumeError("pending evaluations require accepted lineage parents")
        if self.evidence_arrival_cursor < 0:
            raise SearchResumeError("evidence arrival cursor cannot be negative")
        if self.rng_state.decision_index != self.policy_state.decision_index:
            raise SearchResumeError("policy and deterministic RNG decision cursors disagree")
        if self.budget.evaluations_reserved != len(self.policy_state.selected_candidate_ids):
            raise SearchResumeError("policy selections and reserved evaluation budget disagree")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "search_id": self.search_id,
            "generation": self.generation,
            "policy_state": self.policy_state.to_record(),
            "rng_state": self.rng_state.to_record(),
            "frontier_checkpoint_ids": list(self.frontier_checkpoint_ids),
            "lineage_checkpoint_ids": list(self.lineage_checkpoint_ids),
            "budget": self.budget.to_record(),
            "pending_evaluations": [item.to_record() for item in self.pending_evaluations],
            "evidence_arrival_cursor": self.evidence_arrival_cursor,
        }


def _exact_mapping(value: object, keys: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise SearchResumeError(f"stored {label} has missing or unknown fields")
    return value


def _strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SearchResumeError(f"stored {label} must be a string list")
    return tuple(value)


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SearchResumeError(f"stored {label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SearchResumeError(f"stored {label} must be numeric")
    return float(value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise SearchResumeError(f"stored {label} must be text")
    return value


def _snapshot_from_record(value: object) -> SearchResumeSnapshot:
    record = _exact_mapping(
        value,
        {
            "schema_version",
            "search_id",
            "generation",
            "policy_state",
            "rng_state",
            "frontier_checkpoint_ids",
            "lineage_checkpoint_ids",
            "budget",
            "pending_evaluations",
            "evidence_arrival_cursor",
        },
        "search snapshot",
    )
    policy = _exact_mapping(
        record["policy_state"],
        {"schema_version", "policy_id", "selected_candidate_ids", "decision_index"},
        "policy state",
    )
    if _integer(policy["schema_version"], "policy schema version") != 1:
        raise SearchResumeError("stored policy state schema is unsupported")
    policy_state = SearchPolicyState(
        _text(policy["policy_id"], "policy ID"),
        _strings(policy["selected_candidate_ids"], "selected candidates"),
        _integer(policy["decision_index"], "policy decision index"),
    )
    rng = _exact_mapping(
        record["rng_state"],
        {"seed", "decision_index", "tie_namespace_version"},
        "RNG state",
    )
    rng_state = SearchRngState(
        _integer(rng["seed"], "RNG seed"),
        _integer(rng["decision_index"], "RNG decision index"),
        _integer(rng["tie_namespace_version"], "RNG namespace version"),
    )
    budget_record = _exact_mapping(
        record["budget"],
        {"evaluation_limit", "evaluations_reserved", "gpu_seconds_used", "disk_bytes_used"},
        "budget",
    )
    budget = SearchBudgetSnapshot(
        _integer(budget_record["evaluation_limit"], "evaluation limit"),
        _integer(budget_record["evaluations_reserved"], "reserved evaluations"),
        _number(budget_record["gpu_seconds_used"], "GPU seconds"),
        _integer(budget_record["disk_bytes_used"], "disk bytes"),
    )
    pending_raw = record["pending_evaluations"]
    if not isinstance(pending_raw, list):
        raise SearchResumeError("stored pending evaluations must be a list")
    pending: list[PendingSearchEvaluation] = []
    for item in pending_raw:
        pending_record = _exact_mapping(
            item,
            {"candidate_id", "candidate_state_id", "parent_checkpoint_id"},
            "pending evaluation",
        )
        pending.append(
            PendingSearchEvaluation(
                _text(pending_record["candidate_id"], "pending candidate ID"),
                _text(pending_record["candidate_state_id"], "pending state ID"),
                _text(pending_record["parent_checkpoint_id"], "pending parent ID"),
            )
        )
    return SearchResumeSnapshot(
        _text(record["search_id"], "search ID"),
        _integer(record["generation"], "generation"),
        policy_state,
        rng_state,
        _strings(record["frontier_checkpoint_ids"], "frontier checkpoint IDs"),
        _strings(record["lineage_checkpoint_ids"], "lineage checkpoint IDs"),
        budget,
        tuple(pending),
        _integer(record["evidence_arrival_cursor"], "evidence arrival cursor"),
        _integer(record["schema_version"], "schema version"),
    )


class SearchResumeStore:
    """Append-only search snapshots with an atomically advanced latest pointer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS search_resume_snapshots (
                search_id TEXT NOT NULL,
                generation INTEGER NOT NULL CHECK(generation >= 0),
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                PRIMARY KEY(search_id, generation)
            );
            CREATE TABLE IF NOT EXISTS search_resume_heads (
                search_id TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                FOREIGN KEY(search_id, generation)
                    REFERENCES search_resume_snapshots(search_id, generation)
            );
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> SearchResumeStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def save(
        self,
        snapshot: SearchResumeSnapshot,
        *,
        expected_generation: int | None,
    ) -> None:
        payload = canonical_identity_json(snapshot.to_record())
        digest = hashlib.sha256(payload.encode()).hexdigest()
        try:
            with self._connection:
                row = self._connection.execute(
                    "SELECT generation FROM search_resume_heads WHERE search_id = ?",
                    (snapshot.search_id,),
                ).fetchone()
                current = None if row is None else int(row[0])
                if current != expected_generation:
                    raise SearchResumeError(
                        "stale search snapshot generation: "
                        f"expected {expected_generation}, found {current}"
                    )
                required = 0 if current is None else current + 1
                if snapshot.generation != required:
                    raise SearchResumeError(f"next search snapshot generation must be {required}")
                self._connection.execute(
                    "INSERT INTO search_resume_snapshots VALUES (?, ?, ?, ?)",
                    (snapshot.search_id, snapshot.generation, payload, digest),
                )
                if current is None:
                    self._connection.execute(
                        "INSERT INTO search_resume_heads VALUES (?, ?)",
                        (snapshot.search_id, snapshot.generation),
                    )
                else:
                    cursor = self._connection.execute(
                        "UPDATE search_resume_heads SET generation = ? "
                        "WHERE search_id = ? AND generation = ?",
                        (snapshot.generation, snapshot.search_id, current),
                    )
                    if cursor.rowcount != 1:
                        raise SearchResumeError("search resume head changed concurrently")
        except sqlite3.IntegrityError as error:
            raise SearchResumeError("search resume snapshot identity conflicts") from error

    def load_latest(self, search_id: str) -> SearchResumeSnapshot:
        row = self._connection.execute(
            """
            SELECT snapshots.payload_json, snapshots.payload_sha256
            FROM search_resume_heads AS heads
            JOIN search_resume_snapshots AS snapshots
              ON snapshots.search_id = heads.search_id
             AND snapshots.generation = heads.generation
            WHERE heads.search_id = ?
            """,
            (search_id,),
        ).fetchone()
        if row is None:
            raise SearchResumeError("search has no persisted resume snapshot")
        payload = str(row["payload_json"])
        if hashlib.sha256(payload.encode()).hexdigest() != str(row["payload_sha256"]):
            raise SearchResumeError("search resume snapshot checksum mismatch")
        try:
            record = json.loads(payload)
        except json.JSONDecodeError as error:
            raise SearchResumeError("search resume snapshot JSON is corrupt") from error
        return _snapshot_from_record(record)

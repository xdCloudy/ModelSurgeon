"""Evaluated keep/rollback policy with immutable checkpoint lineage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.search.constraints import ConstraintEvaluation
from modelsurgeon.surgery.contracts import MutationTransaction, TransactionState

CHECKPOINT_LINEAGE_SCHEMA_VERSION = 1


class CheckpointLineageError(RuntimeError):
    """Raised when evaluated lineage cannot preserve keep/rollback invariants."""


class LineageDecisionKind(StrEnum):
    KEEP = "keep"
    ROLLBACK = "rollback"


class CandidateArtifactLease(Protocol):
    @property
    def artifact_ids(self) -> tuple[str, ...]: ...

    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class MeasuredConstraintEvidence:
    evaluation_id: str
    evaluation: ConstraintEvaluation

    def __post_init__(self) -> None:
        if not self.evaluation_id.startswith("evaluation_"):
            raise CheckpointLineageError("measured evidence requires an evaluation ID")
        if not self.evaluation.results:
            raise CheckpointLineageError("measured evidence requires constraint results")

    def to_record(self) -> dict[str, object]:
        return {
            "evaluation_id": self.evaluation_id,
            "measured": True,
            "constraint_evaluation": self.evaluation.to_record(),
        }


@dataclass(frozen=True, slots=True)
class AcceptedCheckpoint:
    checkpoint_id: str
    parent_checkpoint_id: str | None
    state_id: str
    artifact_digest: str
    evaluation_id: str
    evidence: dict[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "state_id": self.state_id,
            "artifact_digest": self.artifact_digest,
            "evaluation_id": self.evaluation_id,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class LineageDecision:
    candidate_id: str
    parent_checkpoint_id: str
    candidate_state_id: str
    kind: LineageDecisionKind
    checkpoint_id: str | None
    evaluation_id: str
    released_artifact_ids: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "candidate_state_id": self.candidate_state_id,
            "kind": self.kind.value,
            "checkpoint_id": self.checkpoint_id,
            "evaluation_id": self.evaluation_id,
            "released_artifact_ids": list(self.released_artifact_ids),
        }


def _validate_identity(value: str, prefix: str, label: str) -> None:
    if not value.startswith(prefix):
        raise CheckpointLineageError(f"{label} requires a canonical {prefix} identity")


def _validate_digest(value: str) -> None:
    if not value.startswith("sha256:") or len(value) != 71:
        raise CheckpointLineageError("checkpoint artifacts require a SHA-256 digest")
    hexadecimal = value[7:]
    if hexadecimal != hexadecimal.lower() or any(
        character not in "0123456789abcdef" for character in hexadecimal
    ):
        raise CheckpointLineageError("checkpoint artifact digest is not canonical")


class CheckpointLineageStore:
    """SQLite lineage store whose only roots are accepted checkpoints."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS checkpoint_lineage_metadata (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                schema_version INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO checkpoint_lineage_metadata VALUES (1, 1);
            CREATE TABLE IF NOT EXISTS accepted_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                parent_checkpoint_id TEXT REFERENCES accepted_checkpoints(checkpoint_id),
                state_id TEXT NOT NULL UNIQUE,
                artifact_digest TEXT NOT NULL,
                evaluation_id TEXT NOT NULL,
                evidence_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lineage_decisions (
                candidate_id TEXT PRIMARY KEY,
                parent_checkpoint_id TEXT NOT NULL REFERENCES accepted_checkpoints(checkpoint_id),
                candidate_state_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL CHECK(kind IN ('keep', 'rollback')),
                checkpoint_id TEXT REFERENCES accepted_checkpoints(checkpoint_id),
                evaluation_id TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                released_artifacts_json TEXT NOT NULL,
                CHECK((kind = 'keep' AND checkpoint_id IS NOT NULL) OR
                      (kind = 'rollback' AND checkpoint_id IS NULL))
            );
            """
        )
        row = self._connection.execute(
            "SELECT schema_version FROM checkpoint_lineage_metadata WHERE singleton = 1"
        ).fetchone()
        if row is None or int(row[0]) != CHECKPOINT_LINEAGE_SCHEMA_VERSION:
            self._connection.close()
            raise CheckpointLineageError("checkpoint lineage schema is unsupported")
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> CheckpointLineageStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _checkpoint_from_row(self, row: sqlite3.Row) -> AcceptedCheckpoint:
        evidence = json.loads(str(row["evidence_json"]))
        if not isinstance(evidence, dict):
            raise CheckpointLineageError("stored checkpoint evidence is corrupt")
        return AcceptedCheckpoint(
            str(row["checkpoint_id"]),
            None if row["parent_checkpoint_id"] is None else str(row["parent_checkpoint_id"]),
            str(row["state_id"]),
            str(row["artifact_digest"]),
            str(row["evaluation_id"]),
            evidence,
        )

    def register_root(
        self,
        checkpoint_id: str,
        state_id: str,
        artifact_digest: str,
        evidence: MeasuredConstraintEvidence,
    ) -> AcceptedCheckpoint:
        _validate_identity(checkpoint_id, "checkpoint_", "root checkpoint")
        _validate_identity(state_id, "state_", "root state")
        _validate_digest(artifact_digest)
        if not evidence.evaluation.passed:
            raise CheckpointLineageError("root checkpoint evidence must pass constraints")
        evidence_json = canonical_identity_json(evidence.to_record())
        try:
            with self._connection:
                self._connection.execute(
                    "INSERT INTO accepted_checkpoints VALUES (?, NULL, ?, ?, ?, ?)",
                    (
                        checkpoint_id,
                        state_id,
                        artifact_digest,
                        evidence.evaluation_id,
                        evidence_json,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise CheckpointLineageError("root checkpoint identity already exists") from error
        return self.require_search_root(checkpoint_id)

    def require_search_root(self, checkpoint_id: str) -> AcceptedCheckpoint:
        row = self._connection.execute(
            "SELECT * FROM accepted_checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        ).fetchone()
        if row is None:
            raise CheckpointLineageError("search roots must be accepted checkpoints")
        return self._checkpoint_from_row(row)

    def checkpoints(self) -> tuple[AcceptedCheckpoint, ...]:
        rows = self._connection.execute(
            "SELECT * FROM accepted_checkpoints ORDER BY checkpoint_id"
        ).fetchall()
        return tuple(self._checkpoint_from_row(row) for row in rows)

    def _checkpoint_id(
        self,
        parent_checkpoint_id: str,
        candidate_id: str,
        candidate_state_id: str,
        artifact_digest: str,
        evaluation_id: str,
    ) -> str:
        payload = canonical_identity_json(
            {
                "parent_checkpoint_id": parent_checkpoint_id,
                "candidate_id": candidate_id,
                "candidate_state_id": candidate_state_id,
                "artifact_digest": artifact_digest,
                "evaluation_id": evaluation_id,
            }
        ).encode()
        return f"checkpoint_{hashlib.sha256(payload).hexdigest()}"

    def decide(
        self,
        *,
        parent_checkpoint_id: str,
        candidate_id: str,
        candidate_state_id: str,
        artifact_digest: str,
        evidence: MeasuredConstraintEvidence,
        transaction: MutationTransaction,
        artifact_lease: CandidateArtifactLease,
    ) -> LineageDecision:
        """Commit passing candidates or restore/release failing candidates."""

        self.require_search_root(parent_checkpoint_id)
        _validate_identity(candidate_id, "candidate_", "candidate")
        _validate_identity(candidate_state_id, "state_", "candidate state")
        _validate_digest(artifact_digest)
        if transaction.state is not TransactionState.APPLIED or not transaction.owns_mutable_inputs:
            raise CheckpointLineageError("lineage decisions require an applied owning transaction")
        if (
            self._connection.execute(
                "SELECT 1 FROM lineage_decisions WHERE candidate_id = ? OR candidate_state_id = ?",
                (candidate_id, candidate_state_id),
            ).fetchone()
            is not None
        ):
            raise CheckpointLineageError("candidate lineage decision is immutable")
        evidence_json = canonical_identity_json(evidence.to_record())
        if evidence.evaluation.passed:
            checkpoint_id = self._checkpoint_id(
                parent_checkpoint_id,
                candidate_id,
                candidate_state_id,
                artifact_digest,
                evidence.evaluation_id,
            )
            try:
                with self._connection:
                    self._connection.execute(
                        "INSERT INTO accepted_checkpoints VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            checkpoint_id,
                            parent_checkpoint_id,
                            candidate_state_id,
                            artifact_digest,
                            evidence.evaluation_id,
                            evidence_json,
                        ),
                    )
                    self._connection.execute(
                        "INSERT INTO lineage_decisions VALUES (?, ?, ?, 'keep', ?, ?, ?, '[]')",
                        (
                            candidate_id,
                            parent_checkpoint_id,
                            candidate_state_id,
                            checkpoint_id,
                            evidence.evaluation_id,
                            evidence_json,
                        ),
                    )
                    transaction.commit()
            except sqlite3.IntegrityError as error:
                raise CheckpointLineageError("accepted checkpoint lineage conflicts") from error
            return LineageDecision(
                candidate_id,
                parent_checkpoint_id,
                candidate_state_id,
                LineageDecisionKind.KEEP,
                checkpoint_id,
                evidence.evaluation_id,
                (),
            )

        transaction.rollback()
        released = tuple(artifact_lease.artifact_ids)
        artifact_lease.release()
        with self._connection:
            self._connection.execute(
                "INSERT INTO lineage_decisions VALUES (?, ?, ?, 'rollback', NULL, ?, ?, ?)",
                (
                    candidate_id,
                    parent_checkpoint_id,
                    candidate_state_id,
                    evidence.evaluation_id,
                    evidence_json,
                    canonical_identity_json(list(released)),
                ),
            )
        return LineageDecision(
            candidate_id,
            parent_checkpoint_id,
            candidate_state_id,
            LineageDecisionKind.ROLLBACK,
            None,
            evidence.evaluation_id,
            released,
        )

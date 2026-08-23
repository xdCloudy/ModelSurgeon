"""SQLite-backed experiment metadata persistence with WAL-safe readers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self, cast

from modelsurgeon.experiments.identity import (
    canonical_identity_json,
    derive_candidate_identity,
)
from modelsurgeon.experiments.migrations import open_experiment_database
from modelsurgeon.experiments.schema import ExperimentRecord, MetricObservation


class ExperimentStoreError(RuntimeError):
    """Raised when persisted experiment metadata violates store invariants."""


class MetricPhase(StrEnum):
    BASELINE = "baseline"
    POST = "post"
    DELTA = "delta"


@dataclass(frozen=True, slots=True)
class PersistedExperiment:
    input_id: str
    run_id: str
    candidate_id: str


@dataclass(frozen=True, slots=True)
class StoredInput:
    input_id: str
    model_identifier: str
    model_revision: str
    model_family: str
    model_format: str
    model_parameter_count: int | None
    model_quantization: str | None
    dataset_identifier: str
    dataset_revision: str
    dataset_split: str
    dataset_manifest_id: str
    tokenizer: str
    tokenizer_revision: str
    config_digest: str


@dataclass(frozen=True, slots=True)
class StoredRun:
    run_id: str
    experiment_id: str
    attempt_id: str
    input_id: str
    mutation_id: str
    experiment_schema_version: int
    mutation_record_schema_version: int
    mutation: dict[str, object]
    outcome: dict[str, object]
    hardware: dict[str, object]
    versions: dict[str, object]
    seeds: dict[str, object]
    quantization_control: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class StoredCandidate:
    candidate_id: str
    run_id: str
    mutation_id: str
    affected_components: tuple[str, ...]
    candidate_order: int


@dataclass(frozen=True, slots=True)
class StoredStateEvent:
    state_event_id: int
    candidate_id: str
    sequence: int
    state: str
    detail: str | None


@dataclass(frozen=True, slots=True)
class StoredMetric:
    metric_id: int
    candidate_id: str
    phase: MetricPhase
    name: str
    state: str
    value: float | None
    unit: str | None
    reason: str | None
    precision: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class StoredArtifactReference:
    reference_id: str
    candidate_id: str
    role: str
    digest: str
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class CandidateTrace:
    candidate_id: str
    run_id: str
    experiment_id: str
    input_id: str
    model_identifier: str
    model_revision: str
    dataset_manifest_id: str
    config_digest: str


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_object(payload: str) -> dict[str, object]:
    value = json.loads(payload)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ExperimentStoreError("stored JSON object is malformed")
    return cast(dict[str, object], value)


def _json_string_tuple(payload: str) -> tuple[str, ...]:
    value = json.loads(payload)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ExperimentStoreError("stored component list is malformed")
    return tuple(value)


def derive_input_id(record: ExperimentRecord) -> str:
    payload = {
        "model": record.model.to_record(),
        "dataset": record.dataset.to_record(),
        "config_digest": record.versions.config_digest,
    }
    encoded = canonical_identity_json(payload).encode("utf-8")
    return f"input_{hashlib.sha256(encoded).hexdigest()}"


def _artifact_reference_id(candidate_id: str, role: str, digest: str) -> str:
    encoded = canonical_identity_json(
        {"candidate_id": candidate_id, "role": role, "digest": digest}
    ).encode("utf-8")
    return f"ref_{hashlib.sha256(encoded).hexdigest()}"


class ExperimentMetadataStore:
    """One WAL writer plus short-lived read-only connections for experiment metadata."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = open_experiment_database(self.path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._lock = threading.RLock()
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()

    @property
    def journal_mode(self) -> str:
        self._require_open()
        row = self._connection.execute("PRAGMA journal_mode").fetchone()
        if row is None:
            raise ExperimentStoreError("SQLite did not report a journal mode")
        return str(row[0]).lower()

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._connection.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise ExperimentStoreError("experiment metadata store is closed")

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        """Open a separate query-only connection that can read while the WAL writer is active."""

        self._require_open()
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA query_only = ON")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        self._require_open()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    @staticmethod
    def _insert_or_verify(
        connection: sqlite3.Connection,
        *,
        table: str,
        key_column: str,
        key_value: object,
        columns: tuple[str, ...],
        values: tuple[object, ...],
    ) -> None:
        column_sql = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        connection.execute(
            f"INSERT OR IGNORE INTO {table} ({column_sql}) VALUES ({placeholders})",
            values,
        )
        row = connection.execute(
            f"SELECT {column_sql} FROM {table} WHERE {key_column} = ?",
            (key_value,),
        ).fetchone()
        if row is None or tuple(row[column] for column in columns) != values:
            raise ExperimentStoreError(
                f"immutable {table} record {key_value!r} conflicts with existing metadata"
            )

    def persist_experiment(
        self,
        record: ExperimentRecord,
        *,
        candidate_order: int = 0,
    ) -> PersistedExperiment:
        """Persist one experiment candidate and its immutable context atomically."""

        if candidate_order < 0:
            raise ExperimentStoreError("candidate order must be non-negative")
        if not record.experiment_id.startswith("exp_") or not record.run_id.startswith("run_"):
            raise ExperimentStoreError(
                "experiment persistence requires deterministic experiment and run IDs"
            )
        input_id = derive_input_id(record)
        candidate_id = derive_candidate_identity(record.run_id, record.mutation_id).candidate_id
        input_columns = (
            "input_id",
            "model_identifier",
            "model_revision",
            "model_family",
            "model_format",
            "model_parameter_count",
            "model_quantization",
            "dataset_identifier",
            "dataset_revision",
            "dataset_split",
            "dataset_manifest_id",
            "tokenizer",
            "tokenizer_revision",
            "config_digest",
        )
        input_values: tuple[object, ...] = (
            input_id,
            record.model.identifier,
            record.model.revision,
            record.model.family,
            record.model.format,
            record.model.parameter_count,
            record.model.quantization,
            record.dataset.identifier,
            record.dataset.revision,
            record.dataset.split,
            record.dataset.manifest_id,
            record.dataset.tokenizer,
            record.dataset.tokenizer_revision,
            record.versions.config_digest,
        )
        mutation_json = _json(record.mutation.to_record())
        outcome_json = _json(record.outcome.to_record())
        run_columns = (
            "run_id",
            "experiment_id",
            "attempt_id",
            "input_id",
            "mutation_id",
            "experiment_schema_version",
            "mutation_record_schema_version",
            "mutation_json",
            "outcome_json",
            "hardware_json",
            "versions_json",
            "seeds_json",
            "quantization_control_json",
        )
        run_values: tuple[object, ...] = (
            record.run_id,
            record.experiment_id,
            record.attempt_id,
            input_id,
            record.mutation_id,
            record.schema_version,
            record.mutation.schema_version,
            mutation_json,
            outcome_json,
            _json(record.hardware.to_record()),
            _json(record.versions.to_record()),
            _json(record.seeds.to_record()),
            None
            if record.quantization_control is None
            else _json(record.quantization_control.to_record()),
        )
        candidate_columns = (
            "candidate_id",
            "run_id",
            "mutation_id",
            "affected_components_json",
            "candidate_order",
        )
        candidate_values: tuple[object, ...] = (
            candidate_id,
            record.run_id,
            record.mutation_id,
            _json([str(item) for item in record.components]),
            candidate_order,
        )
        with self._write() as connection:
            self._insert_or_verify(
                connection,
                table="experiment_inputs",
                key_column="input_id",
                key_value=input_id,
                columns=input_columns,
                values=input_values,
            )
            self._insert_or_verify(
                connection,
                table="experiment_runs",
                key_column="run_id",
                key_value=record.run_id,
                columns=run_columns,
                values=run_values,
            )
            self._insert_or_verify(
                connection,
                table="experiment_candidates",
                key_column="candidate_id",
                key_value=candidate_id,
                columns=candidate_columns,
                values=candidate_values,
            )
            for phase, metrics in (
                (MetricPhase.BASELINE, record.baseline_metrics),
                (MetricPhase.POST, record.post_metrics),
                (MetricPhase.DELTA, record.delta_metrics),
            ):
                for metric in metrics:
                    self._persist_metric(connection, candidate_id, phase, metric)
            for timing in record.timings:
                columns = (
                    "candidate_id",
                    "stage",
                    "wall_seconds",
                    "cpu_seconds",
                    "tokens",
                    "candidates",
                )
                values: tuple[object, ...] = (
                    candidate_id,
                    timing.stage,
                    timing.wall_seconds,
                    timing.cpu_seconds,
                    timing.tokens,
                    timing.candidates,
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO experiment_stage_timings
                        (candidate_id, stage, wall_seconds, cpu_seconds, tokens, candidates)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                row = connection.execute(
                    """
                    SELECT candidate_id, stage, wall_seconds, cpu_seconds, tokens, candidates
                    FROM experiment_stage_timings
                    WHERE candidate_id = ? AND stage = ?
                    """,
                    (candidate_id, timing.stage),
                ).fetchone()
                if row is None or tuple(row[column] for column in columns) != values:
                    raise ExperimentStoreError("immutable stage timing conflicts with existing data")
        return PersistedExperiment(input_id, record.run_id, candidate_id)

    @staticmethod
    def _persist_metric(
        connection: sqlite3.Connection,
        candidate_id: str,
        phase: MetricPhase,
        metric: MetricObservation,
    ) -> None:
        precision_json = None if metric.precision is None else _json(metric.precision.to_record())
        values: tuple[object, ...] = (
            candidate_id,
            phase.value,
            metric.name,
            metric.state.value,
            metric.value,
            metric.unit,
            metric.reason,
            precision_json,
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO experiment_metrics
                (candidate_id, phase, name, state, value, unit, reason, precision_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        row = connection.execute(
            """
            SELECT candidate_id, phase, name, state, value, unit, reason, precision_json
            FROM experiment_metrics
            WHERE candidate_id = ? AND phase = ? AND name = ?
            """,
            (candidate_id, phase.value, metric.name),
        ).fetchone()
        columns = (
            "candidate_id",
            "phase",
            "name",
            "state",
            "value",
            "unit",
            "reason",
            "precision_json",
        )
        if row is None or tuple(row[column] for column in columns) != values:
            raise ExperimentStoreError("immutable metric conflicts with existing data")

    def append_state(self, candidate_id: str, state: str, detail: str | None = None) -> StoredStateEvent:
        if not candidate_id or not state or (detail is not None and not detail):
            raise ExperimentStoreError("state events require candidate/state and non-blank detail")
        with self._write() as connection:
            if connection.execute(
                "SELECT 1 FROM experiment_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone() is None:
                raise ExperimentStoreError(f"unknown experiment candidate {candidate_id}")
            row = connection.execute(
                """
                SELECT COALESCE(MAX(sequence), -1) + 1
                FROM experiment_state_events
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise ExperimentStoreError("failed to derive the next state sequence")
            sequence = int(row[0])
            cursor = connection.execute(
                """
                INSERT INTO experiment_state_events(candidate_id, sequence, state, detail)
                VALUES (?, ?, ?, ?)
                """,
                (candidate_id, sequence, state, detail),
            )
            event_id = int(cursor.lastrowid)
        return StoredStateEvent(event_id, candidate_id, sequence, state, detail)

    def add_artifact_reference(
        self,
        candidate_id: str,
        *,
        role: str,
        digest: str,
        metadata: Mapping[str, object] | None = None,
    ) -> StoredArtifactReference:
        if not candidate_id or not role or not digest:
            raise ExperimentStoreError("artifact references require candidate, role, and digest")
        canonical_metadata = canonical_identity_json({} if metadata is None else metadata)
        reference_id = _artifact_reference_id(candidate_id, role, digest)
        values: tuple[object, ...] = (
            reference_id,
            candidate_id,
            role,
            digest,
            canonical_metadata,
        )
        with self._write() as connection:
            try:
                self._insert_or_verify(
                    connection,
                    table="experiment_artifact_references",
                    key_column="reference_id",
                    key_value=reference_id,
                    columns=("reference_id", "candidate_id", "role", "digest", "metadata_json"),
                    values=values,
                )
            except sqlite3.IntegrityError as error:
                raise ExperimentStoreError(
                    f"artifact reference points to unknown candidate {candidate_id}"
                ) from error
        return StoredArtifactReference(
            reference_id,
            candidate_id,
            role,
            digest,
            _json_object(canonical_metadata),
        )

    def get_input(self, input_id: str) -> StoredInput | None:
        with self.reader() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_inputs WHERE input_id = ?", (input_id,)
            ).fetchone()
        if row is None:
            return None
        return StoredInput(
            str(row["input_id"]),
            str(row["model_identifier"]),
            str(row["model_revision"]),
            str(row["model_family"]),
            str(row["model_format"]),
            None if row["model_parameter_count"] is None else int(row["model_parameter_count"]),
            None if row["model_quantization"] is None else str(row["model_quantization"]),
            str(row["dataset_identifier"]),
            str(row["dataset_revision"]),
            str(row["dataset_split"]),
            str(row["dataset_manifest_id"]),
            str(row["tokenizer"]),
            str(row["tokenizer_revision"]),
            str(row["config_digest"]),
        )

    def get_run(self, run_id: str) -> StoredRun | None:
        with self.reader() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        return StoredRun(
            str(row["run_id"]),
            str(row["experiment_id"]),
            str(row["attempt_id"]),
            str(row["input_id"]),
            str(row["mutation_id"]),
            int(row["experiment_schema_version"]),
            int(row["mutation_record_schema_version"]),
            _json_object(str(row["mutation_json"])),
            _json_object(str(row["outcome_json"])),
            _json_object(str(row["hardware_json"])),
            _json_object(str(row["versions_json"])),
            _json_object(str(row["seeds_json"])),
            None
            if row["quantization_control_json"] is None
            else _json_object(str(row["quantization_control_json"])),
        )

    def get_candidate(self, candidate_id: str) -> StoredCandidate | None:
        with self.reader() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        if row is None:
            return None
        return StoredCandidate(
            str(row["candidate_id"]),
            str(row["run_id"]),
            str(row["mutation_id"]),
            _json_string_tuple(str(row["affected_components_json"])),
            int(row["candidate_order"]),
        )

    def list_states(self, candidate_id: str) -> tuple[StoredStateEvent, ...]:
        with self.reader() as connection:
            rows = connection.execute(
                """
                SELECT state_event_id, candidate_id, sequence, state, detail
                FROM experiment_state_events
                WHERE candidate_id = ?
                ORDER BY sequence
                """,
                (candidate_id,),
            ).fetchall()
        return tuple(
            StoredStateEvent(
                int(row["state_event_id"]),
                str(row["candidate_id"]),
                int(row["sequence"]),
                str(row["state"]),
                None if row["detail"] is None else str(row["detail"]),
            )
            for row in rows
        )

    def list_metrics(self, candidate_id: str) -> tuple[StoredMetric, ...]:
        with self.reader() as connection:
            rows = connection.execute(
                """
                SELECT metric_id, candidate_id, phase, name, state, value, unit, reason,
                       precision_json
                FROM experiment_metrics
                WHERE candidate_id = ?
                ORDER BY phase, name
                """,
                (candidate_id,),
            ).fetchall()
        return tuple(
            StoredMetric(
                int(row["metric_id"]),
                str(row["candidate_id"]),
                MetricPhase(str(row["phase"])),
                str(row["name"]),
                str(row["state"]),
                None if row["value"] is None else float(row["value"]),
                None if row["unit"] is None else str(row["unit"]),
                None if row["reason"] is None else str(row["reason"]),
                None
                if row["precision_json"] is None
                else _json_object(str(row["precision_json"])),
            )
            for row in rows
        )

    def list_artifact_references(
        self,
        candidate_id: str,
    ) -> tuple[StoredArtifactReference, ...]:
        with self.reader() as connection:
            rows = connection.execute(
                """
                SELECT reference_id, candidate_id, role, digest, metadata_json
                FROM experiment_artifact_references
                WHERE candidate_id = ?
                ORDER BY role, digest
                """,
                (candidate_id,),
            ).fetchall()
        return tuple(
            StoredArtifactReference(
                str(row["reference_id"]),
                str(row["candidate_id"]),
                str(row["role"]),
                str(row["digest"]),
                _json_object(str(row["metadata_json"])),
            )
            for row in rows
        )

    def trace_candidate(self, candidate_id: str) -> CandidateTrace | None:
        with self.reader() as connection:
            row = connection.execute(
                """
                SELECT c.candidate_id, c.run_id, r.experiment_id, r.input_id,
                       i.model_identifier, i.model_revision, i.dataset_manifest_id,
                       i.config_digest
                FROM experiment_candidates AS c
                JOIN experiment_runs AS r ON r.run_id = c.run_id
                JOIN experiment_inputs AS i ON i.input_id = r.input_id
                WHERE c.candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()
        if row is None:
            return None
        return CandidateTrace(
            str(row["candidate_id"]),
            str(row["run_id"]),
            str(row["experiment_id"]),
            str(row["input_id"]),
            str(row["model_identifier"]),
            str(row["model_revision"]),
            str(row["dataset_manifest_id"]),
            str(row["config_digest"]),
        )

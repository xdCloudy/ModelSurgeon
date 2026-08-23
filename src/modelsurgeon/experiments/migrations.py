"""Ordered, checksummed SQLite schema migrations for experiment metadata."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

EXPERIMENT_DB_SCHEMA_VERSION = 4


class ExperimentMigrationError(RuntimeError):
    """Raised when experiment metadata migrations cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class ExperimentMigration:
    version: int
    name: str
    statements: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.version <= 0 or not self.name or not self.statements:
            raise ValueError("migrations require a positive version, name, and SQL statements")
        if any(not statement.strip() for statement in self.statements):
            raise ValueError("migration SQL statements cannot be blank")

    @property
    def checksum(self) -> str:
        payload = "\n".join((str(self.version), self.name, *self.statements)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class MigrationBackupGuidance:
    recommended: bool
    reason: str
    guidance: str


@dataclass(frozen=True, slots=True)
class MigrationReport:
    start_version: int
    end_version: int
    applied_versions: tuple[int, ...]
    backup: MigrationBackupGuidance


_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    checksum TEXT NOT NULL
)
""".strip()


MIGRATIONS: tuple[ExperimentMigration, ...] = (
    ExperimentMigration(
        1,
        "experiment identities and runs",
        (
            """
            CREATE TABLE experiment_inputs (
                input_id TEXT PRIMARY KEY,
                model_identifier TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                model_family TEXT NOT NULL,
                model_format TEXT NOT NULL,
                model_parameter_count INTEGER,
                model_quantization TEXT,
                dataset_identifier TEXT NOT NULL,
                dataset_revision TEXT NOT NULL,
                dataset_split TEXT NOT NULL,
                dataset_manifest_id TEXT NOT NULL,
                tokenizer TEXT NOT NULL,
                tokenizer_revision TEXT NOT NULL,
                config_digest TEXT NOT NULL,
                UNIQUE (
                    model_identifier,
                    model_revision,
                    model_format,
                    dataset_manifest_id,
                    config_digest
                )
            )
            """.strip(),
            """
            CREATE TABLE experiment_runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                input_id TEXT NOT NULL REFERENCES experiment_inputs(input_id),
                mutation_id TEXT NOT NULL,
                experiment_schema_version INTEGER NOT NULL,
                mutation_record_schema_version INTEGER NOT NULL,
                mutation_json TEXT NOT NULL,
                outcome_json TEXT NOT NULL,
                hardware_json TEXT NOT NULL,
                versions_json TEXT NOT NULL,
                seeds_json TEXT NOT NULL,
                quantization_control_json TEXT,
                UNIQUE (experiment_id, attempt_id)
            )
            """.strip(),
            "CREATE INDEX experiment_runs_experiment_idx ON experiment_runs(experiment_id)",
        ),
    ),
    ExperimentMigration(
        2,
        "candidates states metrics and artifacts",
        (
            """
            CREATE TABLE experiment_candidates (
                candidate_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
                mutation_id TEXT NOT NULL,
                affected_components_json TEXT NOT NULL,
                candidate_order INTEGER NOT NULL CHECK(candidate_order >= 0),
                UNIQUE (run_id, mutation_id)
            )
            """.strip(),
            """
            CREATE TABLE experiment_state_events (
                state_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL
                    REFERENCES experiment_candidates(candidate_id) ON DELETE CASCADE,
                sequence INTEGER NOT NULL CHECK(sequence >= 0),
                state TEXT NOT NULL,
                detail TEXT,
                UNIQUE (candidate_id, sequence)
            )
            """.strip(),
            """
            CREATE TABLE experiment_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL
                    REFERENCES experiment_candidates(candidate_id) ON DELETE CASCADE,
                phase TEXT NOT NULL,
                name TEXT NOT NULL,
                state TEXT NOT NULL,
                value REAL,
                unit TEXT,
                reason TEXT,
                precision_json TEXT,
                UNIQUE (candidate_id, phase, name)
            )
            """.strip(),
            """
            CREATE TABLE experiment_artifact_references (
                reference_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL
                    REFERENCES experiment_candidates(candidate_id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                digest TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE (candidate_id, role, digest)
            )
            """.strip(),
            """
            CREATE TABLE experiment_stage_timings (
                timing_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL
                    REFERENCES experiment_candidates(candidate_id) ON DELETE CASCADE,
                stage TEXT NOT NULL,
                wall_seconds REAL NOT NULL CHECK(wall_seconds >= 0),
                cpu_seconds REAL CHECK(cpu_seconds >= 0),
                tokens INTEGER CHECK(tokens >= 0),
                candidates INTEGER CHECK(candidates >= 0),
                UNIQUE (candidate_id, stage)
            )
            """.strip(),
            "CREATE INDEX experiment_candidates_run_idx ON experiment_candidates(run_id)",
            "CREATE INDEX experiment_metrics_candidate_idx ON experiment_metrics(candidate_id)",
        ),
    ),
    ExperimentMigration(
        3,
        "candidate work leases",
        (
            """
            CREATE TABLE experiment_work_leases (
                candidate_id TEXT PRIMARY KEY
                    REFERENCES experiment_candidates(candidate_id) ON DELETE CASCADE,
                attempt_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                lease_token TEXT NOT NULL UNIQUE,
                generation INTEGER NOT NULL CHECK(generation > 0),
                acquired_at_ns INTEGER NOT NULL CHECK(acquired_at_ns >= 0),
                heartbeat_at_ns INTEGER NOT NULL CHECK(heartbeat_at_ns >= acquired_at_ns),
                expires_at_ns INTEGER NOT NULL CHECK(expires_at_ns > heartbeat_at_ns),
                completed_at_ns INTEGER CHECK(completed_at_ns >= heartbeat_at_ns)
            )
            """.strip(),
            (
                "CREATE INDEX experiment_work_leases_expiry_idx "
                "ON experiment_work_leases(expires_at_ns)"
            ),
        ),
    ),
    ExperimentMigration(
        4,
        "campaign plans checkpoints and results",
        (
            """
            CREATE TABLE experiment_campaign_runs (
                run_id TEXT PRIMARY KEY
                    REFERENCES experiment_runs(run_id) ON DELETE CASCADE,
                plan_digest TEXT NOT NULL,
                baseline_key_json TEXT NOT NULL,
                candidate_count INTEGER NOT NULL CHECK(candidate_count >= 0)
            )
            """.strip(),
            """
            CREATE TABLE experiment_campaign_status (
                candidate_id TEXT PRIMARY KEY
                    REFERENCES experiment_candidates(candidate_id) ON DELETE CASCADE,
                checkpoint_json TEXT,
                evaluation_json TEXT,
                recovery_json TEXT NOT NULL DEFAULT '{}',
                outcome TEXT NOT NULL DEFAULT 'pending'
                    CHECK(outcome IN ('pending', 'succeeded', 'rejected', 'failed')),
                detail TEXT
            )
            """.strip(),
            (
                "CREATE INDEX experiment_campaign_status_outcome_idx "
                "ON experiment_campaign_status(outcome)"
            ),
        ),
    ),
)


def _validate_registry(migrations: tuple[ExperimentMigration, ...]) -> None:
    versions = tuple(item.version for item in migrations)
    if versions != tuple(range(1, len(migrations) + 1)):
        raise ExperimentMigrationError("migration registry versions must be contiguous from 1")
    if len({item.name for item in migrations}) != len(migrations):
        raise ExperimentMigrationError("migration registry names must be unique")


_validate_registry(MIGRATIONS)


def _configure_connection(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    if connection.in_transaction:
        raise ExperimentMigrationError("migration runner requires an idle SQLite connection")


def _bootstrap(connection: sqlite3.Connection) -> None:
    connection.execute(_BOOTSTRAP_SQL)
    connection.commit()


def _applied_rows(connection: sqlite3.Connection) -> tuple[tuple[int, str, str], ...]:
    rows = connection.execute(
        "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
    ).fetchall()
    return tuple((int(row[0]), str(row[1]), str(row[2])) for row in rows)


def _validate_applied(
    rows: tuple[tuple[int, str, str], ...],
    migrations: tuple[ExperimentMigration, ...],
) -> int:
    versions = tuple(row[0] for row in rows)
    if versions != tuple(range(1, len(rows) + 1)):
        raise ExperimentMigrationError("applied migration versions contain a gap or invalid order")
    if len(rows) > len(migrations):
        raise ExperimentMigrationError("database schema is newer than this ModelSurgeon build")
    for version, name, checksum in rows:
        expected = migrations[version - 1]
        if name != expected.name or checksum != expected.checksum:
            raise ExperimentMigrationError(
                f"applied migration {version} does not match the registered checksum"
            )
    return len(rows)


def migration_backup_guidance(start_version: int, end_version: int) -> MigrationBackupGuidance:
    if start_version == 0 or end_version <= start_version:
        return MigrationBackupGuidance(
            False,
            "fresh database or no upgrade is required",
            "No pre-upgrade backup is required by the migration runner.",
        )
    return MigrationBackupGuidance(
        True,
        f"database will be upgraded from schema {start_version} to {end_version}",
        (
            "Create a consistent copy before upgrading, preferably with "
            "sqlite3.Connection.backup(), then keep the source database until validation passes."
        ),
    )


def apply_experiment_migrations(
    connection: sqlite3.Connection,
    *,
    target_version: int = EXPERIMENT_DB_SCHEMA_VERSION,
    migrations: tuple[ExperimentMigration, ...] = MIGRATIONS,
) -> MigrationReport:
    """Apply missing migrations one-by-one, recording each only after its transaction commits."""

    _validate_registry(migrations)
    if target_version < 0 or target_version > len(migrations):
        supported = f"0..{len(migrations)}"
        raise ExperimentMigrationError(
            f"target schema version {target_version} is outside supported range {supported}"
        )
    _configure_connection(connection)
    _bootstrap(connection)
    start_version = _validate_applied(_applied_rows(connection), migrations)
    if target_version < start_version:
        raise ExperimentMigrationError("schema downgrades are not supported")
    backup = migration_backup_guidance(start_version, target_version)
    applied: list[int] = []
    for migration in migrations[start_version:target_version]:
        try:
            connection.execute("BEGIN IMMEDIATE")
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO schema_migrations(version, name, checksum) VALUES (?, ?, ?)",
                (migration.version, migration.name, migration.checksum),
            )
            connection.commit()
        except Exception as error:
            connection.rollback()
            raise ExperimentMigrationError(
                f"migration {migration.version} ({migration.name}) failed and was rolled back"
            ) from error
        applied.append(migration.version)
    end_version = _validate_applied(_applied_rows(connection), migrations)
    if end_version != target_version:
        raise ExperimentMigrationError("database did not reach the requested schema version")
    return MigrationReport(start_version, end_version, tuple(applied), backup)


def open_experiment_database(
    path: str | Path,
    *,
    target_version: int = EXPERIMENT_DB_SCHEMA_VERSION,
) -> sqlite3.Connection:
    """Open a SQLite database, enable integrity pragmas, and migrate it before returning."""

    connection = sqlite3.connect(Path(path))
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        apply_experiment_migrations(connection, target_version=target_version)
    except BaseException:
        connection.close()
        raise
    return connection

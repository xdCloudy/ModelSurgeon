"""Tests for ordered, idempotent, transactional experiment database migrations."""

from __future__ import annotations

import sqlite3

import pytest

from modelsurgeon.experiments import (
    EXPERIMENT_DB_SCHEMA_VERSION,
    MIGRATIONS,
    ExperimentMigration,
    ExperimentMigrationError,
    apply_experiment_migrations,
)


def _schema_snapshot(connection: sqlite3.Connection) -> tuple[tuple[str, str, str], ...]:
    rows = connection.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    return tuple((str(row[0]), str(row[1]), str(row[2])) for row in rows)


def test_fresh_and_incrementally_upgraded_databases_reach_identical_schema() -> None:
    fresh = sqlite3.connect(":memory:")
    upgraded = sqlite3.connect(":memory:")
    try:
        fresh_report = apply_experiment_migrations(fresh)
        first_report = apply_experiment_migrations(upgraded, target_version=1)
        upgrade_report = apply_experiment_migrations(upgraded)

        assert fresh_report.start_version == 0
        assert fresh_report.end_version == EXPERIMENT_DB_SCHEMA_VERSION
        assert fresh_report.applied_versions == tuple(
            range(1, EXPERIMENT_DB_SCHEMA_VERSION + 1)
        )
        assert first_report.applied_versions == (1,)
        assert upgrade_report.start_version == 1
        assert upgrade_report.applied_versions == tuple(
            range(2, EXPERIMENT_DB_SCHEMA_VERSION + 1)
        )
        assert upgrade_report.backup.recommended
        assert "Connection.backup" in upgrade_report.backup.guidance
        assert _schema_snapshot(fresh) == _schema_snapshot(upgraded)
    finally:
        fresh.close()
        upgraded.close()


def test_repeated_run_is_idempotent_and_records_checksums() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        apply_experiment_migrations(connection)
        second = apply_experiment_migrations(connection)
        assert second.start_version == EXPERIMENT_DB_SCHEMA_VERSION
        assert second.end_version == EXPERIMENT_DB_SCHEMA_VERSION
        assert second.applied_versions == ()
        assert not second.backup.recommended

        rows = connection.execute(
            "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert rows == [
            (migration.version, migration.name, migration.checksum)
            for migration in MIGRATIONS
        ]
    finally:
        connection.close()


def test_failed_migration_rolls_back_schema_and_version_record() -> None:
    migrations = (
        ExperimentMigration(1, "base", ("CREATE TABLE stable(value TEXT NOT NULL)",)),
        ExperimentMigration(
            2,
            "failing",
            (
                "CREATE TABLE transient(value TEXT NOT NULL)",
                "INSERT INTO missing_table(value) VALUES ('boom')",
            ),
        ),
    )
    connection = sqlite3.connect(":memory:")
    try:
        apply_experiment_migrations(connection, target_version=1, migrations=migrations)
        with pytest.raises(ExperimentMigrationError, match="rolled back"):
            apply_experiment_migrations(connection, target_version=2, migrations=migrations)

        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "stable" in tables
        assert "transient" not in tables
        assert connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(1,)]
    finally:
        connection.close()


def test_checksum_drift_and_schema_downgrade_fail_closed() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        apply_experiment_migrations(connection)
        connection.execute(
            "UPDATE schema_migrations SET checksum = 'tampered' WHERE version = 1"
        )
        connection.commit()
        with pytest.raises(ExperimentMigrationError, match="checksum"):
            apply_experiment_migrations(connection)
    finally:
        connection.close()

    downgrade = sqlite3.connect(":memory:")
    try:
        apply_experiment_migrations(downgrade)
        with pytest.raises(ExperimentMigrationError, match="downgrades"):
            apply_experiment_migrations(downgrade, target_version=1)
    finally:
        downgrade.close()


def test_runner_rejects_active_transactions_and_invalid_registry() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE caller_owned(value TEXT)")
        connection.execute("INSERT INTO caller_owned VALUES ('open')")
        with pytest.raises(ExperimentMigrationError, match="idle"):
            apply_experiment_migrations(connection)
        connection.rollback()

        invalid = (
            ExperimentMigration(2, "wrong-start", ("CREATE TABLE nope(value TEXT)",)),
        )
        with pytest.raises(ExperimentMigrationError, match="contiguous"):
            apply_experiment_migrations(connection, migrations=invalid)
    finally:
        connection.close()

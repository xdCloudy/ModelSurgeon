"""Audit the bounded v3.0 migration compatibility release record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modelsurgeon.migration import migrate_record, migration_matrix_records


class MigrationCompatibilityAuditError(ValueError):
    """Raised when the migration release record is incomplete or inconsistent."""


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MigrationCompatibilityAuditError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MigrationCompatibilityAuditError(f"{label} must be non-empty text")
    return value


def _files(root: Path, value: object, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise MigrationCompatibilityAuditError(f"{label} must be a non-empty array")
    for index, item in enumerate(value):
        relative = Path(_text(item, f"{label}[{index}]"))
        if relative.is_absolute() or ".." in relative.parts or not (root / relative).is_file():
            raise MigrationCompatibilityAuditError(
                f"{label}[{index}] references a missing or unsafe repository path"
            )


def audit_release(
    root: Path = Path("."), *, manifest: Path | None = None
) -> None:
    """Validate the checked-in matrix, fixtures, and executable migration claims."""

    root = root.resolve()
    manifest_path = manifest or (
        root / "docs" / "research" / "v3.0-migration-compatibility-v1.json"
    )
    try:
        record = _object(json.loads(manifest_path.read_text(encoding="utf-8")), "release record")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MigrationCompatibilityAuditError(
            f"could not read migration release record {manifest_path}"
        ) from error
    if record.get("record_type") != "migration_compatibility_release":
        raise MigrationCompatibilityAuditError("unexpected migration release record type")
    if record.get("schema_version") != 1:
        raise MigrationCompatibilityAuditError("unsupported migration release schema version")
    if record.get("protocol_revision") != "modelsurgeon-v3.0-migration-compatibility-v1":
        raise MigrationCompatibilityAuditError("unexpected migration protocol revision")
    if record.get("status") != "implemented" or record.get("issue") != 489:
        raise MigrationCompatibilityAuditError("migration release status or issue is incorrect")

    matrix = {(item["source"], item["target"]): item for item in migration_matrix_records()}
    supported = record.get("supported_cells")
    if not isinstance(supported, list) or not supported:
        raise MigrationCompatibilityAuditError("supported_cells must be non-empty")
    for index, raw in enumerate(supported):
        cell = _object(raw, f"supported_cells[{index}]")
        key = (_text(cell.get("source"), "source"), _text(cell.get("target"), "target"))
        if key not in matrix or matrix[key]["status"] != "supported":
            raise MigrationCompatibilityAuditError(f"unsupported or undeclared cell {key}")
        _text(cell.get("guarantee"), f"supported_cells[{index}].guarantee")

    unsupported = record.get("unsupported_cells")
    if not isinstance(unsupported, list) or not unsupported:
        raise MigrationCompatibilityAuditError("unsupported_cells must be non-empty")
    if not any("unknown or future" in str(item) for item in unsupported):
        raise MigrationCompatibilityAuditError("future-schema refusal is not declared")
    if not any("source artifact digest" in str(item) for item in unsupported):
        raise MigrationCompatibilityAuditError("lineage refusal is not declared")

    invariants = record.get("invariants")
    if not isinstance(invariants, list) or not any(
        "fail closed" in str(item) for item in invariants
    ):
        raise MigrationCompatibilityAuditError("fail-closed invariant is missing")
    direct = _object(record.get("direct_automation"), "direct_automation")
    if direct.get("provider_default") != "none":
        raise MigrationCompatibilityAuditError("direct automation must default to none")
    _files(root, record.get("fixtures"), "fixtures")
    _files(root, record.get("documentation"), "documentation")
    _files(root, record.get("implementation"), "implementation")
    _files(root, record.get("tests"), "tests")

    fixture_kinds = {
        "tests/fixtures/v2.0-config.json": "config",
        "tests/fixtures/v2.0-campaign-run.json": "campaign",
        "tests/fixtures/v2.0-campaign-evidence.json": "evidence",
    }
    for relative, kind in fixture_kinds.items():
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        result = migrate_record(payload, kind=kind)
        if not result.record:
            raise MigrationCompatibilityAuditError(f"fixture {relative} produced an empty record")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v3.0 migration compatibility boundary verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit the bounded v2.7 campaign recovery and determinism boundary."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class CampaignRecoveryAuditError(ValueError):
    """Raised when the v2.7 recovery evidence drifts or overclaims."""


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DEPENDENCIES = {
    "v27-lifecycle": (470, "270bf0e"),
    "v27-stale": (471, "8b3eadc"),
    "v27-summary": (472, "67a7371"),
}
_EXPECTED_CELLS = {
    "pause_resume",
    "cancel",
    "reconnect",
    "restart",
    "stale_plan",
    "expired_approval",
    "transcript_loss",
    "partial_evidence",
    "fault_transition_after_sql",
    "fault_evidence_after_sql",
    "process_restart",
    "direct_chat_recovery",
}
_GUARANTEES = {
    "approval_is_required_for_resume",
    "hard_constraints_are_preserved",
    "transaction_boundaries_are_atomic",
    "identical_inputs_have_identical_next_action",
    "negative_and_inconclusive_evidence_is_retained",
    "resource_budgets_and_provenance_are_preserved",
    "ui_reconnect_is_not_correctness_evidence",
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignRecoveryAuditError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignRecoveryAuditError(f"{label} must be non-empty text")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CampaignRecoveryAuditError(f"{label} must be an array")
    return value


def _files(root: Path, value: object, label: str) -> None:
    for index, raw in enumerate(_array(value, label)):
        path = Path(_text(raw, f"{label}[{index}]"))
        if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
            raise CampaignRecoveryAuditError(f"{label}[{index}] references a missing file")


def audit_release(root: Path = Path("."), *, manifest: Path | None = None) -> None:
    """Validate the checked-in v2.7 recovery matrix and guarantee record."""

    root = root.resolve()
    manifest_path = manifest or (root / "docs" / "research" / "v2.7-campaign-recovery-v1.json")
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignRecoveryAuditError("could not read v2.7 recovery record") from error
    if record.get("record_type") != "bounded_campaign_recovery_release":
        raise CampaignRecoveryAuditError("unexpected v2.7 recovery record type")
    if record.get("schema_version") != 1:
        raise CampaignRecoveryAuditError("unsupported v2.7 recovery record schema")
    if record.get("protocol_revision") != "modelsurgeon-v2.7-campaign-recovery-v1":
        raise CampaignRecoveryAuditError("unexpected v2.7 recovery protocol revision")
    if record.get("status") != "bounded_release_boundary":
        raise CampaignRecoveryAuditError(
            "v2.7 recovery status must remain "
            "bounded_release_boundary"
        )

    dependencies = _array(record.get("dependency_evidence"), "dependency_evidence")
    seen: set[str] = set()
    for index, raw in enumerate(dependencies):
        item = _object(raw, f"dependency_evidence[{index}]")
        key = _text(item.get("key"), f"dependency_evidence[{index}].key")
        if key in seen or key not in _DEPENDENCIES:
            raise CampaignRecoveryAuditError(f"unexpected or duplicate dependency {key}")
        seen.add(key)
        if item.get("status") != "merged":
            raise CampaignRecoveryAuditError(f"dependency {key} is not marked merged")
        issue, short_commit = _DEPENDENCIES[key]
        commit = _text(item.get("commit"), f"dependency {key}.commit")
        if not _COMMIT.fullmatch(commit) or not commit.startswith(short_commit):
            raise CampaignRecoveryAuditError(f"dependency {key} identity drifted")
        if item.get("issue") != issue:
            raise CampaignRecoveryAuditError(f"dependency {key} issue drifted")
        _files(root, item.get("artifacts"), f"dependency {key}.artifacts")
        _files(root, item.get("tests"), f"dependency {key}.tests")
    if seen != set(_DEPENDENCIES):
        raise CampaignRecoveryAuditError("dependency evidence is incomplete")

    protocol = _object(record.get("protocol"), "protocol")
    _files(root, [protocol.get("fixture")], "protocol.fixture")
    fixture = json.loads((root / _text(protocol["fixture"], "protocol.fixture")).read_text())
    cells = _array(fixture.get("cells"), "fixture.cells")
    cell_ids = [
        _text(_object(item, f"fixture.cells[{index}]").get("cell_id"), f"cell {index}.cell_id")
        for index, item in enumerate(cells)
    ]
    if len(cell_ids) != len(set(cell_ids)) or set(cell_ids) != _EXPECTED_CELLS:
        raise CampaignRecoveryAuditError("recovery matrix cells are incomplete or duplicated")
    if protocol.get("cell_count") != len(_EXPECTED_CELLS):
        raise CampaignRecoveryAuditError("recovery matrix cell count drifted")
    if protocol.get("process_restart") != "tested_subprocess_reopen":
        raise CampaignRecoveryAuditError("process restart evidence is not explicit")
    if protocol.get("fault_injection") != "bounded_sql_write_checkpoints":
        raise CampaignRecoveryAuditError("fault injection boundary is not explicit")
    if protocol.get("ui_reconnect_correctness") != "not_claimed":
        raise CampaignRecoveryAuditError("UI reconnect must not be correctness evidence")

    guarantees = set(_array(record.get("guarantees"), "guarantees"))
    if guarantees != _GUARANTEES:
        raise CampaignRecoveryAuditError("recovery guarantees are incomplete or overclaimed")
    negative = _object(record.get("negative_and_inconclusive"), "negative_and_inconclusive")
    if negative.get("retained") is not True or negative.get("failed_cells_visible") is not True:
        raise CampaignRecoveryAuditError("negative and inconclusive cells must remain visible")

    _files(root, record.get("tests"), "tests")
    _files(root, record.get("documentation"), "documentation")
    _files(root, record.get("examples"), "examples")
    commands = _array(record.get("reproduction_commands"), "reproduction_commands")
    if not any("audit_v27_campaign_recovery.py" in str(command) for command in commands):
        raise CampaignRecoveryAuditError("recovery audit command is missing")
    if not any("test_campaign_recovery.py" in str(command) for command in commands):
        raise CampaignRecoveryAuditError("focused recovery test command is missing")
    quality = _object(record.get("quality_gate"), "quality_gate")
    quality_commands = [
        str(item)
        for item in _array(quality.get("commands"), "quality_gate.commands")
    ]
    for required in (
        "git diff --check",
        "ruff check src tests",
        "mypy src/modelsurgeon",
        "pytest -q",
    ):
        if not any(required in command for command in quality_commands):
            raise CampaignRecoveryAuditError(f"quality gate is missing {required}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v2.7 campaign recovery and determinism boundary verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

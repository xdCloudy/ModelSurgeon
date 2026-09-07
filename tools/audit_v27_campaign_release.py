"""Audit the closed v2.7 stateful conversational campaign release boundary."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

from modelsurgeon.conversation import CAMPAIGN_STATE_SCHEMA_VERSION, CampaignState

try:
    from tools.audit_v27_campaign_recovery import audit_release as audit_recovery_release
except ModuleNotFoundError:  # pragma: no cover - direct ``python tools/...`` execution
    from audit_v27_campaign_recovery import audit_release as audit_recovery_release


class CampaignReleaseAuditError(ValueError):
    """Raised when the v2.7 release boundary is incomplete or overclaims."""


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DEPENDENCIES = {
    "v27-canonical-state": (469, "0049ccdc5a4771ee0fc38d0b23a1d1fb2aac262f"),
    "v27-lifecycle": (470, "270bf0e9e1e5aca953d5640938173ca29f319e3e"),
    "v27-stale": (471, "8b3eadce4e72ffef80e809f858834fcd8e3aaab0"),
    "v27-summary": (472, "67a737159d7c5d63daab7c52efda4664bfd910d9"),
    "v27-recovery-matrix": (473, "535a13b5802e88db1c9c98d3344b47f26bc2f702"),
}
_STATE_FIELDS = {
    "campaign_id",
    "session_id",
    "run_id",
    "source_model_digest",
    "spec",
    "policy_state",
    "approval",
    "evidence_cursor",
    "provider_context",
    "budget",
    "lifecycle",
    "outcome",
    "state_version",
    "last_transition_id",
    "provenance",
}
_SUPPORTED_INTERRUPTION_IDS = {
    "pause_resume",
    "cancel",
    "reconnect",
    "restart",
    "process_restart",
    "transcript_loss",
}
_REQUIRED_DOCS = {
    "ROADMAP.md",
    "docs/goal.md",
    "ARCHITECTURE.md",
    "README.md",
    "CHANGELOG.md",
    "docs/design/conversational-campaign-state.md",
    "docs/design/campaign-recovery-matrix.md",
    "docs/design/stale-conversational-replanning.md",
    "docs/design/conversational-summary.md",
    "docs/design/conversational-control-plane.md",
    "docs/release/v2.7-stateful-campaigns-boundary.md",
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CampaignReleaseAuditError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignReleaseAuditError(f"{label} must be non-empty text")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CampaignReleaseAuditError(f"{label} must be an array")
    return value


def _files(root: Path, value: object, label: str) -> None:
    values = _array(value, label)
    if not values:
        raise CampaignReleaseAuditError(f"{label} must not be empty")
    for index, raw in enumerate(values):
        relative = Path(_text(raw, f"{label}[{index}]"))
        if relative.is_absolute() or ".." in relative.parts or not (root / relative).is_file():
            raise CampaignReleaseAuditError(f"{label}[{index}] references a missing file")


def _require_strings(value: object, label: str) -> list[str]:
    result = [_text(item, f"{label}[{index}]") for index, item in enumerate(_array(value, label))]
    if not result:
        raise CampaignReleaseAuditError(f"{label} must not be empty")
    return result


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignReleaseAuditError(f"could not read {label}") from error


def _check_dependencies(root: Path, record: dict[str, Any]) -> None:
    seen: set[str] = set()
    for index, raw in enumerate(_array(record.get("dependency_evidence"), "dependency_evidence")):
        item = _object(raw, f"dependency_evidence[{index}]")
        key = _text(item.get("key"), f"dependency_evidence[{index}].key")
        if key in seen or key not in _DEPENDENCIES:
            raise CampaignReleaseAuditError(f"unexpected or duplicate dependency {key}")
        seen.add(key)
        issue, expected_commit = _DEPENDENCIES[key]
        if item.get("issue") != issue or item.get("status") != "merged":
            raise CampaignReleaseAuditError(f"dependency {key} is not the merged dependency")
        commit = _text(item.get("commit"), f"dependency {key}.commit")
        if _COMMIT.fullmatch(commit) is None or commit != expected_commit:
            raise CampaignReleaseAuditError(f"dependency {key} identity drifted")
        _files(root, item.get("artifacts"), f"dependency {key}.artifacts")
        _files(root, item.get("tests"), f"dependency {key}.tests")
    if seen != set(_DEPENDENCIES):
        raise CampaignReleaseAuditError("v2.7 dependency evidence is incomplete")


def _check_schema(record: dict[str, Any]) -> None:
    schema = _object(record.get("state_schema"), "state_schema")
    if schema.get("record_type") != "conversational_campaign_state":
        raise CampaignReleaseAuditError("canonical state record type drifted")
    if schema.get("schema_version") != CAMPAIGN_STATE_SCHEMA_VERSION:
        raise CampaignReleaseAuditError("canonical state schema version drifted")
    declared = set(
        _require_strings(schema.get("authoritative_fields"), "state_schema.authoritative_fields")
    )
    actual = {item.name for item in fields(CampaignState) if item.name != "schema_version"}
    if declared != _STATE_FIELDS or actual != _STATE_FIELDS:
        raise CampaignReleaseAuditError("canonical state field set drifted")
    if schema.get("transition_policy") != "expected_version_atomic_wal" or schema.get(
        "evidence_policy"
    ) != "append_only_cursor_addressed":
        raise CampaignReleaseAuditError("canonical transition/evidence policy drifted")


def audit_release(root: Path = Path("."), *, manifest: Path | None = None) -> None:
    """Validate the complete v2.7 milestone boundary and its evidence links."""

    root = root.resolve()
    manifest_path = manifest or (
        root / "docs" / "research" / "v2.7-stateful-campaign-release-v1.json"
    )
    record = _read_json(manifest_path, "v2.7 stateful campaign release record")
    if record.get("record_type") != "bounded_stateful_campaign_release":
        raise CampaignReleaseAuditError("unexpected v2.7 release record type")
    if record.get("schema_version") != 1:
        raise CampaignReleaseAuditError("unsupported v2.7 release schema")
    if record.get("protocol_revision") != "modelsurgeon-v2.7-stateful-campaign-release-v1":
        raise CampaignReleaseAuditError("unexpected v2.7 release protocol revision")
    if record.get("milestone") != "v2.7" or record.get("status") != "closed":
        raise CampaignReleaseAuditError("v2.7 milestone status is not closed")

    _check_dependencies(root, record)
    _check_schema(record)

    authority = _object(record.get("authority_boundary"), "authority_boundary")
    expected_authority = {
        "structured_campaign_apis": "authoritative",
        "canonical_campaign_store": "authoritative",
        "chat_transcript": "not_authoritative",
        "conversation_summary": "bounded_non_authoritative",
        "provider_output": "not_authoritative",
        "ui_reconnect": "not_correctness_evidence",
    }
    for key, expected in expected_authority.items():
        if authority.get(key) != expected:
            raise CampaignReleaseAuditError(f"authority boundary drifted: {key}")

    recovery = _object(record.get("recovery_evidence"), "recovery_evidence")
    matrix_manifest = root / _text(
        recovery.get("matrix_manifest"), "recovery_evidence.matrix_manifest"
    )
    _files(root, [recovery.get("matrix_manifest")], "recovery_evidence.matrix_manifest")
    audit_recovery_release(root, manifest=matrix_manifest)
    fixture = _read_json(
        root / _text(recovery["matrix_fixture"], "recovery_evidence.matrix_fixture"),
        "recovery matrix fixture",
    )
    _files(root, [recovery.get("matrix_fixture")], "recovery_evidence.matrix_fixture")
    cells = _array(fixture.get("cells"), "recovery matrix cells")
    cell_ids = {
        _text(_object(item, "recovery matrix cell").get("cell_id"), "recovery cell ID")
        for item in cells
    }
    if recovery.get("cell_count") != len(cells) or len(cells) != 12:
        raise CampaignReleaseAuditError("recovery matrix cell count drifted")
    if not _SUPPORTED_INTERRUPTION_IDS.issubset(cell_ids):
        raise CampaignReleaseAuditError("supported interruption evidence is incomplete")
    if recovery.get("comparison") != [
        "canonical_state_digest",
        "evidence_cursor",
        "retained_evidence_and_artifacts",
        "next_action",
        "hard_constraints",
        "resource_budget",
        "source_model_digest",
        "provenance",
    ]:
        raise CampaignReleaseAuditError("recovery comparison boundary drifted")
    if recovery.get("process_restart") != "tested_subprocess_reopen":
        raise CampaignReleaseAuditError("process restart recovery is not explicit")
    if recovery.get("atomic_faults") != "bounded_sql_write_checkpoints_rollback":
        raise CampaignReleaseAuditError("atomic fault recovery is not explicit")

    stale = _object(record.get("stale_expired_boundary"), "stale_expired_boundary")
    expected_stale = {
        "stale_replan": "fail_closed_rebuild_from_canonical_state",
        "expired_approval": "record_expiry_then_refuse_resume",
        "cancelled_replay": "fail_closed",
        "altered_summary": "fail_closed",
    }
    for key, expected in expected_stale.items():
        if stale.get(key) != expected:
            raise CampaignReleaseAuditError(f"stale/expired boundary drifted: {key}")

    limitations = _object(record.get("operational_limitations"), "operational_limitations")
    expected_limits = {
        "writer_model": "single_store_owner_per_campaign",
        "in_process_lock": "local_only",
        "multi_writer_concurrency": "unsupported_release_guarantee",
        "distributed_recovery": "unsupported",
        "hostile_process_containment": "not_claimed",
        "live_provider_and_model_quality": "not_run",
    }
    for key, expected in expected_limits.items():
        if limitations.get(key) != expected:
            raise CampaignReleaseAuditError(f"operational limitation drifted: {key}")

    duplicate = _object(record.get("duplicate_scope_review"), "duplicate_scope_review")
    if duplicate.get("status") != "passed" or duplicate.get("no_duplicate_authority") is not True:
        raise CampaignReleaseAuditError("duplicate-scope review did not pass")
    issues = {
        _object(item, "duplicate scope item").get("issue")
        for item in _array(
            duplicate.get("reviewed_issues"), "duplicate_scope_review.reviewed_issues"
        )
    }
    if issues != {415, 427, 428}:
        raise CampaignReleaseAuditError("duplicate-scope review must cover #415, #427, and #428")

    _files(root, record.get("documentation"), "documentation")
    _files(root, record.get("tests"), "tests")
    _files(root, record.get("examples"), "examples")
    commands = _require_strings(record.get("reproduction_commands"), "reproduction_commands")
    required_commands = (
        "audit_v27_campaign_release.py",
        "docs/examples/campaign_recovery.py",
        "docs/examples/stale_replanning.py",
        "pytest tests/test_v27_campaign_release.py",
    )
    for required in required_commands:
        if not any(required in command for command in commands):
            raise CampaignReleaseAuditError(f"reproduction commands are missing {required}")

    quality = _object(record.get("quality_gate"), "quality_gate")
    if quality.get("status") != "passed":
        raise CampaignReleaseAuditError("v2.7 quality gate is not recorded as passed")
    quality_commands = _require_strings(quality.get("commands"), "quality_gate.commands")
    for required in (
        "git diff --check",
        "ruff check src tests",
        "mypy src/modelsurgeon",
        "pytest -q",
    ):
        if not any(required in command for command in quality_commands):
            raise CampaignReleaseAuditError(f"quality gate is missing {required}")
    _text(quality.get("revision"), "quality_gate.revision")
    _require_strings(quality.get("outcomes"), "quality_gate.outcomes")
    _require_strings(quality.get("known_skips"), "quality_gate.known_skips")

    declared_docs = set(_require_strings(record.get("documentation"), "documentation"))
    missing_docs = _REQUIRED_DOCS.difference(declared_docs)
    if missing_docs:
        raise CampaignReleaseAuditError(
            f"documentation coverage is incomplete: {sorted(missing_docs)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v2.7 stateful conversational campaign release boundary verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

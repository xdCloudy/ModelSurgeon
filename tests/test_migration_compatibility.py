"""Compatibility guarantees for the bounded v2.0 migration boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.audit_migration_compatibility import audit_release
from typer.testing import CliRunner

from modelsurgeon.cli.app import app
from modelsurgeon.config import Settings
from modelsurgeon.migration import (
    MIGRATION_CAPABILITY_MATRIX,
    MigrationKind,
    MigrationRefusal,
    detect_migration_kind,
    migrate_record,
    migrate_v20_campaign,
    migrate_v20_config,
    migrate_v20_evidence,
    migration_matrix_records,
)
from modelsurgeon.optimization import build_optimize_plan

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_v20_config_migration_defaults_to_no_llm_and_preserves_direct_api() -> None:
    result = migrate_v20_config(_fixture("v2.0-config.json"))

    assert result.kind is MigrationKind.CONFIG
    assert result.changed
    assert result.record["provider"]["kind"] == "none"  # type: ignore[index]
    settings = Settings.model_validate(result.record)
    plan = build_optimize_plan(settings)
    assert plan.executable
    assert settings.provider.kind.value == "none"


def test_v20_campaign_migration_preserves_identity_evidence_and_artifact() -> None:
    source = _fixture("v2.0-campaign-run.json")
    result = migrate_v20_campaign(source)
    migrated = result.to_record()

    assert result.changed
    assert migrated["schema_version"] == 3
    assert migrated["run_id"] == source["run_id"]
    assert migrated["plan_digest"] == source["plan_digest"]
    assert migrated["source_artifact_digest"] == source["source_artifact_digest"]
    assert migrated["accepted_artifact_digest"] == source["accepted_artifact_digest"]
    assert migrated["approval_audit"] == []
    assert len(migrated["stages"]) == 11  # type: ignore[arg-type]


def test_v20_campaign_unknown_fields_are_not_silently_dropped() -> None:
    source = _fixture("v2.0-campaign-run.json")
    source["future_field"] = "must refuse"

    with pytest.raises(MigrationRefusal, match="unknown fields"):
        migrate_v20_campaign(source)


def test_v20_evidence_migration_retains_unknown_as_inconclusive() -> None:
    result = migrate_v20_evidence(_fixture("v2.0-campaign-evidence.json"))
    migrated = result.to_record()

    assert migrated["record_type"] == "canonical_campaign_evidence"
    assert migrated["outcome"] == "unknown"
    assert migrated["inconclusive"] is True
    assert migrated["source_digest"] == (
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )
    assert migrated["provenance"]["run_id"] == "run_v20_fixture"  # type: ignore[index]


def test_current_records_are_read_validated_and_not_rewritten() -> None:
    legacy = migrate_v20_evidence(_fixture("v2.0-campaign-evidence.json")).to_record()
    result = migrate_v20_evidence(legacy)

    assert not result.changed
    assert result.to_record() == legacy


@pytest.mark.parametrize(
    ("kind", "payload", "message"),
    [
        (MigrationKind.CONFIG, {"schema_version": 2}, "unsupported"),
        (
            MigrationKind.CAMPAIGN,
            {"record_type": "autonomous_optimize_run", "schema_version": 1},
            "unsupported",
        ),
        (
            MigrationKind.EVIDENCE,
            {
                "record_type": "v2.0_campaign_evidence",
                "schema_version": 1,
                "evidence_id": "evidence_bad",
            },
            "shape mismatch",
        ),
    ],
)
def test_schema_mismatch_and_incomplete_records_fail_closed(
    kind: MigrationKind, payload: dict[str, object], message: str
) -> None:
    with pytest.raises(MigrationRefusal, match=message):
        migrate_record(payload, kind=kind)


def test_provider_and_stage_only_evidence_cells_are_unsupported() -> None:
    config = _fixture("v2.0-config.json")
    config["provider"] = {
        "kind": "local",
        "provider_id": "local",
        "model_id": "chat",
        "model_revision": "revision",
    }
    with pytest.raises(MigrationRefusal, match="non-none conversational provider"):
        migrate_v20_config(config)

    with pytest.raises(MigrationRefusal, match="source artifact lineage"):
        migrate_v20_evidence({"record_type": "stage_result", "schema_version": 1})


def test_migration_kind_detection_rejects_ambiguous_json() -> None:
    with pytest.raises(MigrationRefusal, match="unambiguous"):
        detect_migration_kind({"schema_version": 1, "value": "ambiguous"})


def test_cli_migration_writes_without_overwriting_and_reports_json(tmp_path: Path) -> None:
    source = FIXTURES / "v2.0-config.json"
    output = tmp_path / "config.json"
    result = CliRunner().invoke(
        app,
        [
            "migrate",
            str(source),
            "--output",
            str(output),
            "--json",
        ],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8"))["provider"]["kind"] == "none"
    assert json.loads(result.output)["record_type"] == "modelsurgeon_migration_report"

    refused = CliRunner().invoke(
        app,
        ["migrate", str(source), "--output", str(output)],
        color=False,
    )
    assert refused.exit_code == 2


def test_matrix_declares_supported_and_fail_closed_cells() -> None:
    records = tuple(cell.to_record() for cell in MIGRATION_CAPABILITY_MATRIX)
    assert any(item["status"] == "supported" for item in records)
    assert any(item["status"] == "unsupported" for item in records)
    assert all(item["guarantee"] for item in records)


def test_docs_and_machine_release_record_match_the_implementation() -> None:
    release = json.loads(
        (ROOT / "docs" / "research" / "v3.0-migration-compatibility-v1.json").read_text(
            encoding="utf-8"
        )
    )
    docs = (ROOT / "docs" / "migration.md").read_text(encoding="utf-8").replace("`", "")
    matrix_sources = {(item["source"], item["target"]) for item in migration_matrix_records()}
    release_cells = tuple(release["supported_cells"])

    assert release["status"] == "implemented"
    assert release["issue"] == 489
    assert len(release["deprecation_policy"]) == 4
    assert any(item["state"] == "unsupported" for item in release["deprecation_policy"])
    for cell in release_cells:
        assert (cell["source"], cell["target"]) in matrix_sources
        assert cell["source"] in docs
        assert cell["target"] in docs
    for fixture in release["fixtures"]:
        assert (ROOT / fixture).is_file()
    for path in ("README.md", "ROADMAP.md", "ARCHITECTURE.md", "docs/api-compatibility.md"):
        assert "migration" in (ROOT / path).read_text(encoding="utf-8").lower()


def test_focused_migration_release_audit_passes() -> None:
    audit_release(ROOT)

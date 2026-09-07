"""Tests for the v2.6 bounded conversational tool release boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.audit_v26_tool_release import ToolReleaseAuditError, audit_release

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "v2.6-bounded-conversational-tool-release-v1.json"


def _altered_manifest(tmp_path: Path, **changes: object) -> Path:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record.update(changes)
    path = tmp_path / "release.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_v26_tool_release_boundary_is_complete() -> None:
    audit_release(ROOT)


def test_release_audit_rejects_changed_dependency_identity(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["dependency_evidence"][0]["commit"] = "0" * 40
    path = tmp_path / "release.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ToolReleaseAuditError, match="unexpected commit"):
        audit_release(ROOT, manifest=path)


def test_release_audit_withholds_live_evidence_claims(tmp_path: Path) -> None:
    path = _altered_manifest(
        tmp_path,
        evidence_policy={
            "fixture_evidence": "contract and boundary evidence only",
            "live_provider_benchmarks": "measured",
            "live_campaigns": "not_run",
            "live_model_quality": "not_run",
            "negative_and_inconclusive": "not claimed",
        },
    )

    with pytest.raises(ToolReleaseAuditError, match="live_provider_benchmarks"):
        audit_release(ROOT, manifest=path)


def test_release_audit_requires_the_process_isolation_limitation(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["adversarial_evidence"]["process_isolation"] = "claimed"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ToolReleaseAuditError, match="process isolation"):
        audit_release(ROOT, manifest=path)

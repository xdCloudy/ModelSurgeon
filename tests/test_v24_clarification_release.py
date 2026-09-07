"""Acceptance tests for the frozen v2.4 clarification release boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.audit_v24_clarification_release import (
    ClarificationReleaseAuditError,
    audit_release,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "v2.4-clarification-release-v1.json"


def _altered_manifest(tmp_path: Path, **changes: object) -> Path:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record.update(changes)
    path = tmp_path / "release.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_v24_clarification_release_boundary_is_complete() -> None:
    audit_release(ROOT)


def test_release_audit_rejects_changed_dependency_identity(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["dependency_evidence"][0]["commit"] = "0" * 40
    path = tmp_path / "release.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ClarificationReleaseAuditError, match="unexpected commit"):
        audit_release(ROOT, manifest=path)


def test_release_audit_rejects_general_language_claim(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["clarification_contract"]["general_natural_language_understanding"] = "supported"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ClarificationReleaseAuditError, match="general_natural_language"):
        audit_release(ROOT, manifest=path)


def test_release_audit_rejects_dropped_negative_evidence(tmp_path: Path) -> None:
    path = _altered_manifest(
        tmp_path,
        evidence={
            "protocol_id": "v24-clarification-measurement-v1",
            "protocol_record": "tests/fixtures/clarification_measurement_v1.json",
            "source_corpus": "tests/fixtures/intent_compiler_corpus_v1.json",
            "retained_case_count": 11,
            "retained_cell_count": 44,
            "policies": ["schema_driven"],
            "schema_driven_metrics": {},
            "run_id": "run",
            "negative_and_inconclusive_retained": False,
            "live_provider_benchmarks": "not_run",
            "live_model_quality": "not_run",
            "provider_replay_seeds": [],
        },
    )

    with pytest.raises(ClarificationReleaseAuditError, match="negative"):
        audit_release(ROOT, manifest=path)

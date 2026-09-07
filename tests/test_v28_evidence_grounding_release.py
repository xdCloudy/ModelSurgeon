"""Acceptance tests for the closed v2.8 evidence-grounding release boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.audit_v28_evidence_grounding_release import (
    EvidenceGroundingReleaseAuditError,
    audit_release,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "v2.8-evidence-grounding-release-v1.json"


def test_checked_in_v28_release_audit_replays_all_dependencies_and_limits() -> None:
    audit_release(ROOT)


def test_release_manifest_declares_canonicality_and_claim_limits() -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert record["status"] == "closed"
    assert [item["issue"] for item in record["dependency_evidence"]] == [475, 476, 477, 478, 479]
    assert record["authority_boundary"]["provider_text"] == "not_authoritative"
    assert record["authority_boundary"]["prediction_values"] == "not_measurements"
    assert record["claim_policy"]["optimizer_proof"] == "forbidden"
    assert record["claim_policy"]["unavailable_measurements"] == "forbidden"
    assert record["factuality_evidence"]["thresholds"]["critical_escapes"] == 0
    assert record["factuality_evidence"]["negative_and_inconclusive_retained"] is True
    assert record["duplicate_scope_review"]["status"] == "passed"


def test_release_audit_fails_closed_on_manifest_status_drift(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["status"] = "candidate"
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(EvidenceGroundingReleaseAuditError, match="status is not closed"):
        audit_release(ROOT, manifest=manifest)


def test_release_audit_fails_closed_on_missing_deferred_explanation_type(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["limitations"]["deferred_explanation_types"].remove("optimizer_optimality_or_proof")
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(EvidenceGroundingReleaseAuditError, match="deferred explanation"):
        audit_release(ROOT, manifest=manifest)

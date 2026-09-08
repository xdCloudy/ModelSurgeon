"""Acceptance tests for the bounded v2.9 conversational security gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.audit_v29_conversational_security_release import (
    ConversationalSecurityReleaseAuditError,
    audit_release,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "v2.9-conversational-security-release-v1.json"


def test_checked_in_v29_gate_replays_corpus_and_checks_all_suites() -> None:
    audit_release(ROOT)


def test_manifest_preserves_limits_and_negative_evidence() -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert record["status"] == "closed_bounded_gate"
    assert [item["issue"] for item in record["dependency_evidence"]] == [481, 482, 483, 484]
    assert record["policy_contract"]["prompt_or_provider_authority"] == "never_authoritative"
    assert record["isolation_contract"]["hostile_process_containment"] == "not_claimed"
    assert record["adversarial_evidence"]["unresolved_cases_retained"] is True
    assert record["clean_environment_evidence"]["locked_dependencies"] is True


def test_gate_fails_closed_on_status_drift(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["status"] = "candidate"
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ConversationalSecurityReleaseAuditError, match="status is not closed"):
        audit_release(ROOT, manifest=manifest)


def test_gate_fails_closed_if_process_containment_is_overclaimed(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["isolation_contract"]["hostile_process_containment"] = "supported"
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ConversationalSecurityReleaseAuditError, match="process-containment"):
        audit_release(ROOT, manifest=manifest)

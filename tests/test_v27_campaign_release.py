"""Acceptance tests for the closed v2.7 stateful campaign boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tools.audit_v27_campaign_release import (
    CampaignReleaseAuditError,
    audit_release,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "v2.7-stateful-campaign-release-v1.json"


def test_v27_stateful_campaign_release_boundary_is_complete() -> None:
    audit_release(ROOT)


def test_release_examples_replay_canonical_recovery_and_replanning() -> None:
    recovery = subprocess.run(
        [sys.executable, str(ROOT / "docs/examples/campaign_recovery.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    recovered = json.loads(recovery.stdout)
    assert recovered["record_type"] == "conversational_campaign_state"
    assert recovered["lifecycle"] == "running"

    replan = subprocess.run(
        [sys.executable, str(ROOT / "docs/examples/stale_replanning.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert replan.stdout.strip().startswith("campaign_")


def test_release_audit_rejects_distributed_recovery_claim(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["operational_limitations"]["distributed_recovery"] = "supported"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(CampaignReleaseAuditError, match="distributed_recovery"):
        audit_release(ROOT, manifest=path)


def test_release_audit_rejects_transcript_authority_claim(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["authority_boundary"]["chat_transcript"] = "authoritative"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(CampaignReleaseAuditError, match="chat_transcript"):
        audit_release(ROOT, manifest=path)

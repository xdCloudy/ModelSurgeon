"""Tests for the machine-readable v2.7 campaign recovery boundary."""

from pathlib import Path

from tools.audit_v27_campaign_recovery import audit_release

ROOT = Path(__file__).resolve().parents[1]


def test_v27_campaign_recovery_boundary_is_complete() -> None:
    audit_release(ROOT)

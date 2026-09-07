"""Tests for the v2.2 provider-layer release boundary record."""

from pathlib import Path

from tools.audit_v22_provider_release import audit_release

ROOT = Path(__file__).resolve().parents[1]


def test_v22_provider_release_boundary_is_complete() -> None:
    audit_release(ROOT)

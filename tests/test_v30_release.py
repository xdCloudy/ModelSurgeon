"""Executable contract checks for the bounded v3.0 product release."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tools.audit_v30_release import V30ReleaseAuditError, audit_release

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "v3.0-product-release-v1.json"


def _copy_manifest(tmp_path: Path) -> Path:
    target = tmp_path / "release.json"
    target.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def test_v30_product_release_manifest_is_complete() -> None:
    audit_release(ROOT)


def test_manifest_preserves_authority_states_and_residual_risks() -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert record["status"] == "bounded_product_release_candidate"
    assert {item["issue"] for item in record["dependency_issues"]} == {
        430,
        485,
        486,
        487,
        488,
        489,
        490,
    }
    assert set(record["capability_states"]) == {
        "verified",
        "experimental",
        "unsupported",
        "unknown",
        "deferred_v3_1",
    }
    assert record["authority_boundary"]["text_llm_is_optimizer_authority"] is False
    assert record["authority_boundary"]["learned_meta_surgeon_is_measurement_authority"] is False
    assert any(item["id"] == "hostile_process" for item in record["residual_risks"])


def test_release_audit_rejects_status_drift(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["status"] = "released_without_limits"
    manifest = _copy_manifest(tmp_path)
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(V30ReleaseAuditError, match="status"):
        audit_release(ROOT, manifest=manifest)


def test_release_audit_rejects_missing_p0_boundary(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["p0_exit_evidence"] = record["p0_exit_evidence"][:-1]
    manifest = _copy_manifest(tmp_path)
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(V30ReleaseAuditError, match="P0 evidence inventory"):
        audit_release(ROOT, manifest=manifest)


def test_release_audit_rejects_authority_promotion(tmp_path: Path) -> None:
    record = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record["authority_boundary"]["text_llm_is_optimizer_authority"] = True
    manifest = _copy_manifest(tmp_path)
    manifest.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(V30ReleaseAuditError, match="authority boundary"):
        audit_release(ROOT, manifest=manifest)

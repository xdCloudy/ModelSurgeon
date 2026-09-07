from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from modelsurgeon.experiments import (
    EvidenceBundleExternalReference,
    EvidenceBundleLineage,
    EvidenceBundleMember,
    EvidenceBundleRedaction,
    EvidenceKeyRecord,
    EvidenceKeyStatus,
    EvidenceVerificationStatus,
    SignedEvidenceError,
    build_evidence_bundle,
    render_evidence_bundle_index,
    sign_evidence_bundle,
    verify_evidence_bundle,
)

KEY = b"local-test-key-for-evidence"


def _lineage() -> EvidenceBundleLineage:
    return EvidenceBundleLineage(
        "a" * 40,
        "b" * 64,
        ("c" * 64,),
        ("dataset-v1",),
        ("hwctx-v1",),
    )


def _signed(tmp_path, *, external: bool = False):
    (tmp_path / "config.json").write_text('{"seed": 7}\n', encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "summary.json").write_text("{}\n", encoding="utf-8")
    redactions = (
        EvidenceBundleRedaction("secret.txt", "credential material intentionally excluded"),
    )
    external_references = (
        EvidenceBundleExternalReference(
            "model.gguf", "d" * 64, 10, "https://example.invalid/model.gguf"
        ),
    ) if external else ()
    index = build_evidence_bundle(
        tmp_path,
        bundle_kind="consumer-repair-study",
        lineage=_lineage(),
        roles={"config.json": "config", "reports/summary.json": "report"},
        redactions=redactions,
        external_references=external_references,
    )
    key = EvidenceKeyRecord("key-2026", hashlib.sha256(KEY).hexdigest(), EvidenceKeyStatus.ACTIVE)
    return sign_evidence_bundle(index, key=key, key_material=KEY)


def test_signed_bundle_verifies_streaming_members_and_renders_without_secret(tmp_path) -> None:
    index = _signed(tmp_path)

    result = verify_evidence_bundle(tmp_path, index, key_material={"key-2026": KEY})

    assert result.status is EvidenceVerificationStatus.VERIFIED
    assert result.verified
    assert result.checked_members == 2
    assert result.claim_strength == "bounded_reproducibility"
    rendered = render_evidence_bundle_index(index)
    assert "local-test-key" not in rendered
    assert index.bundle_id in rendered


def test_byte_tampering_and_missing_members_fail_verification(tmp_path) -> None:
    index = _signed(tmp_path)
    (tmp_path / "config.json").write_text('{"seed": 8}\n', encoding="utf-8")

    tampered = verify_evidence_bundle(tmp_path, index, key_material={"key-2026": KEY})
    assert tampered.status is EvidenceVerificationStatus.FAILED
    assert any("changed" in issue for issue in tampered.issues)

    (tmp_path / "config.json").unlink()
    missing = verify_evidence_bundle(tmp_path, index, key_material={"key-2026": KEY})
    assert missing.status is EvidenceVerificationStatus.FAILED
    assert any("missing members" in issue for issue in missing.issues)


def test_external_reference_is_explicit_and_downgrades_offline_claim(tmp_path) -> None:
    index = _signed(tmp_path, external=True)

    result = verify_evidence_bundle(tmp_path, index, key_material={"key-2026": KEY})

    assert result.status is EvidenceVerificationStatus.INCOMPLETE
    assert not result.verified
    assert result.claim_strength == "bounded_reproducibility"
    assert any("external references" in issue for issue in result.issues)


def test_revoked_key_and_unsafe_paths_are_rejected() -> None:
    with pytest.raises(SignedEvidenceError, match="unsafe"):
        EvidenceBundleMember("../secret", "a" * 64, 1, "artifact")
    with pytest.raises(SignedEvidenceError, match="revoked"):
        EvidenceKeyRecord(
            "revoked",
            "a" * 64,
            EvidenceKeyStatus.REVOKED,
        )


def test_revocation_metadata_invalidates_an_existing_attestation(tmp_path) -> None:
    index = _signed(tmp_path)
    revoked = EvidenceKeyRecord(
        "key-2026",
        hashlib.sha256(KEY).hexdigest(),
        EvidenceKeyStatus.REVOKED,
        revocation_reason="key compromise",
    )
    revoked_index = replace(index, key_records=(revoked,))

    result = verify_evidence_bundle(tmp_path, revoked_index, key_material={"key-2026": KEY})

    assert result.status is EvidenceVerificationStatus.FAILED
    assert any("revoked" in issue for issue in result.issues)

"""Offline-verifiable, content-addressed optimization evidence bundles.

The bundle stores public key metadata but never signing key material.  The
reference implementation uses HMAC-SHA256 so verification remains available
in the standard library; deployments may replace the signer while preserving
the canonical payload and detached-attestation boundary.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Final

SIGNED_EVIDENCE_SCHEMA_VERSION: Final[int] = 1
SIGNATURE_ALGORITHM: Final[str] = "hmac-sha256"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_CHUNK_BYTES = 1024 * 1024


class SignedEvidenceError(ValueError):
    """Raised when a bundle cannot meet its immutable evidence contract."""


class EvidenceKeyStatus(StrEnum):
    ACTIVE = "active"
    ROTATED = "rotated"
    REVOKED = "revoked"


class EvidenceVerificationStatus(StrEnum):
    VERIFIED = "verified"
    INCOMPLETE = "incomplete"
    UNSIGNED = "unsigned"
    FAILED = "failed"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SignedEvidenceError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise SignedEvidenceError(f"{label} must be a lowercase SHA-256")
    return result


def _path(value: object, label: str) -> str:
    result = _text(value, label)
    if "\\" in result or result.startswith("/") or PurePosixPath(result).is_absolute():
        raise SignedEvidenceError(f"{label} must be a relative POSIX path")
    parts = PurePosixPath(result).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise SignedEvidenceError(f"{label} contains an unsafe path component")
    return result


@dataclass(frozen=True, slots=True)
class EvidenceBundleLimits:
    max_members: int = 10_000
    max_total_bytes: int = 1 << 40
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES

    def __post_init__(self) -> None:
        for label, value in (
            ("maximum members", self.max_members),
            ("maximum total bytes", self.max_total_bytes),
            ("chunk bytes", self.chunk_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SignedEvidenceError(f"{label} must be a positive integer")

    def to_record(self) -> dict[str, int]:
        return {
            "max_members": self.max_members,
            "max_total_bytes": self.max_total_bytes,
            "chunk_bytes": self.chunk_bytes,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundleMember:
    path: str
    digest: str
    size_bytes: int
    role: str

    def __post_init__(self) -> None:
        _path(self.path, "bundle member path")
        _digest(self.digest, "bundle member digest")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise SignedEvidenceError("bundle member size cannot be negative")
        _text(self.role, "bundle member role")

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundleRedaction:
    path: str
    reason: str
    claim_limit: str = "bounded_reproducibility"

    def __post_init__(self) -> None:
        _path(self.path, "redaction path")
        _text(self.reason, "redaction reason")
        _text(self.claim_limit, "redaction claim limit")

    def to_record(self) -> dict[str, str]:
        return {"path": self.path, "reason": self.reason, "claim_limit": self.claim_limit}


@dataclass(frozen=True, slots=True)
class EvidenceBundleExternalReference:
    path: str
    digest: str
    size_bytes: int
    locator: str
    availability: str = "external"

    def __post_init__(self) -> None:
        _path(self.path, "external reference path")
        _digest(self.digest, "external reference digest")
        if isinstance(self.size_bytes, bool) or self.size_bytes <= 0:
            raise SignedEvidenceError("external reference size must be positive")
        _text(self.locator, "external reference locator")
        if self.availability not in {"external", "available", "missing"}:
            raise SignedEvidenceError("external reference availability is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "locator": self.locator,
            "availability": self.availability,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundleLineage:
    source_commit: str
    config_digest: str
    parent_artifact_digests: tuple[str, ...]
    data_revisions: tuple[str, ...]
    hardware_contexts: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.source_commit, "source commit")
        _digest(self.config_digest, "bundle config digest")
        for label, values in (
            ("parent artifact digests", self.parent_artifact_digests),
            ("data revisions", self.data_revisions),
            ("hardware contexts", self.hardware_contexts),
        ):
            if values != tuple(sorted(set(values))) or not values:
                raise SignedEvidenceError(f"{label} must be non-empty and canonical")
        for digest in self.parent_artifact_digests:
            _digest(digest, "parent artifact digest")

    def to_record(self) -> dict[str, object]:
        return {
            "source_commit": self.source_commit,
            "config_digest": self.config_digest,
            "parent_artifact_digests": list(self.parent_artifact_digests),
            "data_revisions": list(self.data_revisions),
            "hardware_contexts": list(self.hardware_contexts),
        }


@dataclass(frozen=True, slots=True)
class EvidenceKeyRecord:
    key_id: str
    fingerprint: str
    status: EvidenceKeyStatus
    algorithm: str = SIGNATURE_ALGORITHM
    predecessor_key_id: str | None = None
    successor_key_id: str | None = None
    revocation_reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.key_id, "evidence key ID")
        _digest(self.fingerprint, "evidence key fingerprint")
        if self.algorithm != SIGNATURE_ALGORITHM:
            raise SignedEvidenceError("unsupported evidence signature algorithm")
        if self.predecessor_key_id is not None:
            _text(self.predecessor_key_id, "predecessor key ID")
        if self.successor_key_id is not None:
            _text(self.successor_key_id, "successor key ID")
        if self.status is EvidenceKeyStatus.REVOKED and not self.revocation_reason:
            raise SignedEvidenceError("revoked keys require a revocation reason")
        if self.status is not EvidenceKeyStatus.REVOKED and self.revocation_reason is not None:
            raise SignedEvidenceError("only revoked keys carry a revocation reason")

    def to_record(self) -> dict[str, object]:
        return {
            "key_id": self.key_id,
            "fingerprint": self.fingerprint,
            "status": self.status.value,
            "algorithm": self.algorithm,
            "predecessor_key_id": self.predecessor_key_id,
            "successor_key_id": self.successor_key_id,
            "revocation_reason": self.revocation_reason,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundleSignature:
    key_id: str
    algorithm: str
    signed_payload_digest: str
    signature: str

    def __post_init__(self) -> None:
        _text(self.key_id, "signature key ID")
        if self.algorithm != SIGNATURE_ALGORITHM:
            raise SignedEvidenceError("unsupported signature algorithm")
        _digest(self.signed_payload_digest, "signed payload digest")
        if not re.fullmatch(r"[0-9a-f]{64}", self.signature):
            raise SignedEvidenceError("signature must be canonical hexadecimal")

    def to_record(self) -> dict[str, str]:
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "signed_payload_digest": self.signed_payload_digest,
            "signature": self.signature,
        }


@dataclass(frozen=True, slots=True)
class EvidenceBundleIndex:
    bundle_kind: str
    members: tuple[EvidenceBundleMember, ...]
    lineage: EvidenceBundleLineage
    merkle_root: str
    redactions: tuple[EvidenceBundleRedaction, ...] = ()
    external_references: tuple[EvidenceBundleExternalReference, ...] = ()
    key_records: tuple[EvidenceKeyRecord, ...] = ()
    signature: EvidenceBundleSignature | None = None
    schema_version: int = SIGNED_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.bundle_kind, "bundle kind")
        if self.schema_version != SIGNED_EVIDENCE_SCHEMA_VERSION:
            raise SignedEvidenceError("unsupported evidence bundle schema")
        if not self.members:
            raise SignedEvidenceError("evidence bundles require members")
        paths = tuple(item.path for item in self.members)
        if paths != tuple(sorted(set(paths))):
            raise SignedEvidenceError("bundle members must be unique and sorted")
        for label, items in (
            ("redactions", self.redactions),
            ("external references", self.external_references),
            ("key records", self.key_records),
        ):
            item_ids = tuple(item.path if hasattr(item, "path") else item.key_id for item in items)
            if item_ids != tuple(sorted(set(item_ids))):
                raise SignedEvidenceError(f"{label} must be canonical and unique")
        member_paths = set(paths)
        declared_paths = {item.path for item in self.redactions} | {
            item.path for item in self.external_references
        }
        if member_paths & declared_paths:
            raise SignedEvidenceError("a path cannot be both bundled and redacted/external")
        _digest(self.merkle_root, "bundle Merkle root")
        if self.signature is not None and self.signature.key_id not in {
            item.key_id for item in self.key_records
        }:
            raise SignedEvidenceError("bundle signature references unknown key metadata")

    @property
    def bundle_id(self) -> str:
        return "evidence_bundle_" + hashlib.sha256(self._payload()).hexdigest()

    @property
    def claim_strength(self) -> str:
        return "bounded_reproducibility" if self.redactions or self.external_references else "full"

    def _payload_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "bundle_kind": self.bundle_kind,
            "members": [item.to_record() for item in self.members],
            "lineage": self.lineage.to_record(),
            "merkle_root": self.merkle_root,
            "redactions": [item.to_record() for item in self.redactions],
            "external_references": [item.to_record() for item in self.external_references],
            "key_records": [item.to_record() for item in self.key_records],
        }

    def _payload(self) -> bytes:
        return _canonical(self._payload_record())

    def to_record(self) -> dict[str, object]:
        record = {
            **self._payload_record(),
            "bundle_id": self.bundle_id,
            "claim_strength": self.claim_strength,
            "signature": None if self.signature is None else self.signature.to_record(),
        }
        return record


@dataclass(frozen=True, slots=True)
class EvidenceVerification:
    status: EvidenceVerificationStatus
    bundle_id: str
    issues: tuple[str, ...]
    checked_members: int
    checked_bytes: int
    claim_strength: str

    @property
    def verified(self) -> bool:
        return self.status is EvidenceVerificationStatus.VERIFIED

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "verified": self.verified,
            "bundle_id": self.bundle_id,
            "issues": list(self.issues),
            "checked_members": self.checked_members,
            "checked_bytes": self.checked_bytes,
            "claim_strength": self.claim_strength,
        }


def _file_digest(path: Path, chunk_bytes: int) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _merkle_root(members: tuple[EvidenceBundleMember, ...]) -> str:
    nodes = [
        hashlib.sha256(b"member\0" + _canonical(item.to_record())).digest()
        for item in members
    ]
    while len(nodes) > 1:
        if len(nodes) % 2:
            nodes.append(nodes[-1])
        nodes = [
            hashlib.sha256(b"node\0" + nodes[index] + nodes[index + 1]).digest()
            for index in range(0, len(nodes), 2)
        ]
    return nodes[0].hex()


def build_evidence_bundle(
    root: str | Path,
    *,
    bundle_kind: str,
    lineage: EvidenceBundleLineage,
    roles: Mapping[str, str] | None = None,
    redactions: tuple[EvidenceBundleRedaction, ...] = (),
    external_references: tuple[EvidenceBundleExternalReference, ...] = (),
    key_records: tuple[EvidenceKeyRecord, ...] = (),
    limits: EvidenceBundleLimits | None = None,
) -> EvidenceBundleIndex:
    """Hash a directory without loading member contents into memory."""

    resolved = Path(root).absolute().resolve(strict=False)
    if not resolved.is_dir() or resolved.is_symlink():
        raise SignedEvidenceError("bundle root must be a real directory")
    active_limits = limits or EvidenceBundleLimits()
    role_map = {} if roles is None else dict(roles)
    members: list[EvidenceBundleMember] = []
    total = 0
    for candidate in sorted(resolved.rglob("*")):
        if candidate.is_symlink():
            raise SignedEvidenceError("bundle trees cannot contain symbolic links")
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(resolved).as_posix()
        path = _path(relative, "bundle member path")
        if len(members) >= active_limits.max_members:
            raise SignedEvidenceError("bundle exceeds its member limit")
        digest, size = _file_digest(candidate, active_limits.chunk_bytes)
        total += size
        if total > active_limits.max_total_bytes:
            raise SignedEvidenceError("bundle exceeds its byte limit")
        members.append(EvidenceBundleMember(path, digest, size, role_map.get(path, "artifact")))
    ordered = tuple(members)
    if not ordered:
        raise SignedEvidenceError("bundle root contains no regular files")
    index = EvidenceBundleIndex(
        bundle_kind,
        ordered,
        lineage,
        _merkle_root(ordered),
        tuple(sorted(redactions, key=lambda item: item.path)),
        tuple(sorted(external_references, key=lambda item: item.path)),
        tuple(sorted(key_records, key=lambda item: item.key_id)),
    )
    declared = {item.path for item in index.redactions} | {
        item.path for item in index.external_references
    }
    if declared & {item.path for item in index.members}:
        raise SignedEvidenceError("declared redaction or external path is present locally")
    return index


def sign_evidence_bundle(
    index: EvidenceBundleIndex,
    *,
    key: EvidenceKeyRecord,
    key_material: bytes,
) -> EvidenceBundleIndex:
    """Attach a detached HMAC attestation; key material is never retained."""

    if key.status is EvidenceKeyStatus.REVOKED:
        raise SignedEvidenceError("revoked keys cannot sign bundles")
    if key.algorithm != SIGNATURE_ALGORITHM:
        raise SignedEvidenceError("unsupported signing key algorithm")
    if hashlib.sha256(key_material).hexdigest() != key.fingerprint:
        raise SignedEvidenceError("signing key fingerprint does not match metadata")
    records = (*tuple(item for item in index.key_records if item.key_id != key.key_id), key)
    unsigned = EvidenceBundleIndex(
        index.bundle_kind,
        index.members,
        index.lineage,
        index.merkle_root,
        index.redactions,
        index.external_references,
        tuple(sorted(records, key=lambda item: item.key_id)),
    )
    payload_digest = hashlib.sha256(unsigned._payload()).hexdigest()
    signature = hmac.new(key_material, unsigned._payload(), hashlib.sha256).hexdigest()
    return EvidenceBundleIndex(
        unsigned.bundle_kind,
        unsigned.members,
        unsigned.lineage,
        unsigned.merkle_root,
        unsigned.redactions,
        unsigned.external_references,
        unsigned.key_records,
        EvidenceBundleSignature(key.key_id, key.algorithm, payload_digest, signature),
    )


def verify_evidence_bundle(
    root: str | Path,
    index: EvidenceBundleIndex,
    *,
    key_material: Mapping[str, bytes] | None = None,
    limits: EvidenceBundleLimits | None = None,
) -> EvidenceVerification:
    """Verify index, Merkle tree, signatures, and local bytes in bounded memory."""

    active_limits = limits or EvidenceBundleLimits()
    issues: list[str] = []
    resolved = Path(root).absolute().resolve(strict=False)
    if not resolved.is_dir() or resolved.is_symlink():
        issues.append("bundle root is missing or not a real directory")
        return EvidenceVerification(
            EvidenceVerificationStatus.FAILED,
            index.bundle_id,
            tuple(issues),
            0,
            0,
            index.claim_strength,
        )
    checked_bytes = 0
    checked_members = 0
    expected_paths = {item.path for item in index.members}
    try:
        actual_paths = {
            candidate.relative_to(resolved).as_posix()
            for candidate in resolved.rglob("*")
            if candidate.is_file() and not candidate.is_symlink()
        }
    except OSError as error:
        issues.append(f"cannot enumerate bundle: {error}")
        actual_paths = set()
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        if missing:
            issues.append("missing members: " + ", ".join(missing))
        if extra:
            issues.append("unexpected members: " + ", ".join(extra))
    for member in index.members:
        if checked_members >= active_limits.max_members:
            issues.append("verification member limit exceeded")
            break
        path = resolved / Path(*member.path.split("/"))
        if not path.is_file() or path.is_symlink():
            continue
        try:
            digest, size = _file_digest(path, active_limits.chunk_bytes)
        except OSError as error:
            issues.append(f"cannot read {member.path}: {error}")
            continue
        checked_members += 1
        checked_bytes += size
        if checked_bytes > active_limits.max_total_bytes:
            issues.append("verification byte limit exceeded")
            break
        if digest != member.digest or size != member.size_bytes:
            issues.append(f"member bytes changed: {member.path}")
    if _merkle_root(index.members) != index.merkle_root:
        issues.append("Merkle root does not reconcile with the index")
    status = EvidenceVerificationStatus.VERIFIED
    if index.signature is None:
        status = EvidenceVerificationStatus.UNSIGNED
        issues.append("detached signature is missing")
    else:
        key = next(
            (item for item in index.key_records if item.key_id == index.signature.key_id),
            None,
        )
        material = None if key_material is None else key_material.get(index.signature.key_id)
        if key is None:
            issues.append("signature key metadata is missing")
        elif key.status is EvidenceKeyStatus.REVOKED:
            issues.append("signature key is revoked")
        elif material is None:
            issues.append("signature key material is unavailable")
        else:
            payload = index._payload()
            if hashlib.sha256(payload).hexdigest() != index.signature.signed_payload_digest:
                issues.append("signed payload digest does not reconcile")
            expected_signature = hmac.new(material, payload, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected_signature, index.signature.signature):
                issues.append("detached signature is invalid")
            if hashlib.sha256(material).hexdigest() != key.fingerprint:
                issues.append("verification key fingerprint does not reconcile")
    if index.external_references:
        issues.append("external references are unavailable to this offline verifier")
        if status is EvidenceVerificationStatus.VERIFIED:
            status = EvidenceVerificationStatus.INCOMPLETE
    if issues and status not in {
        EvidenceVerificationStatus.UNSIGNED,
        EvidenceVerificationStatus.INCOMPLETE,
    }:
        status = EvidenceVerificationStatus.FAILED
    return EvidenceVerification(
        status,
        index.bundle_id,
        tuple(issues),
        checked_members,
        checked_bytes,
        index.claim_strength,
    )


def render_evidence_bundle_index(index: EvidenceBundleIndex) -> str:
    """Render the signed index without embedding member bytes or secrets."""

    return _canonical(index.to_record()).decode("utf-8") + "\n"


def write_evidence_bundle_index(index: EvidenceBundleIndex, path: str | Path) -> None:
    """Write an index atomically enough for inspection outside the content tree."""

    destination = Path(path)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_text(render_evidence_bundle_index(index), encoding="utf-8")
    os.replace(temporary, destination)

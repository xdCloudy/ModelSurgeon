"""Offline local registry for immutable model, surgeon, and evidence artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast, runtime_checkable
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from modelsurgeon.experiments.artifacts import (
    ArtifactDigest,
    ArtifactStoreError,
    ContentAddressedArtifactStore,
)
from modelsurgeon.experiments.identity import canonical_identity_json

REGISTRY_SCHEMA_VERSION = 1
REGISTRY_BUNDLE_SCHEMA_VERSION = 1


class RegistryError(ValueError):
    """Raised when a registry operation would violate visibility or integrity."""


class RegistryOutcome(StrEnum):
    VERIFIED = "verified"
    INCOMPLETE = "incomplete"
    UNSIGNED = "unsigned"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RegistryArtifact:
    """Catalog metadata for one immutable content-addressed object."""

    digest: str
    kind: str
    size_bytes: int
    schema_version: int = 1
    license_id: str | None = None
    tags: tuple[str, ...] = ()
    source: bool = False
    accepted: bool = False
    externally_unresolved: bool = False
    lineage: tuple[str, ...] = ()
    signature: str | None = None

    def __post_init__(self) -> None:
        ArtifactDigest.parse(self.digest)
        if not self.kind.strip() or self.size_bytes < 0 or self.schema_version <= 0:
            raise RegistryError("artifact kind, size, and schema version must be valid")
        if self.license_id is not None and not self.license_id.strip():
            raise RegistryError("license identifier cannot be blank")
        if self.tags != tuple(sorted(set(self.tags))) or any(not tag.strip() for tag in self.tags):
            raise RegistryError("artifact tags must be sorted, unique, and non-empty")
        if self.lineage != tuple(sorted(set(self.lineage))):
            raise RegistryError("artifact lineage must be sorted and unique")
        if self.signature is not None and len(self.signature) != 64:
            raise RegistryError("artifact signature must be a SHA-256 HMAC")

    def to_record(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "schema_version": self.schema_version,
            "license_id": self.license_id,
            "tags": list(self.tags),
            "source": self.source,
            "accepted": self.accepted,
            "externally_unresolved": self.externally_unresolved,
            "lineage": list(self.lineage),
            "signature": self.signature,
        }


@dataclass(frozen=True, slots=True)
class RegistryLease:
    digest: str
    lease_id: str

    def __post_init__(self) -> None:
        ArtifactDigest.parse(self.digest)
        if not self.lease_id.strip():
            raise RegistryError("lease ID cannot be blank")

    def to_record(self) -> dict[str, str]:
        return {"digest": self.digest, "lease_id": self.lease_id}


@dataclass(frozen=True, slots=True)
class RegistryPage:
    artifacts: tuple[RegistryArtifact, ...]
    next_cursor: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "artifacts": [item.to_record() for item in self.artifacts],
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True, slots=True)
class RegistryVerification:
    digest: str
    outcome: RegistryOutcome
    detail: str

    def to_record(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "outcome": self.outcome.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RegistryGcResult:
    candidates: tuple[str, ...]
    deleted: tuple[str, ...]
    protected: tuple[str, ...]
    dry_run: bool

    def to_record(self) -> dict[str, object]:
        return {
            "candidates": list(self.candidates),
            "deleted": list(self.deleted),
            "protected": list(self.protected),
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True, slots=True)
class RegistryComparison:
    left: str
    right: str
    same_content: bool
    differences: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "left": self.left,
            "right": self.right,
            "same_content": self.same_content,
            "differences": list(self.differences),
        }


@runtime_checkable
class ArtifactRegistryProvider(Protocol):
    """Provider boundary; remote implementations must be explicit at the caller."""

    def list(self, *, kind: str | None = None) -> RegistryPage: ...

    def verify(self, reference: str) -> RegistryVerification: ...


def _record_from_raw(value: object) -> RegistryArtifact:
    if not isinstance(value, Mapping):
        raise RegistryError("registry artifact record must be an object")
    try:
        tags_raw = value["tags"]
        lineage_raw = value["lineage"]
        if not isinstance(tags_raw, list) or not all(isinstance(item, str) for item in tags_raw):
            raise RegistryError("registry artifact tags are malformed")
        if not isinstance(lineage_raw, list) or not all(
            isinstance(item, str) for item in lineage_raw
        ):
            raise RegistryError("registry artifact lineage is malformed")
        return RegistryArtifact(
            str(value["digest"]),
            str(value["kind"]),
            int(value["size_bytes"]),
            int(value.get("schema_version", 1)),
            None if value.get("license_id") is None else str(value["license_id"]),
            tuple(tags_raw),
            bool(value.get("source", False)),
            bool(value.get("accepted", False)),
            bool(value.get("externally_unresolved", False)),
            tuple(lineage_raw),
            None if value.get("signature") is None else str(value["signature"]),
        )
    except (KeyError, TypeError, ValueError, RegistryError) as error:
        if isinstance(error, RegistryError):
            raise
        raise RegistryError("registry artifact record is malformed") from error


class LocalArtifactRegistry:
    """Content-addressed local registry with explicit catalog visibility."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().absolute().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects = ContentAddressedArtifactStore(self.root / "objects")
        self.catalog_path = self.root / "catalog.json"
        self._artifacts: dict[str, RegistryArtifact] = {}
        self._aliases: dict[str, str] = {}
        self._leases: dict[str, RegistryLease] = {}
        self._references: dict[str, tuple[str, ...]] = {}
        self._load_catalog()

    def _load_catalog(self) -> None:
        if not self.catalog_path.exists():
            return
        try:
            raw = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RegistryError("registry catalog is unreadable or corrupt") from error
        if not isinstance(raw, Mapping) or raw.get("schema_version") != REGISTRY_SCHEMA_VERSION:
            raise RegistryError("registry catalog schema version is unsupported")
        artifacts = raw.get("artifacts")
        aliases = raw.get("aliases")
        leases = raw.get("leases")
        references = raw.get("references")
        if not isinstance(artifacts, list) or not isinstance(aliases, Mapping):
            raise RegistryError("registry catalog sections are malformed")
        self._artifacts = {}
        for value in artifacts:
            item = _record_from_raw(value)
            if item.digest in self._artifacts:
                raise RegistryError("registry catalog contains duplicate artifacts")
            self._artifacts[item.digest] = item
        self._aliases = {str(key): str(value) for key, value in aliases.items()}
        if not isinstance(leases, list):
            raise RegistryError("registry leases are malformed")
        self._leases = {}
        for value in leases:
            if not isinstance(value, Mapping):
                raise RegistryError("registry lease is malformed")
            lease = RegistryLease(str(value.get("digest")), str(value.get("lease_id")))
            self._leases[lease.lease_id] = lease
        if not isinstance(references, Mapping):
            raise RegistryError("registry references are malformed")
        self._references = {}
        for key, value in references.items():
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise RegistryError("registry references are malformed")
            self._references[str(key)] = tuple(sorted(set(value)))

    def _catalog_record(self) -> dict[str, object]:
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "artifacts": [self._artifacts[key].to_record() for key in sorted(self._artifacts)],
            "aliases": {key: self._aliases[key] for key in sorted(self._aliases)},
            "leases": [self._leases[key].to_record() for key in sorted(self._leases)],
            "references": {
                key: list(self._references[key]) for key in sorted(self._references)
            },
        }

    def _save_catalog(self) -> None:
        temporary = self.catalog_path.with_name(f".catalog-{uuid.uuid4().hex}.partial")
        temporary.write_text(canonical_identity_json(self._catalog_record()), encoding="utf-8")
        try:
            os.replace(temporary, self.catalog_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def resolve(self, reference: str) -> RegistryArtifact:
        digest = self._aliases.get(reference, reference)
        artifact = self._artifacts.get(digest)
        if artifact is None:
            raise RegistryError(f"unknown registry artifact or alias: {reference}")
        return artifact

    def publish_bytes(
        self,
        payload: bytes,
        *,
        kind: str,
        schema_version: int = 1,
        license_id: str | None = None,
        tags: Iterable[str] = (),
        source: bool = False,
        accepted: bool = False,
        externally_unresolved: bool = False,
        lineage: Iterable[str] = (),
        expected_digest: str | None = None,
        signature: str | None = None,
    ) -> RegistryArtifact:
        stored = self.objects.put_bytes(payload, expected_digest=expected_digest)
        artifact = RegistryArtifact(
            str(stored.metadata.digest),
            kind,
            stored.metadata.size_bytes,
            schema_version,
            license_id,
            tuple(sorted(set(tags))),
            source,
            accepted,
            externally_unresolved,
            tuple(sorted(set(lineage))),
            signature,
        )
        existing = self._artifacts.get(artifact.digest)
        if existing is not None:
            if existing.size_bytes != artifact.size_bytes or existing.kind != artifact.kind:
                raise RegistryError("immutable artifact identity conflicts with catalog metadata")
            return existing
        self._artifacts[artifact.digest] = artifact
        self._save_catalog()
        return artifact

    def publish_file(self, path: str | Path, **metadata: object) -> RegistryArtifact:
        source = Path(path)
        if not source.is_file() or source.is_symlink():
            raise RegistryError("registry source must be a regular file")
        stored = self.objects.put_file(source)
        schema_version = metadata.get("schema_version", 1)
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise RegistryError("artifact schema version must be an integer")
        return self.publish_bytes(
            stored.data_path.read_bytes(),
            kind=str(metadata["kind"]),
            schema_version=schema_version,
            license_id=(
                None if metadata.get("license_id") is None else str(metadata["license_id"])
            ),
            tags=cast(Iterable[str], metadata.get("tags", ())),
            source=bool(metadata.get("source", False)),
            accepted=bool(metadata.get("accepted", False)),
            externally_unresolved=bool(metadata.get("externally_unresolved", False)),
            lineage=cast(Iterable[str], metadata.get("lineage", ())),
            signature=(
                None if metadata.get("signature") is None else str(metadata["signature"])
            ),
            expected_digest=str(stored.metadata.digest),
        )

    def list(
        self,
        *,
        kind: str | None = None,
        tag: str | None = None,
        page_size: int = 100,
        cursor: str | None = None,
    ) -> RegistryPage:
        if page_size <= 0:
            raise RegistryError("page size must be positive")
        values = [
            item
            for item in sorted(self._artifacts.values(), key=lambda item: item.digest)
            if (kind is None or item.kind == kind) and (tag is None or tag in item.tags)
        ]
        if cursor is not None:
            values = [item for item in values if item.digest > cursor]
        page = tuple(values[:page_size])
        next_cursor = page[-1].digest if len(values) > page_size else None
        return RegistryPage(page, next_cursor)

    def inspect(self, reference: str) -> RegistryArtifact:
        return self.resolve(reference)

    def verify(self, reference: str) -> RegistryVerification:
        artifact = self.resolve(reference)
        try:
            stored = self.objects.get(artifact.digest)
        except (ArtifactStoreError, OSError) as error:
            return RegistryVerification(artifact.digest, RegistryOutcome.FAILED, str(error))
        if stored.metadata.size_bytes != artifact.size_bytes:
            return RegistryVerification(
                artifact.digest, RegistryOutcome.FAILED, "catalog size does not match object"
            )
        if artifact.signature is None:
            return RegistryVerification(
                artifact.digest,
                RegistryOutcome.VERIFIED,
                "content digest and catalog metadata verified",
            )
        return RegistryVerification(
            artifact.digest,
            RegistryOutcome.VERIFIED,
            "content digest and detached signature metadata verified",
        )

    def tag(self, reference: str, tag: str) -> RegistryArtifact:
        artifact = self.resolve(reference)
        if not tag.strip():
            raise RegistryError("tag cannot be blank")
        updated = RegistryArtifact(
            artifact.digest,
            artifact.kind,
            artifact.size_bytes,
            artifact.schema_version,
            artifact.license_id,
            tuple(sorted(set((*artifact.tags, tag)))),
            artifact.source,
            artifact.accepted,
            artifact.externally_unresolved,
            artifact.lineage,
            artifact.signature,
        )
        self._artifacts[artifact.digest] = updated
        self._save_catalog()
        return updated

    def set_alias(self, alias: str, reference: str) -> RegistryArtifact:
        artifact = self.resolve(reference)
        if not alias.strip() or alias in {".", ".."}:
            raise RegistryError("alias must be a non-special non-empty name")
        existing = self._aliases.get(alias)
        if existing is not None and existing != artifact.digest:
            raise RegistryError("aliases cannot silently change object identity")
        self._aliases[alias] = artifact.digest
        self._save_catalog()
        return artifact

    def remove_alias(self, alias: str) -> None:
        """Remove an alias without touching the immutable object it names."""

        if alias in self._aliases:
            del self._aliases[alias]
            self._save_catalog()

    def lease(self, reference: str, lease_id: str) -> RegistryLease:
        artifact = self.resolve(reference)
        lease = RegistryLease(artifact.digest, lease_id)
        existing = self._leases.get(lease.lease_id)
        if existing is not None and existing != lease:
            raise RegistryError("lease ID is already held for another artifact")
        self._leases[lease.lease_id] = lease
        self._save_catalog()
        return lease

    def release_lease(self, lease_id: str) -> None:
        if lease_id in self._leases:
            del self._leases[lease_id]
            self._save_catalog()

    def add_reference(self, owner: str, child: str) -> None:
        owner_artifact = self.resolve(owner)
        child_artifact = self.resolve(child)
        children = set(self._references.get(owner_artifact.digest, ()))
        children.add(child_artifact.digest)
        self._references[owner_artifact.digest] = tuple(sorted(children))
        self._save_catalog()

    def collect_garbage(self, *, dry_run: bool = True) -> RegistryGcResult:
        leased = {lease.digest for lease in self._leases.values()}
        aliased = set(self._aliases.values())
        referenced = {child for children in self._references.values() for child in children}
        protected = {
            item.digest
            for item in self._artifacts.values()
            if item.source or item.accepted or item.externally_unresolved
        }
        protected.update(leased | aliased | referenced)
        candidates = tuple(sorted(set(self._artifacts) - protected))
        deleted: list[str] = []
        if not dry_run:
            for digest in candidates:
                try:
                    stored = self.objects.get(digest)
                except (ArtifactStoreError, OSError):
                    continue
                shutil.rmtree(stored.directory)
                deleted.append(digest)
                self._artifacts.pop(digest, None)
            self._save_catalog()
        return RegistryGcResult(candidates, tuple(deleted), tuple(sorted(protected)), dry_run)

    def export_bundle(
        self,
        reference: str,
        destination: str | Path,
        *,
        signing_key: bytes | None = None,
        allow_overwrite: bool = False,
    ) -> Path:
        artifact = self.resolve(reference)
        if Path(destination).exists() and not allow_overwrite:
            raise RegistryError(f"refusing to overwrite export bundle: {destination}")
        stored = self.objects.get(artifact.digest)
        manifest: dict[str, object] = {
            "bundle_schema_version": REGISTRY_BUNDLE_SCHEMA_VERSION,
            "artifact": artifact.to_record(),
            "payload_digest": artifact.digest,
            "payload_size": artifact.size_bytes,
        }
        if signing_key is not None:
            manifest["signature"] = hmac.new(
                signing_key,
                canonical_identity_json(manifest).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(destination_path, "w") as archive:
            manifest_info = ZipInfo("manifest.json")
            manifest_info.compress_type = ZIP_STORED
            archive.writestr(manifest_info, canonical_identity_json(manifest))
            payload_info = ZipInfo("payload")
            payload_info.compress_type = ZIP_STORED
            archive.writestr(payload_info, stored.data_path.read_bytes())
        return destination_path

    def import_bundle(
        self,
        bundle: str | Path,
        *,
        signing_key: bytes | None = None,
        accepted_licenses: Sequence[str] = (),
        allow_unsigned: bool = False,
    ) -> RegistryArtifact:
        try:
            with ZipFile(bundle, "r") as archive:
                if set(archive.namelist()) != {"manifest.json", "payload"}:
                    raise RegistryError(
                        "registry bundle must contain exactly manifest.json and payload"
                    )
                manifest = json.loads(archive.read("manifest.json"))
                payload = archive.read("payload")
        except RegistryError:
            raise
        except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RegistryError("registry bundle is unreadable or corrupt") from error
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("bundle_schema_version") != REGISTRY_BUNDLE_SCHEMA_VERSION
        ):
            raise RegistryError("registry bundle schema version is unsupported")
        artifact_raw = manifest.get("artifact")
        artifact = _record_from_raw(artifact_raw)
        if (
            manifest.get("payload_digest") != artifact.digest
            or manifest.get("payload_size") != len(payload)
        ):
            raise RegistryError("registry bundle payload digest or size is inconsistent")
        actual_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
        if actual_digest != artifact.digest:
            raise RegistryError("registry bundle payload digest does not match manifest")
        license_id = artifact.license_id
        if accepted_licenses and license_id not in set(accepted_licenses):
            raise RegistryError(f"artifact license {license_id!r} is not accepted")
        signature = manifest.get("signature")
        if signature is None:
            if artifact.kind == "evidence" and not allow_unsigned:
                raise RegistryError("evidence bundles require a detached signature")
        elif signing_key is None:
            raise RegistryError("bundle signature is present but no verification key was supplied")
        else:
            unsigned_manifest = dict(manifest)
            unsigned_manifest.pop("signature", None)
            expected = hmac.new(
                signing_key,
                canonical_identity_json(unsigned_manifest).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not isinstance(signature, str) or not hmac.compare_digest(signature, expected):
                raise RegistryError("registry bundle signature verification failed")
        return self.publish_bytes(
            payload,
            kind=artifact.kind,
            schema_version=artifact.schema_version,
            license_id=artifact.license_id,
            tags=artifact.tags,
            source=artifact.source,
            accepted=artifact.accepted,
            externally_unresolved=artifact.externally_unresolved,
            lineage=artifact.lineage,
            expected_digest=artifact.digest,
            signature=None if signature is None else str(signature),
        )

    def compare(self, left: str, right: str) -> RegistryComparison:
        first = self.resolve(left)
        second = self.resolve(right)
        differences: list[str] = []
        if first.kind != second.kind:
            differences.append("kind")
        if first.size_bytes != second.size_bytes:
            differences.append("size_bytes")
        if first.schema_version != second.schema_version:
            differences.append("schema_version")
        if first.license_id != second.license_id:
            differences.append("license_id")
        if first.lineage != second.lineage:
            differences.append("lineage")
        same_content = first.digest == second.digest
        if not same_content:
            differences.append("digest")
        return RegistryComparison(first.digest, second.digest, same_content, tuple(differences))


__all__ = [
    "ArtifactRegistryProvider",
    "LocalArtifactRegistry",
    "RegistryArtifact",
    "RegistryComparison",
    "RegistryError",
    "RegistryGcResult",
    "RegistryLease",
    "RegistryOutcome",
    "RegistryPage",
    "RegistryVerification",
]

"""Immutable content-addressed filesystem artifacts with atomic publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Self

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.store import (
    ExperimentMetadataStore,
    StoredArtifactReference,
)

ARTIFACT_METADATA_SCHEMA_VERSION: Literal[1] = 1
ARTIFACT_DIGEST_ALGORITHM = "sha256"
_DEFAULT_CHUNK_BYTES = 1024 * 1024


class ArtifactStoreError(RuntimeError):
    """Raised when immutable artifact integrity or publication cannot be guaranteed."""


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    algorithm: str
    hexadecimal: str

    def __post_init__(self) -> None:
        if self.algorithm != ARTIFACT_DIGEST_ALGORITHM:
            raise ArtifactStoreError(f"unsupported artifact digest algorithm {self.algorithm!r}")
        if len(self.hexadecimal) != 64:
            raise ArtifactStoreError("SHA-256 artifact digests require 64 hexadecimal characters")
        if self.hexadecimal != self.hexadecimal.lower() or any(
            character not in "0123456789abcdef" for character in self.hexadecimal
        ):
            raise ArtifactStoreError("artifact digest must use canonical lowercase hexadecimal")

    @classmethod
    def parse(cls, value: str) -> Self:
        try:
            algorithm, hexadecimal = value.split(":", 1)
        except ValueError as error:
            raise ArtifactStoreError("artifact digest must use algorithm:hex form") from error
        return cls(algorithm, hexadecimal)

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.hexadecimal}"


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    digest: ArtifactDigest
    size_bytes: int
    schema_version: Literal[1] = ARTIFACT_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_METADATA_SCHEMA_VERSION:
            raise ArtifactStoreError(
                f"unsupported artifact metadata schema version {self.schema_version}"
            )
        if self.size_bytes < 0:
            raise ArtifactStoreError("artifact size cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "digest": str(self.digest),
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class StoredArtifact:
    metadata: ArtifactMetadata
    directory: Path
    data_path: Path
    metadata_path: Path


@dataclass(frozen=True, slots=True)
class PublishedArtifactReference:
    artifact: StoredArtifact
    reference: StoredArtifactReference


def _metadata_from_record(value: object) -> ArtifactMetadata:
    if not isinstance(value, dict) or set(value) != {"schema_version", "digest", "size_bytes"}:
        raise ArtifactStoreError("artifact metadata has missing or unknown fields")
    schema_version = value["schema_version"]
    size_bytes = value["size_bytes"]
    digest = value["digest"]
    if schema_version != ARTIFACT_METADATA_SCHEMA_VERSION:
        raise ArtifactStoreError("artifact metadata schema version is unsupported")
    if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
        raise ArtifactStoreError("artifact metadata size is invalid")
    if not isinstance(digest, str):
        raise ArtifactStoreError("artifact metadata digest is invalid")
    return ArtifactMetadata(ArtifactDigest.parse(digest), size_bytes)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_tree(path: Path) -> None:
    if path.exists() or path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)


class ContentAddressedArtifactStore:
    """Store immutable byte streams under SHA-256 identities without overwriting content."""

    def __init__(self, root: str | Path, *, chunk_bytes: int = _DEFAULT_CHUNK_BYTES) -> None:
        if chunk_bytes <= 0:
            raise ArtifactStoreError("artifact copy chunk size must be positive")
        self.root = Path(root).expanduser().absolute().resolve(strict=False)
        self.chunk_bytes = chunk_bytes
        self.algorithm_root = self.root / ARTIFACT_DIGEST_ALGORITHM
        self.algorithm_root.mkdir(parents=True, exist_ok=True)

    def _paths(self, digest: ArtifactDigest) -> tuple[Path, Path, Path]:
        directory = self.algorithm_root / digest.hexadecimal[:2] / digest.hexadecimal[2:]
        return directory, directory / "data", directory / "metadata.json"

    def _staging_directory(self, parent: Path) -> Path:
        parent.mkdir(parents=True, exist_ok=True)
        return parent / f".partial-{uuid.uuid4().hex}"

    def _load_metadata(self, path: Path) -> ArtifactMetadata:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ArtifactStoreError("artifact metadata is incomplete") from error
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ArtifactStoreError("artifact metadata is unreadable or corrupt") from error
        return _metadata_from_record(value)

    def _verify_existing(self, digest: ArtifactDigest) -> StoredArtifact:
        directory, data_path, metadata_path = self._paths(digest)
        if directory.is_symlink() or data_path.is_symlink() or metadata_path.is_symlink():
            raise ArtifactStoreError("artifact storage paths cannot be symbolic links")
        if not directory.is_dir() or not data_path.is_file() or not metadata_path.is_file():
            raise ArtifactStoreError("artifact publication is incomplete")
        metadata = self._load_metadata(metadata_path)
        if metadata.digest != digest:
            raise ArtifactStoreError("artifact metadata digest does not match storage identity")
        size = data_path.stat().st_size
        if size != metadata.size_bytes:
            raise ArtifactStoreError("artifact size does not match immutable metadata")
        hasher = hashlib.sha256()
        with data_path.open("rb") as stream:
            while chunk := stream.read(self.chunk_bytes):
                hasher.update(chunk)
        if hasher.hexdigest() != digest.hexadecimal:
            raise ArtifactStoreError("artifact content digest does not match storage identity")
        return StoredArtifact(metadata, directory, data_path, metadata_path)

    def get(self, digest: ArtifactDigest | str) -> StoredArtifact:
        resolved = ArtifactDigest.parse(digest) if isinstance(digest, str) else digest
        return self._verify_existing(resolved)

    def exists(self, digest: ArtifactDigest | str) -> bool:
        resolved = ArtifactDigest.parse(digest) if isinstance(digest, str) else digest
        directory, _, _ = self._paths(resolved)
        if not directory.exists() and not directory.is_symlink():
            return False
        self._verify_existing(resolved)
        return True

    def put_bytes(
        self,
        payload: bytes,
        *,
        expected_digest: ArtifactDigest | str | None = None,
    ) -> StoredArtifact:
        from io import BytesIO

        return self.put_stream(BytesIO(payload), expected_digest=expected_digest)

    def put_file(
        self,
        source: str | Path,
        *,
        expected_digest: ArtifactDigest | str | None = None,
    ) -> StoredArtifact:
        path = Path(source)
        if not path.is_file() or path.is_symlink():
            raise ArtifactStoreError("artifact source must be a regular file")
        with path.open("rb") as stream:
            return self.put_stream(stream, expected_digest=expected_digest)

    def put_stream(
        self,
        stream: BinaryIO,
        *,
        expected_digest: ArtifactDigest | str | None = None,
    ) -> StoredArtifact:
        expected = (
            ArtifactDigest.parse(expected_digest)
            if isinstance(expected_digest, str)
            else expected_digest
        )
        staging_root = self.root / ".staging"
        staging = self._staging_directory(staging_root)
        staging.mkdir()
        staging_data = staging / "data"
        size = 0
        hasher = hashlib.sha256()
        try:
            with staging_data.open("xb") as output:
                while chunk := stream.read(self.chunk_bytes):
                    if not isinstance(chunk, bytes):
                        raise ArtifactStoreError("artifact stream must return bytes")
                    output.write(chunk)
                    hasher.update(chunk)
                    size += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            digest = ArtifactDigest(ARTIFACT_DIGEST_ALGORITHM, hasher.hexdigest())
            if expected is not None and expected != digest:
                raise ArtifactStoreError(
                    f"artifact digest mismatch: expected {expected}, computed {digest}"
                )
            final_directory, _, _ = self._paths(digest)
            if final_directory.exists() or final_directory.is_symlink():
                return self._verify_existing(digest)

            metadata = ArtifactMetadata(digest, size)
            staging_metadata = staging / "metadata.json"
            staging_metadata.write_text(
                canonical_identity_json(metadata.to_record()),
                encoding="utf-8",
            )
            _fsync_file(staging_metadata)
            _fsync_directory(staging)
            final_directory.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(staging, final_directory)
            except OSError as error:
                if final_directory.exists() or final_directory.is_symlink():
                    return self._verify_existing(digest)
                raise ArtifactStoreError("artifact publication failed atomically") from error
            _fsync_directory(final_directory.parent)
            return self._verify_existing(digest)
        finally:
            _remove_tree(staging)

    def publish_for_candidate(
        self,
        metadata_store: ExperimentMetadataStore,
        candidate_id: str,
        *,
        role: str,
        payload: bytes,
        reference_metadata: Mapping[str, object] | None = None,
        expected_digest: ArtifactDigest | str | None = None,
    ) -> PublishedArtifactReference:
        artifact = self.put_bytes(payload, expected_digest=expected_digest)
        reference = metadata_store.add_artifact_reference(
            candidate_id,
            role=role,
            digest=str(artifact.metadata.digest),
            metadata={} if reference_metadata is None else reference_metadata,
        )
        return PublishedArtifactReference(artifact, reference)

"""Format-neutral publication, reload, corruption, and rollback correctness gates."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from modelsurgeon.surgery.checkpoint_destination import (
    AtomicCheckpointDestination,
    CheckpointDestinationError,
)


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact cannot pass a fail-closed publication gate."""


class IntegrityFormat(StrEnum):
    HUGGINGFACE = "safetensors"
    GGUF = "GGUF"


class PublicationStage(StrEnum):
    SOURCE = "source"
    STAGE = "stage"
    CHECKSUM = "checksum"
    STRUCTURE = "structure"
    RELOAD = "reload"
    GENERATION = "generation"
    PROMOTE = "promote"
    CLEANUP = "cleanup"


class IntegrityOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class ArtifactSourceIntegrity:
    path: str
    digest: str
    size_bytes: int
    format: IntegrityFormat

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "format": self.format.value,
        }


@dataclass(frozen=True, slots=True)
class ArtifactValidationResult:
    outcome: IntegrityOutcome
    format: IntegrityFormat
    source: ArtifactSourceIntegrity | None
    destination: str
    completed_stages: tuple[PublicationStage, ...]
    failed_stage: PublicationStage | None
    artifact_digest: str | None
    reason: str | None

    @property
    def accepted(self) -> bool:
        return self.outcome is IntegrityOutcome.ACCEPTED

    def to_record(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "format": self.format.value,
            "source": None if self.source is None else self.source.to_record(),
            "destination": self.destination,
            "completed_stages": [stage.value for stage in self.completed_stages],
            "failed_stage": None if self.failed_stage is None else self.failed_stage.value,
            "artifact_digest": self.artifact_digest,
            "reason": self.reason,
        }


def _sha256(path: Path, *, max_bytes: int) -> tuple[str, int]:
    if max_bytes <= 0:
        raise ArtifactIntegrityError("artifact byte budget must be positive")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise ArtifactIntegrityError("artifact exceeds the byte budget")
                digest.update(chunk)
    except OSError as error:
        raise ArtifactIntegrityError(f"artifact cannot be read: {path}") from error
    return digest.hexdigest(), size


def validate_artifact_source(
    source: str | Path,
    *,
    format: IntegrityFormat,
    expected_digest: str | None = None,
    max_bytes: int = 8 * 1024 * 1024 * 1024,
) -> ArtifactSourceIntegrity:
    """Validate a source as a regular, non-symlink, bounded artifact."""
    path = Path(source).expanduser().absolute().resolve(strict=False)
    if not path.is_file() or path.is_symlink():
        raise ArtifactIntegrityError("source artifact must be a regular non-symlink file")
    digest, size = _sha256(path, max_bytes=max_bytes)
    if expected_digest is not None and digest != expected_digest:
        raise ArtifactIntegrityError("source checksum does not match the expected digest")
    return ArtifactSourceIntegrity(str(path), digest, size, format)


def recover_staging(directory: str | Path, *, destination_name: str) -> tuple[str, ...]:
    """Remove only staging siblings owned by one destination name."""
    root = Path(directory).expanduser().absolute().resolve(strict=False)
    if not root.is_dir() or root.is_symlink():
        raise ArtifactIntegrityError("staging recovery root must be a regular directory")
    prefix = f".{destination_name}."
    removed: list[str] = []
    for candidate in root.iterdir():
        if (
            candidate.name.startswith(prefix)
            and candidate.name.endswith(".staging")
            and candidate.is_file()
            and not candidate.is_symlink()
        ):
            candidate.unlink()
            removed.append(str(candidate))
    return tuple(sorted(removed))


def publish_validated_artifact(
    source: str | Path,
    destination: str | Path,
    *,
    format: IntegrityFormat,
    expected_digest: str | None = None,
    max_bytes: int = 8 * 1024 * 1024 * 1024,
    structure_check: Callable[[Path], None] | None = None,
    reload_check: Callable[[Path], None] | None = None,
    generation_check: Callable[[Path], None] | None = None,
    fault_at: PublicationStage | None = None,
) -> ArtifactValidationResult:
    """Run every bounded publication gate and publish only a complete child."""
    destination_path = Path(destination).expanduser().absolute().resolve(strict=False)
    completed: list[PublicationStage] = []
    source_integrity: ArtifactSourceIntegrity | None = None
    current_stage = PublicationStage.SOURCE
    try:
        source_integrity = validate_artifact_source(
            source,
            format=format,
            expected_digest=expected_digest,
            max_bytes=max_bytes,
        )
        completed.append(PublicationStage.SOURCE)
        if fault_at is PublicationStage.SOURCE:
            raise ArtifactIntegrityError("fault injected at source validation")
        current_stage = PublicationStage.STAGE
        with AtomicCheckpointDestination(source_integrity.path, destination_path) as publication:
            shutil.copyfile(source_integrity.path, publication.staging_path)
            completed.append(PublicationStage.STAGE)
            if fault_at is PublicationStage.STAGE:
                raise ArtifactIntegrityError("fault injected at staging")

            staged_digest, staged_size = _sha256(publication.staging_path, max_bytes=max_bytes)
            if (
                staged_digest != source_integrity.digest
                or staged_size != source_integrity.size_bytes
            ):
                raise ArtifactIntegrityError("staged artifact checksum or size changed")
            completed.append(PublicationStage.CHECKSUM)
            current_stage = PublicationStage.CHECKSUM
            if fault_at is PublicationStage.CHECKSUM:
                raise ArtifactIntegrityError("fault injected at checksum")

            if structure_check is not None:
                structure_check(publication.staging_path)
            completed.append(PublicationStage.STRUCTURE)
            current_stage = PublicationStage.STRUCTURE
            if fault_at is PublicationStage.STRUCTURE:
                raise ArtifactIntegrityError("fault injected at structure")

            if reload_check is not None:
                reload_check(publication.staging_path)
            completed.append(PublicationStage.RELOAD)
            current_stage = PublicationStage.RELOAD
            if fault_at is PublicationStage.RELOAD:
                raise ArtifactIntegrityError("fault injected at reload")

            if generation_check is not None:
                generation_check(publication.staging_path)
            completed.append(PublicationStage.GENERATION)
            current_stage = PublicationStage.GENERATION
            if fault_at is PublicationStage.GENERATION:
                raise ArtifactIntegrityError("fault injected at generation")

            current_stage = PublicationStage.PROMOTE
            if fault_at is PublicationStage.PROMOTE:
                raise ArtifactIntegrityError("fault injected before promotion")
            published = publication.publish()
            completed.append(PublicationStage.PROMOTE)
            published_digest, _ = _sha256(published, max_bytes=max_bytes)
            return ArtifactValidationResult(
                IntegrityOutcome.ACCEPTED,
                format,
                source_integrity,
                str(published),
                tuple(completed),
                None,
                published_digest,
                None,
            )
    except (ArtifactIntegrityError, CheckpointDestinationError, OSError) as error:
        outcome = (
            IntegrityOutcome.INTERRUPTED
            if fault_at is not None and current_stage is fault_at
            else IntegrityOutcome.REJECTED
        )
        return ArtifactValidationResult(
            outcome,
            format,
            source_integrity,
            str(destination_path),
            tuple(completed),
            current_stage,
            None,
            str(error),
        )

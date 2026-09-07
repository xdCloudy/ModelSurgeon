from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.surgery import (
    ArtifactIntegrityError,
    IntegrityFormat,
    IntegrityOutcome,
    PublicationStage,
    publish_validated_artifact,
    recover_staging,
    validate_artifact_source,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.safetensors"
    source.write_bytes(b"tiny physical artifact")
    return source, tmp_path / "child.safetensors"


def test_all_fault_boundaries_leave_source_and_destination_safe(tmp_path: Path) -> None:
    for stage in (
        PublicationStage.SOURCE,
        PublicationStage.STAGE,
        PublicationStage.CHECKSUM,
        PublicationStage.STRUCTURE,
        PublicationStage.RELOAD,
        PublicationStage.GENERATION,
        PublicationStage.PROMOTE,
    ):
        source, destination = _fixture(tmp_path / stage.value)
        destination.parent.mkdir(exist_ok=True)
        result = publish_validated_artifact(
            source,
            destination,
            format=IntegrityFormat.HUGGINGFACE,
            fault_at=stage,
            structure_check=lambda path: path.read_bytes(),
            reload_check=lambda path: path.stat(),
            generation_check=lambda path: path.read_bytes(),
        )
        assert result.outcome is IntegrityOutcome.INTERRUPTED
        assert result.failed_stage is stage
        assert source.read_bytes() == b"tiny physical artifact"
        assert not destination.exists()


def test_success_is_reloadable_and_idempotent_across_restart(tmp_path: Path) -> None:
    source, destination = _fixture(tmp_path)
    digest = validate_artifact_source(source, format=IntegrityFormat.GGUF).digest
    result = publish_validated_artifact(
        source,
        destination,
        format=IntegrityFormat.GGUF,
        expected_digest=digest,
        structure_check=lambda path: None,
        reload_check=lambda path: None,
        generation_check=lambda path: None,
    )
    assert result.accepted
    assert destination.read_bytes() == source.read_bytes()

    repeated = publish_validated_artifact(source, destination, format=IntegrityFormat.GGUF)
    assert repeated.outcome is IntegrityOutcome.REJECTED
    assert destination.read_bytes() == source.read_bytes()


def test_corrupt_checksum_and_scoped_recovery_fail_closed(tmp_path: Path) -> None:
    source, destination = _fixture(tmp_path)
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        validate_artifact_source(
            source,
            format=IntegrityFormat.HUGGINGFACE,
            expected_digest="0" * 64,
        )

    neighboring = tmp_path / ".other-artifact.keep.staging"
    neighboring.write_bytes(b"keep")
    removed = recover_staging(tmp_path, destination_name=destination.name)
    assert removed == ()
    assert neighboring.exists()

"""Tests for non-overwriting atomic checkpoint destinations."""

from __future__ import annotations

import os

import pytest

from modelsurgeon.surgery.checkpoint_destination import (
    AtomicCheckpointDestination,
    CheckpointDestinationError,
)


def test_file_checkpoint_publishes_from_sibling_staging_without_touching_source(tmp_path) -> None:
    source = tmp_path / "source.gguf"
    destination = tmp_path / "candidate.gguf"
    source.write_bytes(b"source")

    with AtomicCheckpointDestination(source, destination) as target:
        assert target.staging_path.parent == destination.parent
        target.staging_path.write_bytes(b"candidate")
        published = target.publish()
        assert published == destination.resolve()
        assert target.published

    assert source.read_bytes() == b"source"
    assert destination.read_bytes() == b"candidate"
    assert not target.staging_path.exists()


def test_interrupted_staging_is_cleaned_and_never_replaces_source(tmp_path) -> None:
    source = tmp_path / "source.gguf"
    destination = tmp_path / "candidate.gguf"
    source.write_bytes(b"stable-source")
    staging = None

    with pytest.raises(RuntimeError, match="interrupt"):
        with AtomicCheckpointDestination(source, destination) as target:
            staging = target.staging_path
            target.staging_path.write_bytes(b"partial")
            raise RuntimeError("interrupt")

    assert source.read_bytes() == b"stable-source"
    assert not destination.exists()
    assert staging is not None and not staging.exists()


def test_existing_destination_is_never_overwritten(tmp_path) -> None:
    source = tmp_path / "source.gguf"
    destination = tmp_path / "candidate.gguf"
    source.write_bytes(b"source")
    destination.write_bytes(b"existing")

    with pytest.raises(CheckpointDestinationError, match="already exists"):
        AtomicCheckpointDestination(source, destination)
    assert destination.read_bytes() == b"existing"


def test_same_path_and_directory_descendants_are_rejected(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"x")

    with pytest.raises(CheckpointDestinationError, match="same checkpoint"):
        AtomicCheckpointDestination(source, source)
    with pytest.raises(CheckpointDestinationError, match="inside"):
        AtomicCheckpointDestination(source, source / "candidate")


def test_symlink_alias_is_rejected_when_supported(tmp_path) -> None:
    source = tmp_path / "source.gguf"
    alias = tmp_path / "alias.gguf"
    source.write_bytes(b"source")
    try:
        os.symlink(source, alias)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this host")

    with pytest.raises(CheckpointDestinationError, match="same checkpoint"):
        AtomicCheckpointDestination(source, alias)


def test_directory_checkpoint_stages_then_renames_atomically(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "candidate"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"source")

    with AtomicCheckpointDestination(source, destination) as target:
        target.staging_path.mkdir()
        (target.staging_path / "weights.bin").write_bytes(b"candidate")
        target.publish()

    assert (source / "weights.bin").read_bytes() == b"source"
    assert (destination / "weights.bin").read_bytes() == b"candidate"


def test_publish_requires_active_context_and_staging_output(tmp_path) -> None:
    source = tmp_path / "source.gguf"
    source.write_bytes(b"source")
    target = AtomicCheckpointDestination(source, tmp_path / "candidate.gguf")

    with pytest.raises(CheckpointDestinationError, match="not active"):
        target.publish()
    with target, pytest.raises(CheckpointDestinationError, match="does not exist"):
        target.publish()

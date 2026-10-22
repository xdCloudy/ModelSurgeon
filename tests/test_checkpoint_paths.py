"""Path confinement tests for model checkpoint indexes and shards."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.adapters.checkpoint_paths import (
    CheckpointPathError,
    resolve_checkpoint_shard_path,
    validate_checkpoint_index_path,
)


@pytest.mark.parametrize("hub_snapshot", [False, True])
@pytest.mark.parametrize(
    "shard_name",
    [
        "../outside.safetensors",
        "/outside.safetensors",
        r"..\outside.safetensors",
        r"C:\outside.safetensors",
    ],
)
def test_shard_names_cannot_escape_checkpoint(
    tmp_path: Path,
    hub_snapshot: bool,
    shard_name: str,
) -> None:
    if hub_snapshot:
        checkpoint_root = (
            tmp_path / "models--org--model" / "snapshots" / ("a" * 40)
        )
        (checkpoint_root.parent.parent / "blobs").mkdir(parents=True)
    else:
        checkpoint_root = tmp_path / "model"
    checkpoint_root.mkdir(parents=True)

    with pytest.raises(CheckpointPathError):
        resolve_checkpoint_shard_path(
            checkpoint_root,
            shard_name,
            hub_snapshot=hub_snapshot,
        )


@pytest.mark.parametrize("hub_snapshot", [False, True])
def test_checkpoint_index_must_resolve_inside_its_boundary(
    tmp_path: Path,
    hub_snapshot: bool,
) -> None:
    if hub_snapshot:
        checkpoint_root = (
            tmp_path / "models--org--model" / "snapshots" / ("b" * 40)
        )
        (checkpoint_root.parent.parent / "blobs").mkdir(parents=True)
    else:
        checkpoint_root = tmp_path / "model"
    checkpoint_root.mkdir(parents=True)
    outside_index = tmp_path / "outside.index.json"

    with pytest.raises(CheckpointPathError, match="escapes"):
        validate_checkpoint_index_path(
            outside_index,
            checkpoint_root,
            hub_snapshot=hub_snapshot,
        )

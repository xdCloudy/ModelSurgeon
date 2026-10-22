"""Shared path and size limits for sharded model checkpoint indexes."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

MAX_CHECKPOINT_INDEX_BYTES = 100 * 1024 * 1024


class CheckpointPathError(ValueError):
    """Raised when an index or shard path escapes its checkpoint boundary."""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def hub_snapshot_blob_root(checkpoint_root: Path) -> Path | None:
    """Return the Hugging Face blob root for a standard snapshot directory."""
    snapshots = checkpoint_root.parent
    cache_root = snapshots.parent
    if snapshots.name != "snapshots" or not cache_root.name.startswith("models--"):
        return None
    try:
        resolved_cache = cache_root.resolve()
        resolved_snapshot = checkpoint_root.resolve()
        resolved_blobs = (resolved_cache / "blobs").resolve()
    except (OSError, RuntimeError):
        return None
    if not _inside(resolved_snapshot, resolved_cache) or not _inside(
        resolved_blobs, resolved_cache
    ):
        return None
    return resolved_blobs


def validate_checkpoint_index_path(
    index_path: Path,
    checkpoint_root: Path,
    *,
    hub_snapshot: bool,
) -> None:
    """Ensure an index is inside its local checkpoint or selected Hub snapshot."""
    root = checkpoint_root.absolute()
    try:
        resolved_index = index_path.resolve()
    except (OSError, RuntimeError) as exc:
        raise CheckpointPathError("checkpoint index path cannot be resolved") from exc

    if not hub_snapshot:
        if not _inside(resolved_index, root.resolve()):
            raise CheckpointPathError("checkpoint index escapes its model directory")
        return

    blob_root = hub_snapshot_blob_root(root)
    if blob_root is None:
        raise CheckpointPathError("Hub checkpoint root is not a standard snapshot directory")
    if not (_inside(resolved_index, root.resolve()) or _inside(resolved_index, blob_root)):
        raise CheckpointPathError("Hub checkpoint index escapes its snapshot or blob cache")


def resolve_checkpoint_shard_path(
    checkpoint_root: Path,
    shard_name: str,
    *,
    hub_snapshot: bool,
) -> Path:
    """Validate a cross-platform shard name and return its in-root path."""
    if not shard_name or "\x00" in shard_name or "\\" in shard_name or ":" in shard_name:
        raise CheckpointPathError("checkpoint index contains an invalid shard filename")

    posix_name = PurePosixPath(shard_name)
    windows_name = PureWindowsPath(shard_name)
    if (
        posix_name.is_absolute()
        or windows_name.drive
        or windows_name.root
        or ".." in posix_name.parts
        or ".." in windows_name.parts
    ):
        raise CheckpointPathError(
            f"checkpoint shard path escapes its model directory: {shard_name!r}"
        )

    root = checkpoint_root.absolute()
    shard_path = root.joinpath(*posix_name.parts)
    try:
        resolved_root = root.resolve()
        resolved_shard = shard_path.resolve()
    except (OSError, RuntimeError) as exc:
        raise CheckpointPathError(
            f"checkpoint shard path cannot be resolved: {shard_name!r}"
        ) from exc

    if not hub_snapshot:
        if not _inside(resolved_shard, resolved_root):
            raise CheckpointPathError(
                f"checkpoint shard path escapes its model directory: {shard_name!r}"
            )
        return shard_path

    blob_root = hub_snapshot_blob_root(root)
    if blob_root is None:
        raise CheckpointPathError("Hub checkpoint root is not a standard snapshot directory")
    parent = shard_path.parent
    while parent != root:
        if parent.is_symlink() or not _inside(parent, root):
            raise CheckpointPathError(
                f"checkpoint shard path crosses an unsafe directory: {shard_name!r}"
            )
        parent = parent.parent
    if not (_inside(resolved_shard, resolved_root) or _inside(resolved_shard, blob_root)):
        raise CheckpointPathError(
            f"checkpoint shard target escapes its snapshot or blob cache: {shard_name!r}"
        )
    return shard_path

"""Path validation helpers for repository-local audit manifests."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def is_repository_file(root: Path, value: str | Path) -> bool:
    """Return whether a manifest path resolves to a file inside ``root``."""

    raw = value.as_posix() if isinstance(value, Path) else value
    if not isinstance(raw, str) or not raw or "\\" in raw:
        return False
    posix_path = PurePosixPath(raw)
    windows_path = PureWindowsPath(raw)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
    ):
        return False
    try:
        root_path = root.resolve()
        resolved = (root_path / Path(*posix_path.parts)).resolve()
        return resolved.is_relative_to(root_path) and resolved.is_file()
    except (OSError, RuntimeError, ValueError):
        return False

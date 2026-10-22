"""Small filesystem primitives for publishing complete text files atomically."""

from __future__ import annotations

import os
import stat
import uuid
from pathlib import Path


def atomic_write_text(path: str | Path, contents: str, *, replace_existing: bool) -> None:
    """Stage, flush, and atomically publish text beside its destination.

    When ``replace_existing`` is false, a hard link publishes the staged file
    only if the destination does not already exist. Staging beside the target
    keeps publication on the same filesystem.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_mode: int | None = None
    if replace_existing:
        try:
            existing = destination.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISREG(existing.st_mode):
                existing_mode = stat.S_IMODE(existing.st_mode)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    temporary_created = False
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            temporary_created = True
            stream.write(contents)
            stream.flush()
            if existing_mode is not None:
                os.chmod(temporary, existing_mode)
            os.fsync(stream.fileno())
        if replace_existing:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary_created:
            temporary.unlink(missing_ok=True)

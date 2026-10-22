"""Small cross-platform exclusive file lock for persistent state updates."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive process lock for the lifetime of the context."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            msvcrt_api = cast(Any, msvcrt)
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt_api.locking(stream.fileno(), msvcrt_api.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt_api.locking(stream.fileno(), msvcrt_api.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl_api = cast(Any, fcntl)
            fcntl_api.flock(stream.fileno(), fcntl_api.LOCK_EX)
            try:
                yield
            finally:
                fcntl_api.flock(stream.fileno(), fcntl_api.LOCK_UN)

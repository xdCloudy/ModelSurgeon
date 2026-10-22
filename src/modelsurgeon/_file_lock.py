"""Small cross-platform exclusive file lock for persistent state updates."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock_for(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.expanduser().absolute().resolve(strict=False)))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold exclusive thread and process locks for the lifetime of the context."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock_for(path), path.open("a+b") as stream:
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

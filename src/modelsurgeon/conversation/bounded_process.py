"""Small subprocess runner with bounded pipe capture and process-tree cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from typing import IO

_PIPE_CHUNK_BYTES = 64 * 1024
_POLL_INTERVAL_SECONDS = 0.01


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    output_exceeded: bool
    cancelled: bool = False


def _drain_bounded(
    stream: IO[bytes], output: bytearray, byte_limit: int, exceeded: threading.Event
) -> None:
    while chunk := stream.read(_PIPE_CHUNK_BYTES):
        remaining = byte_limit + 1 - len(output)
        if remaining > 0:
            output.extend(chunk[:remaining])
        if len(output) > byte_limit:
            exceeded.set()
            return


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            with suppress(OSError):
                process.kill()
        return

    killpg = getattr(os, "killpg", None)
    terminate_signal = getattr(signal, "SIGTERM", None)
    kill_signal = getattr(signal, "SIGKILL", terminate_signal)
    if killpg is None or terminate_signal is None:
        with suppress(OSError):
            process.kill()
        return
    with suppress(OSError):
        killpg(process.pid, terminate_signal)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=0.25)
    if kill_signal is not None:
        with suppress(OSError):
            killpg(process.pid, kill_signal)
    with suppress(OSError):
        process.kill()


def run_bounded_process(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    cancelled: Callable[[], bool] | None = None,
) -> BoundedProcessResult:
    """Run a command while retaining only each stream's byte budget plus one byte."""

    if (
        isinstance(command, (str, bytes))
        or not command
        or not all(isinstance(argument, str) for argument in command)
        or not command[0]
    ):
        raise ValueError("subprocess command must be non-empty")
    if (
        isinstance(timeout_seconds, bool)
        or not isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or isinstance(max_stdout_bytes, bool)
        or not isinstance(max_stdout_bytes, int)
        or max_stdout_bytes <= 0
        or isinstance(max_stderr_bytes, bool)
        or not isinstance(max_stderr_bytes, int)
        or max_stderr_bytes <= 0
    ):
        raise ValueError("subprocess time and output limits must be positive")

    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    process = subprocess.Popen(
        tuple(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    stdout_stream = process.stdout
    stderr_stream = process.stderr
    assert stdout_stream is not None
    assert stderr_stream is not None

    stdout = bytearray()
    stderr = bytearray()
    output_exceeded = threading.Event()
    readers = (
        threading.Thread(
            target=_drain_bounded,
            args=(stdout_stream, stdout, max_stdout_bytes, output_exceeded),
            daemon=True,
        ),
        threading.Thread(
            target=_drain_bounded,
            args=(stderr_stream, stderr, max_stderr_bytes, output_exceeded),
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout_seconds
    timed_out = False
    terminated = False
    was_cancelled = False
    while process.poll() is None or any(reader.is_alive() for reader in readers):
        if cancelled is not None and cancelled():
            was_cancelled = True
            _terminate_process_tree(process)
            terminated = True
            break
        if output_exceeded.is_set():
            _terminate_process_tree(process)
            terminated = True
            break
        if time.monotonic() >= deadline:
            timed_out = True
            _terminate_process_tree(process)
            terminated = True
            break
        time.sleep(_POLL_INTERVAL_SECONDS)

    if process.poll() is None:
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0.5)
    if process.poll() is None:
        with suppress(OSError):
            process.kill()
        process.wait()
    for reader in readers:
        reader.join(timeout=0.5 if terminated else None)
    for stream in (stdout_stream, stderr_stream):
        with suppress(OSError, ValueError):
            stream.close()

    return BoundedProcessResult(
        process.returncode if process.returncode is not None else -1,
        bytes(stdout),
        bytes(stderr),
        timed_out,
        output_exceeded.is_set(),
        was_cancelled,
    )

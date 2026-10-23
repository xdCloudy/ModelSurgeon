"""Subprocess helpers that bound captured output in memory."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from os import PathLike
from typing import BinaryIO, Final

from modelsurgeon.utils.numbers import is_finite_number

_SAFE_SUBPROCESS_ENVIRONMENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
)


def sanitized_subprocess_environment(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return common runtime settings plus explicit overrides, excluding ambient secrets."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _SAFE_SUBPROCESS_ENVIRONMENT_KEYS
    }
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ValueError("subprocess environment overrides must be a mapping")
        for key, value in overrides.items():
            if (
                not isinstance(key, str)
                or not key
                or "=" in key
                or "\0" in key
                or not isinstance(value, str)
                or "\0" in value
            ):
                raise ValueError("subprocess environment overrides must contain valid strings")
            existing_key = next((name for name in environment if name.upper() == key.upper()), None)
            if existing_key is not None:
                del environment[existing_key]
            environment[key] = value
    return environment


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    """Captured process result with explicit timeout and output-limit outcomes."""

    returncode: int
    stdout: bytes
    stdout_total_bytes: int
    stderr: bytes
    stderr_total_bytes: int
    timed_out: bool
    output_exceeded: bool
    resource_exceeded: bool


def run_bounded_process(
    command: Sequence[str],
    input_payload: bytes,
    *,
    timeout_seconds: float,
    stdout_limit: int,
    stderr_limit: int,
    total_output_limit: int | None = None,
    poll_interval_seconds: float = 0.05,
    terminate_grace_seconds: float = 1.0,
    terminate_on_output_limit: bool = True,
    stderr_to_stdout: bool = False,
    capture_output: bool = True,
    output_callback: Callable[[str, bytes], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    creationflags: int = 0,
    cwd: str | bytes | PathLike[str] | PathLike[bytes] | None = None,
    env: Mapping[str, str] | None = None,
    resource_limit_check: Callable[[], bool] | None = None,
    process_sample_callback: Callable[[int], None] | None = None,
) -> BoundedProcessResult:
    """Capture bounded pipes and stop when output or sampled resource limits are crossed."""
    if (
        isinstance(command, (str, bytes, bytearray))
        or not isinstance(command, Sequence)
        or not command
        or any(not isinstance(argument, str) for argument in command)
        or not command[0]
        or not isinstance(input_payload, bytes)
        or not callable(monotonic)
        or not is_finite_number(timeout_seconds)
        or timeout_seconds <= 0
        or not is_finite_number(poll_interval_seconds)
        or poll_interval_seconds <= 0
        or not is_finite_number(terminate_grace_seconds)
        or terminate_grace_seconds <= 0
        or not isinstance(terminate_on_output_limit, bool)
        or not isinstance(stderr_to_stdout, bool)
        or not isinstance(capture_output, bool)
        or not isinstance(stdout_limit, int)
        or isinstance(stdout_limit, bool)
        or stdout_limit < 0
        or not isinstance(stderr_limit, int)
        or isinstance(stderr_limit, bool)
        or stderr_limit < 0
        or (
            total_output_limit is not None
            and (
                not isinstance(total_output_limit, int)
                or isinstance(total_output_limit, bool)
                or total_output_limit < 0
            )
        )
        or not isinstance(creationflags, int)
        or isinstance(creationflags, bool)
        or creationflags < 0
    ):
        raise ValueError("subprocess command, input, timeout, and output limits must be valid")
    command = tuple(command)
    started_at = monotonic()
    if not is_finite_number(started_at):
        raise ValueError("subprocess monotonic clock must return a finite number")
    deadline = started_at + timeout_seconds
    if not is_finite_number(deadline):
        raise ValueError("subprocess deadline must be finite")
    last_clock_reading = started_at
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if stderr_to_stdout else subprocess.PIPE,
        shell=False,
        creationflags=creationflags,
        cwd=cwd,
        env=None if env is None else dict(env),
        start_new_session=os.name != "nt",
    )
    if (
        process.stdin is None
        or process.stdout is None
        or (process.stderr is None and not stderr_to_stdout)
    ):
        process.kill()
        raise OSError("subprocess pipes could not be created")
    stdin, stdout, stderr = process.stdin, process.stdout, process.stderr

    output_exceeded = threading.Event()
    callback_failed = threading.Event()
    callback_errors: list[BaseException] = []
    captured: dict[str, tuple[bytes, int]] = {}
    output_lock = threading.Lock()
    total_output_bytes = 0

    def drain(name: str, pipe: BinaryIO, limit: int) -> None:
        nonlocal total_output_bytes
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = pipe.read(64 * 1024)
            except (OSError, ValueError):
                break
            if not chunk:
                break
            with output_lock:
                if output_callback is not None and not callback_errors:
                    try:
                        output_callback(name, chunk)
                    except BaseException as error:
                        callback_errors.append(error)
                        callback_failed.set()
                channel_remaining = limit - total
                total_remaining = (
                    len(chunk)
                    if total_output_limit is None
                    else total_output_limit - total_output_bytes
                )
                retained_bytes = (
                    min(len(chunk), channel_remaining, total_remaining) if capture_output else 0
                )
                if retained_bytes > 0:
                    chunks.append(chunk[:retained_bytes])
                total += len(chunk)
                total_output_bytes += len(chunk)
                if total > limit or (
                    total_output_limit is not None and total_output_bytes > total_output_limit
                ):
                    output_exceeded.set()
        captured[name] = b"".join(chunks), total

    def send_input() -> None:
        try:
            stdin.write(input_payload)
            stdin.flush()
        except (OSError, ValueError):
            pass
        finally:
            with suppress(OSError, ValueError):
                stdin.close()

    readers = [threading.Thread(target=drain, args=("stdout", stdout, stdout_limit), daemon=True)]
    if stderr is not None:
        readers.append(
            threading.Thread(target=drain, args=("stderr", stderr, stderr_limit), daemon=True)
        )
    writer = threading.Thread(target=send_input, daemon=True)

    timed_out = False
    resource_exceeded = False
    termination_requested = False

    def signal_process_tree(*, force: bool) -> None:
        if os.name == "nt":
            taskkill = shutil.which("taskkill.exe") or shutil.which("taskkill")
            if taskkill is not None:
                try:
                    command = [taskkill, "/PID", str(process.pid), "/T"]
                    if force:
                        command.append("/F")
                    result = subprocess.run(
                        command,
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5.0,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                    if result.returncode == 0:
                        return
                except (OSError, subprocess.TimeoutExpired):
                    pass
        else:
            killpg = getattr(os, "killpg", None)
            signal_number = getattr(signal, "SIGKILL" if force else "SIGTERM", None)
            if callable(killpg) and signal_number is not None:
                try:
                    killpg(process.pid, signal_number)
                    return
                except OSError:
                    pass
        with suppress(OSError):
            process.kill() if force else process.terminate()

    def terminate() -> None:
        nonlocal termination_requested
        termination_requested = True
        signal_process_tree(force=False)

    def reap() -> None:
        try:
            process.wait(timeout=terminate_grace_seconds)
        except subprocess.TimeoutExpired:
            signal_process_tree(force=True)
            process.wait()
        if termination_requested:
            signal_process_tree(force=True)
        if writer.ident is not None:
            writer.join(timeout=1.0)
        for reader in readers:
            if reader.ident is not None:
                reader.join(timeout=1.0)
        for pipe in (stdin, stdout, stderr):
            if pipe is not None and not pipe.closed:
                with suppress(OSError, ValueError):
                    pipe.close()

    try:
        for reader in readers:
            reader.start()
        writer.start()
    except BaseException:
        terminate()
        reap()
        raise

    try:
        while process.poll() is None:
            if process_sample_callback is not None:
                process_sample_callback(process.pid)
            if resource_limit_check is not None and resource_limit_check():
                resource_exceeded = True
                terminate()
                break
            if callback_failed.is_set():
                terminate()
                break
            if output_exceeded.is_set() and terminate_on_output_limit:
                terminate()
                break
            now = monotonic()
            if not is_finite_number(now) or now < last_clock_reading:
                raise RuntimeError("subprocess monotonic clock returned an invalid reading")
            last_clock_reading = now
            remaining = deadline - now
            if not is_finite_number(remaining):
                raise RuntimeError("subprocess remaining time is not finite")
            if remaining <= 0:
                timed_out = True
                terminate()
                break
            try:
                process.wait(timeout=min(poll_interval_seconds, remaining))
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        terminate()
        reap()
        raise
    try:
        if resource_limit_check is not None and resource_limit_check():
            resource_exceeded = True
    except BaseException:
        terminate()
        reap()
        raise
    reap()
    if resource_limit_check is not None and resource_limit_check():
        resource_exceeded = True
    if process.returncode is None:
        raise OSError("subprocess did not report an exit status")
    if callback_errors:
        raise RuntimeError("subprocess output callback failed") from callback_errors[0]
    stdout_bytes, stdout_total = captured.get("stdout", (b"", 0))
    stderr_bytes, stderr_total = captured.get("stderr", (b"", 0))
    return BoundedProcessResult(
        process.returncode,
        stdout_bytes,
        stdout_total,
        stderr_bytes,
        stderr_total,
        timed_out,
        output_exceeded.is_set(),
        resource_exceeded,
    )


__all__ = [
    "BoundedProcessResult",
    "run_bounded_process",
    "sanitized_subprocess_environment",
]

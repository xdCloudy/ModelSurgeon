"""Pinned llama.cpp load and bounded forward validation for generated GGUF files."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from modelsurgeon.adapters.gguf.conformance import GGML_UPSTREAM_REVISION

LLAMA_CPP_VALIDATION_COMMIT = GGML_UPSTREAM_REVISION
LLAMA_CPP_VALIDATION_SCHEMA_VERSION = 1
_VERSION_LOG_BYTES = 4096

_VERSION_REVISION = re.compile(
    r"version:\s*[^\r\n]*\(\s*`?([0-9a-fA-F]{7,40})`?\s*\)",
    re.IGNORECASE,
)


class LlamaCppValidationError(RuntimeError):
    """Raised when pinned llama.cpp validation cannot be performed safely."""


@dataclass(frozen=True, slots=True)
class LlamaCppValidationConfig:
    executable: str | Path = "llama-cli"
    expected_revision: str = LLAMA_CPP_VALIDATION_COMMIT
    prompt: str = "ModelSurgeon external GGUF validation."
    predict_tokens: int = 1
    context_size: int = 128
    threads: int = 1
    gpu_layers: int = 0
    seed: int = 0
    timeout_seconds: float = 60.0
    max_log_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        executable = str(self.executable)
        if not executable:
            raise LlamaCppValidationError("llama.cpp executable cannot be empty")
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", self.expected_revision) is None:
            raise LlamaCppValidationError(
                "expected llama.cpp revision must be a 7-40 character hexadecimal commit"
            )
        if not self.prompt.strip():
            raise LlamaCppValidationError("validation prompt cannot be empty")
        if self.predict_tokens <= 0:
            raise LlamaCppValidationError("validation must predict at least one token")
        if self.context_size <= 0 or self.threads <= 0:
            raise LlamaCppValidationError("context size and thread count must be positive")
        if self.gpu_layers < 0 or self.seed < 0:
            raise LlamaCppValidationError("GPU layers and seed must be non-negative")
        if self.timeout_seconds <= 0:
            raise LlamaCppValidationError("validation timeout must be positive")
        if self.max_log_bytes <= 0:
            raise LlamaCppValidationError("maximum captured log bytes must be positive")


@dataclass(frozen=True, slots=True)
class LlamaCppToolProvenance:
    executable: Path
    expected_revision: str
    reported_revision: str
    version_command: tuple[str, ...]
    version_output: str

    def to_record(self) -> dict[str, object]:
        return {
            "executable": str(self.executable),
            "expected_revision": self.expected_revision,
            "reported_revision": self.reported_revision,
            "version_command": list(self.version_command),
            "version_output": self.version_output,
        }


@dataclass(frozen=True, slots=True)
class LlamaCppGGUFValidationReport:
    model_path: Path
    tool: LlamaCppToolProvenance
    command: tuple[str, ...]
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    schema_version: int = LLAMA_CPP_VALIDATION_SCHEMA_VERSION

    @property
    def successful(self) -> bool:
        return not self.timed_out and self.returncode == 0

    @property
    def failure_reason(self) -> str | None:
        if self.successful:
            return None
        if self.timed_out:
            return "llama-cli validation timed out"
        return f"llama-cli exited with status {self.returncode}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "successful": self.successful,
            "failure_reason": self.failure_reason,
            "model_path": str(self.model_path),
            "tool": self.tool.to_record(),
            "command": list(self.command),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
        }


@dataclass(frozen=True, slots=True)
class _BoundedCommandResult:
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool


def _resolve_executable(value: str | Path, expected_revision: str) -> Path:
    raw = str(value)
    candidate = Path(raw).expanduser()
    has_separator = os.sep in raw or (os.altsep is not None and os.altsep in raw)
    if candidate.is_absolute() or has_separator:
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise LlamaCppValidationError(
                f"llama.cpp executable does not exist: {resolved}"
            )
        return resolved

    discovered = shutil.which(raw)
    if discovered is None:
        raise LlamaCppValidationError(
            f"{raw!r} was not found on PATH; install/build llama.cpp at commit "
            f"{expected_revision} or pass an explicit executable path"
        )
    return Path(discovered).resolve()


def _read_bounded_log(stream: BinaryIO, max_bytes: int) -> tuple[str, bool]:
    stream.flush()
    stream.seek(0, os.SEEK_END)
    total = stream.tell()
    stream.seek(0)
    data = stream.read(max_bytes)
    return data.decode("utf-8", errors="replace"), total > max_bytes


def _run_bounded(
    command: tuple[str, ...],
    *,
    timeout_seconds: float,
    max_log_bytes: int,
) -> _BoundedCommandResult:
    with tempfile.TemporaryFile() as stdout_stream, tempfile.TemporaryFile() as stderr_stream:
        try:
            completed = subprocess.run(
                command,
                check=False,
                stdout=stdout_stream,
                stderr=stderr_stream,
                timeout=timeout_seconds,
                shell=False,
            )
            returncode: int | None = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = None
            timed_out = True
        except OSError as error:
            raise LlamaCppValidationError(
                f"failed to execute llama.cpp command {command[0]!r}: {error}"
            ) from error

        stdout, stdout_truncated = _read_bounded_log(stdout_stream, max_log_bytes)
        stderr, stderr_truncated = _read_bounded_log(stderr_stream, max_log_bytes)
    return _BoundedCommandResult(
        returncode,
        timed_out,
        stdout,
        stderr,
        stdout_truncated,
        stderr_truncated,
    )


def _reported_revision(output: str) -> str:
    match = _VERSION_REVISION.search(output)
    if match is None:
        raise LlamaCppValidationError(
            "could not parse a git revision from `llama-cli --version` output"
        )
    return match.group(1).lower()


def _revision_matches(expected: str, reported: str) -> bool:
    expected_normalized = expected.lower()
    reported_normalized = reported.lower()
    return expected_normalized.startswith(reported_normalized) or reported_normalized.startswith(
        expected_normalized
    )


def _probe_tool(config: LlamaCppValidationConfig) -> LlamaCppToolProvenance:
    executable = _resolve_executable(config.executable, config.expected_revision)
    command = (str(executable), "--version")
    result = _run_bounded(
        command,
        timeout_seconds=min(config.timeout_seconds, 10.0),
        max_log_bytes=max(config.max_log_bytes, _VERSION_LOG_BYTES),
    )
    output = "\n".join(
        item for item in (result.stdout.strip(), result.stderr.strip()) if item
    )
    if result.timed_out:
        raise LlamaCppValidationError("`llama-cli --version` timed out")
    if result.returncode != 0:
        detail = output.splitlines()[0] if output else "no version output"
        raise LlamaCppValidationError(
            f"`llama-cli --version` failed with status {result.returncode}: {detail}"
        )
    revision = _reported_revision(output)
    if not _revision_matches(config.expected_revision, revision):
        raise LlamaCppValidationError(
            f"unsupported llama.cpp revision {revision}; ModelSurgeon validation is "
            f"pinned to {config.expected_revision}"
        )
    return LlamaCppToolProvenance(
        executable,
        config.expected_revision.lower(),
        revision,
        command,
        output,
    )


def _validation_command(
    executable: Path,
    model_path: Path,
    config: LlamaCppValidationConfig,
) -> tuple[str, ...]:
    return (
        str(executable),
        "-m",
        str(model_path),
        "-p",
        config.prompt,
        "-n",
        str(config.predict_tokens),
        "-c",
        str(config.context_size),
        "-t",
        str(config.threads),
        "-ngl",
        str(config.gpu_layers),
        "--seed",
        str(config.seed),
        "--temp",
        "0",
    )


def validate_generated_gguf(
    model_path: str | Path,
    *,
    config: LlamaCppValidationConfig | None = None,
) -> LlamaCppGGUFValidationReport:
    """Load a GGUF with pinned llama.cpp and run a bounded generation sanity check."""

    settings = config or LlamaCppValidationConfig()
    model = Path(model_path).expanduser().resolve()
    if not model.is_file():
        raise LlamaCppValidationError(f"GGUF model does not exist: {model}")
    if model.suffix.lower() != ".gguf":
        raise LlamaCppValidationError("llama.cpp validation requires a .gguf file")

    tool = _probe_tool(settings)
    command = _validation_command(tool.executable, model, settings)
    result = _run_bounded(
        command,
        timeout_seconds=settings.timeout_seconds,
        max_log_bytes=settings.max_log_bytes,
    )
    return LlamaCppGGUFValidationReport(
        model,
        tool,
        command,
        result.returncode,
        result.timed_out,
        result.stdout,
        result.stderr,
        result.stdout_truncated,
        result.stderr_truncated,
    )

"""Discovery and transactional execution of pinned external llama.cpp quantization."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from modelsurgeon.adapters.gguf.conformance import GGML_UPSTREAM_REVISION
from modelsurgeon.adapters.gguf.container import open_gguf
from modelsurgeon.evaluation.llama_cpp import (
    LlamaCppToolProvenance,
    LlamaCppValidationConfig,
    LlamaCppValidationError,
    _probe_tool,
    _resolve_executable,
    _run_bounded,
)

LLAMA_CPP_QUANTIZATION_SCHEMA_VERSION = 1
_MINIMUM_HELP_BYTES = 64 * 1024


class LlamaCppQuantizationToolError(RuntimeError):
    """Raised when external quantization cannot preserve its pinned tool contract."""


class LlamaCppQuantizationRecipe(StrEnum):
    Q4_K_M = "Q4_K_M"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"
    Q8_0 = "Q8_0"
    F16 = "F16"
    BF16 = "BF16"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LlamaCppQuantizationConfig:
    executable: str | Path = "llama-quantize"
    expected_revision: str = GGML_UPSTREAM_REVISION
    recipe: LlamaCppQuantizationRecipe = LlamaCppQuantizationRecipe.Q4_K_M
    threads: int = 1
    allow_requantize: bool = False
    timeout_seconds: float = 600.0
    max_log_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        if not str(self.executable):
            raise LlamaCppQuantizationToolError("llama-quantize executable cannot be empty")
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", self.expected_revision) is None:
            raise LlamaCppQuantizationToolError(
                "expected llama.cpp revision must be a 7-40 character hexadecimal commit"
            )
        if self.threads <= 0:
            raise LlamaCppQuantizationToolError("quantization thread count must be positive")
        if self.timeout_seconds <= 0 or self.max_log_bytes <= 0:
            raise LlamaCppQuantizationToolError(
                "quantization timeout and maximum log bytes must be positive"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "expected_revision": self.expected_revision.lower(),
            "recipe": self.recipe.value,
            "threads": self.threads,
            "allow_requantize": self.allow_requantize,
            "timeout_seconds": self.timeout_seconds,
            "max_log_bytes": self.max_log_bytes,
        }


@dataclass(frozen=True, slots=True)
class LlamaCppQuantizationToolProvenance:
    quantizer_executable: Path
    quantizer_sha256: str
    version_tool: LlamaCppToolProvenance
    help_command: tuple[str, ...]
    help_returncode: int
    help_output: str
    help_output_truncated: bool

    def to_record(self) -> dict[str, object]:
        return {
            "quantizer_executable": str(self.quantizer_executable),
            "quantizer_sha256": self.quantizer_sha256,
            "version_tool": self.version_tool.to_record(),
            "help_command": list(self.help_command),
            "help_returncode": self.help_returncode,
            "help_output": self.help_output,
            "help_output_truncated": self.help_output_truncated,
        }


@dataclass(frozen=True, slots=True)
class LlamaCppQuantizationReport:
    source_path: Path
    source_sha256: str
    destination_path: Path
    config: LlamaCppQuantizationConfig
    tool: LlamaCppQuantizationToolProvenance
    command: tuple[str, ...]
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool
    output_size_bytes: int | None
    output_sha256: str | None
    output_validation_error: str | None
    schema_version: int = LLAMA_CPP_QUANTIZATION_SCHEMA_VERSION

    @property
    def successful(self) -> bool:
        return (
            not self.timed_out
            and self.returncode == 0
            and self.output_size_bytes is not None
            and self.output_sha256 is not None
            and self.output_validation_error is None
        )

    @property
    def failure_reason(self) -> str | None:
        if self.successful:
            return None
        if self.timed_out:
            return "llama-quantize timed out"
        if self.returncode != 0:
            return f"llama-quantize exited with status {self.returncode}"
        return self.output_validation_error or "llama-quantize did not produce an output"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "successful": self.successful,
            "failure_reason": self.failure_reason,
            "source_path": str(self.source_path),
            "source_sha256": self.source_sha256,
            "destination_path": str(self.destination_path),
            "config": self.config.to_record(),
            "tool": self.tool.to_record(),
            "command": list(self.command),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "output_size_bytes": self.output_size_bytes,
            "output_sha256": self.output_sha256,
            "output_validation_error": self.output_validation_error,
        }


def _sibling_cli(quantizer: Path) -> Path:
    name = f"llama-cli{quantizer.suffix}" if quantizer.suffix else "llama-cli"
    candidate = quantizer.with_name(name)
    if not candidate.is_file():
        raise LlamaCppQuantizationToolError(
            f"llama-quantize was found at {quantizer}, but the required sibling "
            f"version probe {candidate.name!r} is missing; build/install both tools from "
            f"llama.cpp commit {GGML_UPSTREAM_REVISION}"
        )
    return candidate.resolve()


def discover_llama_cpp_quantizer(
    config: LlamaCppQuantizationConfig | None = None,
) -> LlamaCppQuantizationToolProvenance:
    """Resolve a quantizer, validate its sibling build revision, and verify its recipe."""

    settings = config or LlamaCppQuantizationConfig()
    try:
        quantizer = _resolve_executable(settings.executable, settings.expected_revision)
    except LlamaCppValidationError as error:
        raise LlamaCppQuantizationToolError(
            f"llama.cpp quantizer is unavailable: {error}"
        ) from error
    version_executable = _sibling_cli(quantizer)
    try:
        version_tool = _probe_tool(
            LlamaCppValidationConfig(
                executable=version_executable,
                expected_revision=settings.expected_revision,
                timeout_seconds=min(settings.timeout_seconds, 10.0),
                max_log_bytes=settings.max_log_bytes,
            )
        )
    except LlamaCppValidationError as error:
        raise LlamaCppQuantizationToolError(
            f"llama.cpp quantization tools failed revision validation: {error}"
        ) from error

    help_command = (str(quantizer), "--help")
    help_result = _run_bounded(
        help_command,
        timeout_seconds=min(settings.timeout_seconds, 10.0),
        max_log_bytes=max(settings.max_log_bytes, _MINIMUM_HELP_BYTES),
    )
    if help_result.timed_out:
        raise LlamaCppQuantizationToolError("`llama-quantize --help` timed out")
    help_output = "\n".join(
        item for item in (help_result.stdout.strip(), help_result.stderr.strip()) if item
    )
    if help_result.returncode not in {0, 1} or "allowed quantization types" not in help_output:
        raise LlamaCppQuantizationToolError(
            "resolved llama-quantize does not expose the expected quantization interface"
        )
    if re.search(rf"\b{re.escape(settings.recipe.value)}\b", help_output) is None:
        raise LlamaCppQuantizationToolError(
            f"resolved llama-quantize does not support recipe {settings.recipe.value}"
        )
    return LlamaCppQuantizationToolProvenance(
        quantizer,
        _sha256_file(quantizer),
        version_tool,
        help_command,
        help_result.returncode or 0,
        help_output,
        help_result.stdout_truncated or help_result.stderr_truncated,
    )


def _quantization_command(
    tool: LlamaCppQuantizationToolProvenance,
    source: Path,
    temporary: Path,
    config: LlamaCppQuantizationConfig,
) -> tuple[str, ...]:
    prefix: tuple[str, ...] = (str(tool.quantizer_executable),)
    if config.allow_requantize:
        prefix = (*prefix, "--allow-requantize")
    return (
        *prefix,
        str(source),
        str(temporary),
        config.recipe.value,
        str(config.threads),
    )


def _validated_output(path: Path) -> tuple[int, str]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise LlamaCppQuantizationToolError("llama-quantize did not create a non-empty GGUF")
    with open_gguf(path):
        pass
    return path.stat().st_size, _sha256_file(path)


def _publish_without_overwrite(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise LlamaCppQuantizationToolError(
            f"quantization destination appeared during execution: {destination}"
        ) from error
    except OSError as error:
        raise LlamaCppQuantizationToolError(
            f"could not publish quantized GGUF without overwriting {destination}: {error}"
        ) from error
    temporary.unlink()


def quantize_gguf_with_llama_cpp(
    source_path: str | Path,
    destination_path: str | Path,
    *,
    config: LlamaCppQuantizationConfig | None = None,
) -> LlamaCppQuantizationReport:
    """Run a pinned external quantizer and publish a validated output without overwrite."""

    settings = config or LlamaCppQuantizationConfig()
    source = Path(source_path).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    if not source.is_file():
        raise LlamaCppQuantizationToolError(f"quantization source does not exist: {source}")
    if source.suffix.lower() != ".gguf" or destination.suffix.lower() != ".gguf":
        raise LlamaCppQuantizationToolError("quantization source and destination must be .gguf")
    if source == destination:
        raise LlamaCppQuantizationToolError("quantization source and destination must differ")
    if destination.exists():
        raise LlamaCppQuantizationToolError(
            f"quantization destination already exists and will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    tool = discover_llama_cpp_quantizer(settings)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    command = _quantization_command(tool, source, temporary, settings)
    result = _run_bounded(
        command,
        timeout_seconds=settings.timeout_seconds,
        max_log_bytes=settings.max_log_bytes,
    )
    output_size: int | None = None
    output_sha256: str | None = None
    validation_error: str | None = None
    if not result.timed_out and result.returncode == 0:
        try:
            output_size, output_sha256 = _validated_output(temporary)
            _publish_without_overwrite(temporary, destination)
        except (LlamaCppQuantizationToolError, OSError, ValueError) as error:
            validation_error = str(error)
    if temporary.exists():
        temporary.unlink()
    return LlamaCppQuantizationReport(
        source,
        _sha256_file(source),
        destination,
        settings,
        tool,
        command,
        result.returncode,
        result.timed_out,
        result.stdout,
        result.stderr,
        result.stdout_truncated,
        result.stderr_truncated,
        output_size,
        output_sha256,
        validation_error,
    )

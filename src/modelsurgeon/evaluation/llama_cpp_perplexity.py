"""Paired GGUF perplexity benchmarking with pinned llama.cpp tooling."""

from __future__ import annotations

import hashlib
import json
import math
import re
from codecs import getincrementaldecoder
from dataclasses import dataclass
from pathlib import Path

from modelsurgeon.adapters.gguf.container import open_gguf
from modelsurgeon.evaluation.llama_cpp import (
    LLAMA_CPP_VALIDATION_COMMIT,
    LlamaCppToolProvenance,
    LlamaCppValidationConfig,
    _probe_tool,
    _run_bounded,
)

LLAMA_CPP_PERPLEXITY_SCHEMA_VERSION = 1
_FINAL_PERPLEXITY = re.compile(
    r"Final\s+estimate:\s*PPL\s*=\s*"
    r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)"
    r"(?:\s*\+/-\s*([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?))?",
    re.IGNORECASE,
)


class LlamaCppPerplexityError(RuntimeError):
    """Raised when a paired llama.cpp perplexity comparison is unsafe to run."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LlamaCppPerplexityManifest:
    """Content-addressed text samples and their source/tokenizer provenance."""

    text_path: Path
    text_sha256: str
    sample_count: int
    dataset: str
    dataset_revision: str
    split: str
    tokenizer: str
    tokenizer_revision: str

    def __post_init__(self) -> None:
        if not all(
            (
                self.dataset,
                self.dataset_revision,
                self.split,
                self.tokenizer,
                self.tokenizer_revision,
            )
        ):
            raise LlamaCppPerplexityError(
                "dataset and tokenizer identities and revisions are required"
            )
        if self.sample_count <= 0:
            raise LlamaCppPerplexityError("perplexity manifest sample count must be positive")
        if re.fullmatch(r"[0-9a-f]{64}", self.text_sha256) is None:
            raise LlamaCppPerplexityError(
                "perplexity manifest text digest must be a lowercase SHA-256"
            )

    @property
    def manifest_id(self) -> str:
        identity = self.to_record().copy()
        del identity["text_path"]
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "text_path": str(self.text_path),
            "text_sha256": self.text_sha256,
            "sample_count": self.sample_count,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "tokenizer": self.tokenizer,
            "tokenizer_revision": self.tokenizer_revision,
        }


@dataclass(frozen=True, slots=True)
class LlamaCppPerplexityConfig:
    executable: str | Path = "llama-perplexity"
    expected_revision: str = LLAMA_CPP_VALIDATION_COMMIT
    context_size: int = 512
    batch_size: int = 512
    microbatch_size: int = 512
    threads: int = 1
    gpu_layers: int = 0
    chunks: int = 1
    warmup: bool = False
    timeout_seconds: float = 300.0
    max_log_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        if not str(self.executable):
            raise LlamaCppPerplexityError("llama-perplexity executable cannot be empty")
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", self.expected_revision) is None:
            raise LlamaCppPerplexityError(
                "expected llama.cpp revision must be a 7-40 character hexadecimal commit"
            )
        if any(
            value <= 0
            for value in (
                self.context_size,
                self.batch_size,
                self.microbatch_size,
                self.threads,
                self.chunks,
            )
        ):
            raise LlamaCppPerplexityError(
                "context, batch, microbatch, thread, and chunk counts must be positive"
            )
        if self.batch_size > self.context_size or self.microbatch_size > self.batch_size:
            raise LlamaCppPerplexityError(
                "perplexity requires microbatch <= batch <= context size"
            )
        if self.gpu_layers < 0:
            raise LlamaCppPerplexityError("GPU layer count must be non-negative")
        if self.timeout_seconds <= 0 or self.max_log_bytes <= 0:
            raise LlamaCppPerplexityError("timeout and maximum log bytes must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "expected_revision": self.expected_revision.lower(),
            "context_size": self.context_size,
            "batch_size": self.batch_size,
            "microbatch_size": self.microbatch_size,
            "threads": self.threads,
            "gpu_layers": self.gpu_layers,
            "chunks": self.chunks,
            "warmup": self.warmup,
            "timeout_seconds": self.timeout_seconds,
            "max_log_bytes": self.max_log_bytes,
        }


@dataclass(frozen=True, slots=True)
class LlamaCppPerplexityMeasurement:
    model_path: Path
    model_sha256: str
    tokenizer_sha256: str
    command: tuple[str, ...]
    returncode: int | None
    timed_out: bool
    perplexity: float | None
    uncertainty: float | None
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool

    @property
    def successful(self) -> bool:
        return (
            not self.timed_out
            and self.returncode == 0
            and self.perplexity is not None
        )

    @property
    def failure_reason(self) -> str | None:
        if self.successful:
            return None
        if self.timed_out:
            return "llama-perplexity timed out"
        if self.returncode != 0:
            return f"llama-perplexity exited with status {self.returncode}"
        return "llama-perplexity output did not contain a finite final estimate"

    def to_record(self) -> dict[str, object]:
        return {
            "successful": self.successful,
            "failure_reason": self.failure_reason,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "command": list(self.command),
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "perplexity": self.perplexity,
            "uncertainty": self.uncertainty,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
        }


@dataclass(frozen=True, slots=True)
class LlamaCppPerplexityReport:
    manifest: LlamaCppPerplexityManifest
    config: LlamaCppPerplexityConfig
    tool: LlamaCppToolProvenance
    baseline: LlamaCppPerplexityMeasurement
    candidate: LlamaCppPerplexityMeasurement
    schema_version: int = LLAMA_CPP_PERPLEXITY_SCHEMA_VERSION

    @property
    def successful(self) -> bool:
        return self.baseline.successful and self.candidate.successful

    @property
    def perplexity_delta(self) -> float | None:
        if not self.successful:
            return None
        assert self.baseline.perplexity is not None
        assert self.candidate.perplexity is not None
        return self.candidate.perplexity - self.baseline.perplexity

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "successful": self.successful,
            "manifest_id": self.manifest.manifest_id,
            "manifest": self.manifest.to_record(),
            "config": self.config.to_record(),
            "tool": self.tool.to_record(),
            "baseline": self.baseline.to_record(),
            "candidate": self.candidate.to_record(),
            "perplexity_delta": self.perplexity_delta,
        }


def _verified_manifest(manifest: LlamaCppPerplexityManifest) -> Path:
    path = manifest.text_path.expanduser().resolve()
    if not path.is_file():
        raise LlamaCppPerplexityError(f"perplexity text file does not exist: {path}")
    if _sha256_file(path) != manifest.text_sha256:
        raise LlamaCppPerplexityError("perplexity text file does not match its manifest digest")
    decoder = getincrementaldecoder("utf-8")()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
    except UnicodeDecodeError as error:
        raise LlamaCppPerplexityError("perplexity text file must be valid UTF-8") from error
    return path


def _tokenizer_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open_gguf(path) as source:
        entries = (
            entry
            for entry in source.container.metadata
            if entry.key.startswith("tokenizer.")
        )
        found = False
        for entry in entries:
            found = True
            record = (
                entry.key,
                int(entry.value_type),
                int(entry.element_type) if entry.element_type is not None else None,
                entry.value,
            )
            digest.update(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode()
            )
            digest.update(b"\n")
    if not found:
        raise LlamaCppPerplexityError(f"GGUF has no tokenizer metadata: {path}")
    return digest.hexdigest()


def _verified_model(path: str | Path) -> tuple[Path, str, str]:
    model = Path(path).expanduser().resolve()
    if not model.is_file():
        raise LlamaCppPerplexityError(f"GGUF model does not exist: {model}")
    if model.suffix.lower() != ".gguf":
        raise LlamaCppPerplexityError("llama.cpp perplexity requires .gguf models")
    return model, _sha256_file(model), _tokenizer_digest(model)


def _command(
    executable: Path,
    model: Path,
    text_path: Path,
    config: LlamaCppPerplexityConfig,
) -> tuple[str, ...]:
    return (
        str(executable),
        "-m",
        str(model),
        "-f",
        str(text_path),
        "-c",
        str(config.context_size),
        "-b",
        str(config.batch_size),
        "-ub",
        str(config.microbatch_size),
        "-t",
        str(config.threads),
        "-ngl",
        str(config.gpu_layers),
        "--chunks",
        str(config.chunks),
        "--warmup" if config.warmup else "--no-warmup",
        "--ppl-output-type",
        "1",
        "--log-colors",
        "off",
    )


def _estimate(stdout: str, stderr: str) -> tuple[float | None, float | None]:
    matches = tuple(_FINAL_PERPLEXITY.finditer(f"{stdout}\n{stderr}"))
    if not matches:
        return None, None
    match = matches[-1]
    perplexity = float(match.group(1))
    uncertainty = float(match.group(2)) if match.group(2) is not None else None
    if not math.isfinite(perplexity) or perplexity < 1:
        return None, None
    if uncertainty is not None and (not math.isfinite(uncertainty) or uncertainty < 0):
        return None, None
    return perplexity, uncertainty


def _measure(
    model: Path,
    model_sha256: str,
    tokenizer_sha256: str,
    tool: LlamaCppToolProvenance,
    text_path: Path,
    config: LlamaCppPerplexityConfig,
) -> LlamaCppPerplexityMeasurement:
    command = _command(tool.executable, model, text_path, config)
    result = _run_bounded(
        command,
        timeout_seconds=config.timeout_seconds,
        max_log_bytes=config.max_log_bytes,
    )
    perplexity, uncertainty = _estimate(result.stdout, result.stderr)
    return LlamaCppPerplexityMeasurement(
        model,
        model_sha256,
        tokenizer_sha256,
        command,
        result.returncode,
        result.timed_out,
        perplexity,
        uncertainty,
        result.stdout,
        result.stderr,
        result.stdout_truncated,
        result.stderr_truncated,
    )


def benchmark_gguf_perplexity(
    baseline_model: str | Path,
    candidate_model: str | Path,
    manifest: LlamaCppPerplexityManifest,
    *,
    config: LlamaCppPerplexityConfig | None = None,
) -> LlamaCppPerplexityReport:
    """Benchmark a baseline/candidate pair under one tokenizer and context contract."""

    settings = config or LlamaCppPerplexityConfig()
    text_path = _verified_manifest(manifest)
    baseline, baseline_digest, baseline_tokenizer = _verified_model(baseline_model)
    candidate, candidate_digest, candidate_tokenizer = _verified_model(candidate_model)
    if baseline_tokenizer != candidate_tokenizer:
        raise LlamaCppPerplexityError(
            "baseline and candidate GGUF tokenizer metadata do not match"
        )
    tool = _probe_tool(
        LlamaCppValidationConfig(
            executable=settings.executable,
            expected_revision=settings.expected_revision,
            timeout_seconds=settings.timeout_seconds,
            max_log_bytes=settings.max_log_bytes,
        )
    )
    baseline_result = _measure(
        baseline,
        baseline_digest,
        baseline_tokenizer,
        tool,
        text_path,
        settings,
    )
    candidate_result = _measure(
        candidate,
        candidate_digest,
        candidate_tokenizer,
        tool,
        text_path,
        settings,
    )
    return LlamaCppPerplexityReport(
        manifest,
        settings,
        tool,
        baseline_result,
        candidate_result,
    )

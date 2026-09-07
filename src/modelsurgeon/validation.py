"""Deterministic, bounded hostile-input corpus and parser harnesses.

The harness deliberately accepts parser entry points as callables and runs
subprocess targets without a shell.  It records rejected, unsupported,
failed, timed-out, and resource-limited inputs instead of treating them as
missing evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

HOSTILE_FUZZ_SCHEMA_VERSION = "1.0"
HOSTILE_FUZZ_PROTOCOL_REVISION = "hostile-fuzz-v1"


class HostileFuzzError(ValueError):
    """Raised when a corpus or execution request is not safe to admit."""


class HostileInputKind(StrEnum):
    HF_CHECKPOINT = "hf_checkpoint"
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    CONFIG = "config"
    MANIFEST = "manifest"
    PATH = "path"
    SUBPROCESS = "subprocess"


class HostileOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    PATH_ESCAPE = "path_escape"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class HostileFuzzLimits:
    """Hard admission and subprocess limits for one corpus execution."""

    max_input_bytes: int = 64 * 1024
    max_output_bytes: int = 64 * 1024
    timeout_seconds: float = 2.0
    max_memory_bytes: int = 256 * 1024 * 1024
    max_disk_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        if self.max_input_bytes <= 0 or self.max_output_bytes <= 0:
            raise HostileFuzzError("input and output limits must be positive")
        if self.timeout_seconds <= 0:
            raise HostileFuzzError("timeout must be positive")
        if self.max_memory_bytes <= 0 or self.max_disk_bytes <= 0:
            raise HostileFuzzError("memory and disk limits must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "max_input_bytes": self.max_input_bytes,
            "max_output_bytes": self.max_output_bytes,
            "timeout_seconds": self.timeout_seconds,
            "max_memory_bytes": self.max_memory_bytes,
            "max_disk_bytes": self.max_disk_bytes,
        }


@dataclass(frozen=True, slots=True)
class HostileFuzzCase:
    """One tiny, license-safe, content-addressed hostile input."""

    case_id: str
    kind: HostileInputKind
    payload: bytes
    seed: int
    expected: HostileOutcome
    limits: HostileFuzzLimits = HostileFuzzLimits()
    minimized: bool = True
    regression: bool = False
    source_boundary: str = "modelsurgeon"

    def __post_init__(self) -> None:
        if not self.case_id or len(self.case_id) > 160:
            raise HostileFuzzError("case_id must be a short non-empty identifier")
        if self.seed < 0:
            raise HostileFuzzError("seed must be non-negative")
        if len(self.payload) > self.limits.max_input_bytes:
            raise HostileFuzzError("payload exceeds the input limit")
        expected_id = _case_id(self.kind, self.payload, self.seed, self.expected)
        if self.case_id != expected_id:
            raise HostileFuzzError("case_id is not content-addressed")

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "kind": self.kind.value,
            "payload_sha256": self.digest,
            "payload_bytes": len(self.payload),
            "seed": self.seed,
            "expected": self.expected.value,
            "limits": self.limits.to_record(),
            "minimized": self.minimized,
            "regression": self.regression,
            "source_boundary": self.source_boundary,
        }


@dataclass(frozen=True, slots=True)
class HostileFuzzCorpus:
    cases: tuple[HostileFuzzCase, ...]
    seed: int
    tool_revision: str = HOSTILE_FUZZ_PROTOCOL_REVISION
    cpu_hour_budget: float = 0.25
    max_total_bytes: int = 512 * 1024

    def __post_init__(self) -> None:
        if self.seed < 0 or not self.tool_revision:
            raise HostileFuzzError("corpus seed and tool revision are required")
        if not 0 < self.cpu_hour_budget <= 24:
            raise HostileFuzzError("cpu-hour budget must be in (0, 24]")
        if self.max_total_bytes <= 0:
            raise HostileFuzzError("corpus byte budget must be positive")
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise HostileFuzzError("corpus case identifiers must be unique")
        if sum(len(case.payload) for case in self.cases) > self.max_total_bytes:
            raise HostileFuzzError("corpus exceeds its byte budget")

    @property
    def corpus_id(self) -> str:
        encoded = json.dumps(self.to_record(include_id=False), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": HOSTILE_FUZZ_SCHEMA_VERSION,
            "protocol_revision": self.tool_revision,
            "seed": self.seed,
            "cpu_hour_budget": self.cpu_hour_budget,
            "max_total_bytes": self.max_total_bytes,
            "cases": [case.to_record() for case in self.cases],
        }
        if include_id:
            record["corpus_id"] = self.corpus_id
        return record

    def write_manifest(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_record(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class HostileFuzzResult:
    case_id: str
    outcome: HostileOutcome
    elapsed_seconds: float
    output_bytes: int = 0
    exception_type: str | None = None
    detail: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "outcome": self.outcome.value,
            "elapsed_seconds": self.elapsed_seconds,
            "output_bytes": self.output_bytes,
            "exception_type": self.exception_type,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class HostileFuzzCampaignReport:
    corpus_id: str
    results: tuple[HostileFuzzResult, ...]
    coverage_targets: tuple[str, ...]
    retained_zero_finding_result: bool

    @property
    def counts(self) -> dict[str, int]:
        return {outcome.value: sum(result.outcome is outcome for result in self.results)
                for outcome in HostileOutcome}

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": HOSTILE_FUZZ_SCHEMA_VERSION,
            "corpus_id": self.corpus_id,
            "executions": len(self.results),
            "counts": self.counts,
            "coverage_targets": list(self.coverage_targets),
            "retained_zero_finding_result": self.retained_zero_finding_result,
            "results": [result.to_record() for result in self.results],
        }


class ParserTarget(Protocol):
    def __call__(self, payload: bytes) -> object: ...


def _case_id(
    kind: HostileInputKind, payload: bytes, seed: int, expected: HostileOutcome
) -> str:
    material = f"{kind.value}:{seed}:{expected.value}:".encode() + payload
    return f"{kind.value}-{hashlib.sha256(material).hexdigest()[:20]}"


def make_hostile_fuzz_corpus(
    *, seed: int = 406, limits: HostileFuzzLimits | None = None
) -> HostileFuzzCorpus:
    """Build deterministic minimized cases for every declared input boundary."""
    case_limits = limits or HostileFuzzLimits()
    payloads: tuple[tuple[HostileInputKind, bytes, HostileOutcome], ...] = (
        (HostileInputKind.HF_CHECKPOINT, b"{}\x00\xff", HostileOutcome.REJECTED),
        (HostileInputKind.SAFETENSORS, b"\xff" * 8 + b"\x00", HostileOutcome.REJECTED),
        (HostileInputKind.GGUF, b"GGUF\x00\x00\x00\x00", HostileOutcome.REJECTED),
        (HostileInputKind.CONFIG, b'{"model": [}', HostileOutcome.REJECTED),
        (HostileInputKind.MANIFEST, b'{"artifact": "../escape"}', HostileOutcome.REJECTED),
        (HostileInputKind.PATH, b"..\\outside\\checkpoint.safetensors", HostileOutcome.PATH_ESCAPE),
        (HostileInputKind.SUBPROCESS, b"\\x00unterminated-log", HostileOutcome.REJECTED),
    )
    cases = tuple(
        HostileFuzzCase(
            case_id=_case_id(kind, payload, seed + index, expected),
            kind=kind,
            payload=payload,
            seed=seed + index,
            expected=expected,
            limits=case_limits,
        )
        for index, (kind, payload, expected) in enumerate(payloads)
    )
    return HostileFuzzCorpus(cases=cases, seed=seed)


def run_parser_case(case: HostileFuzzCase, target: ParserTarget) -> HostileFuzzResult:
    """Run one in-process parser target and retain its failure classification."""
    started = time.perf_counter()
    try:
        target(case.payload)
    except (ValueError, KeyError, IndexError, UnicodeError, OSError, EOFError):
        outcome = HostileOutcome.REJECTED
        error_type = None
    except NotImplementedError:
        outcome = HostileOutcome.UNSUPPORTED
        error_type = "NotImplementedError"
    except MemoryError:
        outcome = HostileOutcome.RESOURCE_LIMIT
        error_type = "MemoryError"
    except Exception as error:
        outcome = HostileOutcome.FAILED
        error_type = type(error).__name__
    else:
        outcome = HostileOutcome.ACCEPTED
        error_type = None
    return HostileFuzzResult(
        case_id=case.case_id,
        outcome=outcome,
        elapsed_seconds=time.perf_counter() - started,
        exception_type=error_type,
    )


def run_subprocess_case(
    case: HostileFuzzCase,
    command: Sequence[str],
    *,
    cwd: str | Path | None = None,
) -> HostileFuzzResult:
    """Run a target with shell disabled, timeout, output, and sandbox-byte gates."""
    if not command or any(not isinstance(argument, str) or not argument for argument in command):
        raise HostileFuzzError("subprocess command must contain non-empty strings")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="modelsurgeon-fuzz-") as sandbox:
        run_cwd = Path(cwd).resolve() if cwd is not None else Path(sandbox)
        try:
            completed = subprocess.run(
                list(command),
                input=case.payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=run_cwd,
                shell=False,
                timeout=case.limits.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return HostileFuzzResult(
                case.case_id,
                HostileOutcome.TIMEOUT,
                time.perf_counter() - started,
                detail=f"timeout after {case.limits.timeout_seconds}s",
            )
        except OSError as error:
            return HostileFuzzResult(
                case.case_id,
                HostileOutcome.UNSUPPORTED,
                time.perf_counter() - started,
                exception_type=type(error).__name__,
                detail=str(error),
            )
        output_bytes = len(completed.stdout)
        disk_bytes = sum(path.stat().st_size for path in Path(sandbox).rglob("*") if path.is_file())
        if output_bytes > case.limits.max_output_bytes or disk_bytes > case.limits.max_disk_bytes:
            outcome = HostileOutcome.RESOURCE_LIMIT
        elif completed.returncode == 0:
            outcome = HostileOutcome.ACCEPTED
        else:
            outcome = HostileOutcome.REJECTED
        return HostileFuzzResult(
            case.case_id,
            outcome,
            time.perf_counter() - started,
            output_bytes=output_bytes,
            detail=f"returncode={completed.returncode}",
        )


def resolve_scoped_path(root: str | Path, relative: str | Path) -> Path:
    """Resolve a fixture path while rejecting traversal and symlink escapes."""
    root_path = Path(root).resolve(strict=True)
    candidate = root_path / Path(relative)
    if Path(relative).is_absolute() or any(part == ".." for part in Path(relative).parts):
        raise HostileFuzzError("path escapes the fixture root")
    current = root_path
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise HostileFuzzError("symlink path components are not permitted")
    resolved = candidate.resolve(strict=False)
    if os.path.commonpath((str(root_path), str(resolved))) != str(root_path):
        raise HostileFuzzError("resolved path escapes the fixture root")
    return resolved

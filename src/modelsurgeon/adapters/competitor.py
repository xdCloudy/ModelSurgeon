"""Fail-closed subprocess contract for external benchmark competitors."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from modelsurgeon.adapters.base import ModelFormat, ModelSource

COMPETITOR_ADAPTER_SCHEMA_VERSION = 1


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CompetitorAdapterError("competitor identity must be canonical JSON") from error


class CompetitorAdapterError(ValueError):
    """Raised when an adapter contract is invalid before subprocess execution."""


class CompetitorOutcome(StrEnum):
    """Distinct terminal states retained by the competitor contract."""

    SUCCESS = "success"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    CRASH = "crash"
    MALFORMED_OUTPUT = "malformed_output"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INTERRUPTED = "interrupted"
    SOURCE_MUTATED = "source_mutated"
    PUBLICATION_FAILED = "publication_failed"


@dataclass(frozen=True, slots=True)
class CompetitorIdentity:
    """Version, license, and source identity for one external method."""

    name: str
    revision: str
    license: str
    source_url: str

    def __post_init__(self) -> None:
        for label, value in (
            ("competitor name", self.name),
            ("competitor revision", self.revision),
            ("competitor license", self.license),
            ("competitor source URL", self.source_url),
        ):
            if not isinstance(value, str) or not value.strip():
                raise CompetitorAdapterError(f"{label} is required")

    def to_record(self) -> dict[str, str]:
        return {
            "name": self.name,
            "revision": self.revision,
            "license": self.license,
            "source_url": self.source_url,
        }


@dataclass(frozen=True, slots=True)
class CompetitorBudget:
    """Hard bounds observable at the subprocess boundary."""

    max_wall_seconds: float
    max_stdout_bytes: int
    max_stderr_bytes: int
    max_artifact_bytes: int
    max_disk_bytes: int

    def __post_init__(self) -> None:
        if self.max_wall_seconds <= 0:
            raise CompetitorAdapterError("max_wall_seconds must be positive")
        for label, value in (
            ("max_stdout_bytes", self.max_stdout_bytes),
            ("max_stderr_bytes", self.max_stderr_bytes),
            ("max_artifact_bytes", self.max_artifact_bytes),
            ("max_disk_bytes", self.max_disk_bytes),
        ):
            if isinstance(value, bool) or value <= 0:
                raise CompetitorAdapterError(f"{label} must be positive")

    def to_record(self) -> dict[str, int | float]:
        return {
            "max_wall_seconds": self.max_wall_seconds,
            "max_stdout_bytes": self.max_stdout_bytes,
            "max_stderr_bytes": self.max_stderr_bytes,
            "max_artifact_bytes": self.max_artifact_bytes,
            "max_disk_bytes": self.max_disk_bytes,
        }


@dataclass(frozen=True, slots=True)
class CompetitorTelemetry:
    """Measured subprocess and artifact resource facts."""

    wall_seconds: float
    stdout_bytes: int
    stderr_bytes: int
    artifact_bytes: int
    disk_bytes: int

    def to_record(self) -> dict[str, int | float]:
        return {
            "wall_seconds": self.wall_seconds,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "artifact_bytes": self.artifact_bytes,
            "disk_bytes": self.disk_bytes,
        }


@dataclass(frozen=True, slots=True)
class CompetitorProvenance:
    """Complete reproducibility lineage for an attempted competitor run."""

    identity: CompetitorIdentity
    source: ModelSource
    command: tuple[str, ...]
    version_command: tuple[str, ...]
    executable_version: str | None
    seed: int
    config_digest: str
    environment: tuple[tuple[str, str], ...]
    artifact_path: str | None
    artifact_sha256: str | None
    artifact_size_bytes: int | None
    telemetry: CompetitorTelemetry
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        if self.seed < 0 or not self.command or not self.version_command:
            raise CompetitorAdapterError("provenance requires commands and an unsigned seed")
        if not self.config_digest:
            raise CompetitorAdapterError("provenance requires a configuration digest")
        if len(self.environment) != len(set(self.environment)):
            raise CompetitorAdapterError("provenance environment keys must be unique")

    def to_record(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_record(),
            "source": self.source.to_record(),
            "command": list(self.command),
            "version_command": list(self.version_command),
            "executable_version": self.executable_version,
            "seed": self.seed,
            "config_digest": self.config_digest,
            "environment": dict(self.environment),
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "telemetry": self.telemetry.to_record(),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True, slots=True)
class CompetitorRunResult:
    """Terminal result; success is impossible without version and artifact evidence."""

    outcome: CompetitorOutcome
    reason: str
    provenance: CompetitorProvenance
    schema_version: int = COMPETITOR_ADAPTER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != COMPETITOR_ADAPTER_SCHEMA_VERSION:
            raise CompetitorAdapterError("unsupported competitor result schema version")
        if not self.reason.strip():
            raise CompetitorAdapterError("competitor outcomes require a reason")
        if self.outcome is CompetitorOutcome.SUCCESS:
            if not self.provenance.executable_version:
                raise CompetitorAdapterError("success requires an executable version")
            if not self.provenance.artifact_sha256 or self.provenance.artifact_size_bytes is None:
                raise CompetitorAdapterError("success requires artifact-bound evidence")

    @property
    def run_id(self) -> str:
        identity = self.provenance.to_record().copy()
        identity.pop("artifact_path", None)
        identity.pop("stdout", None)
        identity.pop("stderr", None)
        identity.update({"outcome": self.outcome.value, "schema_version": self.schema_version})
        digest = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
        return f"competitor_{digest}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "provenance": self.provenance.to_record(),
        }


def _digest_path(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    if path.is_file():
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                total += len(chunk)
        return digest.hexdigest(), total
    if not path.is_dir():
        raise CompetitorAdapterError("artifact path is not a file or directory")
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode()
        digest.update(relative)
        child_digest, child_size = _digest_path(child)
        digest.update(child_digest.encode())
        total += child_size
    return digest.hexdigest(), total


def _source_digest(path: Path) -> str:
    return _digest_path(path)[0]


def _json_object(value: str) -> Mapping[str, object] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return cast(Mapping[str, object], parsed)


class SubprocessCompetitorAdapter:
    """Run one external method with array-only commands and transactional output."""

    def __init__(
        self,
        identity: CompetitorIdentity,
        *,
        model_format: ModelFormat,
        operations: frozenset[str],
        version_command: tuple[str, ...],
        command_template: tuple[str, ...],
        budget: CompetitorBudget,
        config: Mapping[str, object] | None = None,
    ) -> None:
        if not version_command or not command_template:
            raise CompetitorAdapterError("competitor commands must be non-empty argument arrays")
        if not operations:
            raise CompetitorAdapterError("competitor adapter must declare an operation")
        self.identity = identity
        self.model_format = model_format
        self.operations = frozenset(operations)
        self.version_command = tuple(version_command)
        self.command_template = tuple(command_template)
        self.budget = budget
        self.config = dict(config or {})
        self._verified_operations: set[str] = set()

    @property
    def capabilities(self) -> frozenset[str]:
        """Only operations with artifact-bound success are advertised."""

        return frozenset(self._verified_operations)

    def capability_decision(self, operation: str) -> tuple[bool, str]:
        if operation not in self.operations:
            return False, f"operation {operation!r} is not declared by this adapter"
        if operation not in self._verified_operations:
            return False, "no executable-version and artifact-bound successful result is retained"
        return True, "verified by an artifact-bound successful result"

    def _command(self, source: Path, output: Path, seed: int) -> tuple[str, ...]:
        values = {"source": str(source), "output": str(output), "seed": str(seed)}
        try:
            return tuple(argument.format(**values) for argument in self.command_template)
        except KeyError as error:
            raise CompetitorAdapterError(
                f"unsupported command template placeholder: {error}"
            ) from error

    def _provenance(
        self,
        source: ModelSource,
        command: tuple[str, ...],
        *,
        version: str | None,
        seed: int,
        environment: tuple[tuple[str, str], ...],
        artifact_path: str | None,
        artifact_sha256: str | None,
        artifact_size_bytes: int | None,
        telemetry: CompetitorTelemetry,
        stdout: bytes,
        stderr: bytes,
    ) -> CompetitorProvenance:
        config_digest = hashlib.sha256(_canonical_json(self.config).encode()).hexdigest()
        return CompetitorProvenance(
            self.identity,
            source,
            command,
            self.version_command,
            version,
            seed,
            config_digest,
            environment,
            artifact_path,
            artifact_sha256,
            artifact_size_bytes,
            telemetry,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )

    def run(
        self,
        source: ModelSource,
        source_path: str | Path,
        destination: str | Path,
        *,
        operation: str,
        seed: int,
    ) -> CompetitorRunResult:
        """Run a competitor and publish only a verified artifact outside the source."""

        source_resolved = Path(source_path).expanduser().absolute().resolve(strict=False)
        destination_resolved = Path(destination).expanduser().absolute().resolve(strict=False)
        if source.format is not self.model_format or operation not in self.operations:
            telemetry = CompetitorTelemetry(0.0, 0, 0, 0, 0)
            provenance = self._provenance(
                source,
                self.command_template,
                version=None,
                seed=seed,
                environment=(),
                artifact_path=None,
                artifact_sha256=None,
                artifact_size_bytes=None,
                telemetry=telemetry,
                stdout=b"",
                stderr=b"",
            )
            return CompetitorRunResult(
                CompetitorOutcome.UNSUPPORTED,
                "model format or operation is not supported by this adapter",
                provenance,
            )
        if not source_resolved.exists():
            raise CompetitorAdapterError("competitor source checkpoint does not exist")

        environment = (("PYTHONHASHSEED", str(seed)), ("MODELSURGEON_SEED", str(seed)))
        process_environment = {key: value for key, value in environment}
        path_value = os.environ.get("PATH")
        if path_value:
            process_environment["PATH"] = path_value
        version_started = time.monotonic()
        try:
            version_process = subprocess.run(
                self.version_command,
                capture_output=True,
                check=False,
                env=process_environment,
                timeout=self.budget.max_wall_seconds,
                shell=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as error:
            telemetry = CompetitorTelemetry(time.monotonic() - version_started, 0, 0, 0, 0)
            provenance = self._provenance(
                source,
                self.version_command,
                version=None,
                seed=seed,
                environment=environment,
                artifact_path=None,
                artifact_sha256=None,
                artifact_size_bytes=None,
                telemetry=telemetry,
                stdout=b"",
                stderr=str(error).encode(),
            )
            return CompetitorRunResult(
                CompetitorOutcome.UNSUPPORTED,
                "executable version could not be established",
                provenance,
            )
        if version_process.returncode != 0 or not version_process.stdout.strip():
            telemetry = CompetitorTelemetry(
                time.monotonic() - version_started,
                len(version_process.stdout),
                len(version_process.stderr),
                0,
                0,
            )
            provenance = self._provenance(
                source,
                self.version_command,
                version=None,
                seed=seed,
                environment=environment,
                artifact_path=None,
                artifact_sha256=None,
                artifact_size_bytes=None,
                telemetry=telemetry,
                stdout=version_process.stdout,
                stderr=version_process.stderr,
            )
            return CompetitorRunResult(
                CompetitorOutcome.UNSUPPORTED,
                "executable version command failed or returned empty output",
                provenance,
            )
        executable_version = version_process.stdout.decode("utf-8", errors="replace").strip()

        # Import lazily: the surgery package also imports the adapter namespace.
        from modelsurgeon.surgery.checkpoint_destination import (
            AtomicCheckpointDestination,
            CheckpointDestinationError,
        )

        try:
            with AtomicCheckpointDestination(source_resolved, destination_resolved) as publication:
                before_digest = _source_digest(source_resolved)
                command = self._command(source_resolved, publication.staging_path, seed)
                workdir = tempfile.mkdtemp(prefix="modelsurgeon-competitor-")
                started = time.monotonic()
                try:
                    process = subprocess.run(
                        command,
                        capture_output=True,
                        check=False,
                        cwd=workdir,
                        env=process_environment,
                        timeout=self.budget.max_wall_seconds,
                        shell=False,
                    )
                except subprocess.TimeoutExpired as error:
                    telemetry = CompetitorTelemetry(
                        time.monotonic() - started,
                        len(error.stdout or b""),
                        len(error.stderr or b""),
                        0,
                        0,
                    )
                    provenance = self._provenance(
                        source,
                        command,
                        version=executable_version,
                        seed=seed,
                        environment=environment,
                        artifact_path=None,
                        artifact_sha256=None,
                        artifact_size_bytes=None,
                        telemetry=telemetry,
                        stdout=error.stdout or b"",
                        stderr=error.stderr or b"",
                    )
                    return CompetitorRunResult(
                        CompetitorOutcome.TIMEOUT,
                        "competitor exceeded the wall-time budget",
                        provenance,
                    )
                except KeyboardInterrupt:
                    telemetry = CompetitorTelemetry(time.monotonic() - started, 0, 0, 0, 0)
                    provenance = self._provenance(
                        source,
                        command,
                        version=executable_version,
                        seed=seed,
                        environment=environment,
                        artifact_path=None,
                        artifact_sha256=None,
                        artifact_size_bytes=None,
                        telemetry=telemetry,
                        stdout=b"",
                        stderr=b"interrupted",
                    )
                    return CompetitorRunResult(
                        CompetitorOutcome.INTERRUPTED,
                        "competitor execution was interrupted",
                        provenance,
                    )
                finally:
                    shutil.rmtree(workdir, ignore_errors=True)

                stdout, stderr = process.stdout, process.stderr
                if (
                    len(stdout) > self.budget.max_stdout_bytes
                    or len(stderr) > self.budget.max_stderr_bytes
                ):
                    telemetry = CompetitorTelemetry(
                        time.monotonic() - started, len(stdout), len(stderr), 0, 0
                    )
                    provenance = self._provenance(
                        source,
                        command,
                        version=executable_version,
                        seed=seed,
                        environment=environment,
                        artifact_path=None,
                        artifact_sha256=None,
                        artifact_size_bytes=None,
                        telemetry=telemetry,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    return CompetitorRunResult(
                        CompetitorOutcome.BUDGET_EXHAUSTED,
                        "competitor log output exceeded the configured budget",
                        provenance,
                    )
                telemetry = CompetitorTelemetry(
                    time.monotonic() - started, len(stdout), len(stderr), 0, 0
                )
                if process.returncode != 0:
                    provenance = self._provenance(
                        source,
                        command,
                        version=executable_version,
                        seed=seed,
                        environment=environment,
                        artifact_path=None,
                        artifact_sha256=None,
                        artifact_size_bytes=None,
                        telemetry=telemetry,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    return CompetitorRunResult(
                        CompetitorOutcome.CRASH,
                        f"competitor exited with status {process.returncode}",
                        provenance,
                    )
                payload = _json_object(stdout.decode("utf-8", errors="replace"))
                expected_path = publication.staging_path.absolute().resolve()
                if payload is None or payload.get("status") != "success":
                    provenance = self._provenance(
                        source,
                        command,
                        version=executable_version,
                        seed=seed,
                        environment=environment,
                        artifact_path=None,
                        artifact_sha256=None,
                        artifact_size_bytes=None,
                        telemetry=telemetry,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    return CompetitorRunResult(
                        CompetitorOutcome.MALFORMED_OUTPUT,
                        "competitor did not emit the required success JSON object",
                        provenance,
                    )
                reported_path = payload.get("artifact_path")
                reported_digest = payload.get("artifact_sha256")
                reported_size = payload.get("artifact_size_bytes")
                if (
                    not isinstance(reported_path, str)
                    or Path(reported_path).absolute().resolve() != expected_path
                    or not isinstance(reported_digest, str)
                    or not isinstance(reported_size, int)
                    or isinstance(reported_size, bool)
                    or not publication.staging_path.exists()
                ):
                    provenance = self._provenance(
                        source,
                        command,
                        version=executable_version,
                        seed=seed,
                        environment=environment,
                        artifact_path=None,
                        artifact_sha256=None,
                        artifact_size_bytes=None,
                        telemetry=telemetry,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    return CompetitorRunResult(
                        CompetitorOutcome.MALFORMED_OUTPUT,
                        "competitor output did not bind to the transactional artifact",
                        provenance,
                    )
                artifact_digest, artifact_size = _digest_path(publication.staging_path)
                telemetry = CompetitorTelemetry(
                    time.monotonic() - started,
                    len(stdout),
                    len(stderr),
                    artifact_size,
                    artifact_size,
                )
                if (
                    artifact_size <= 0
                    or artifact_size > self.budget.max_artifact_bytes
                    or artifact_size > self.budget.max_disk_bytes
                    or reported_digest != artifact_digest
                    or reported_size != artifact_size
                ):
                    provenance = self._provenance(
                        source,
                        command,
                        version=executable_version,
                        seed=seed,
                        environment=environment,
                        artifact_path=None,
                        artifact_sha256=artifact_digest,
                        artifact_size_bytes=artifact_size,
                        telemetry=telemetry,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    return CompetitorRunResult(
                        CompetitorOutcome.BUDGET_EXHAUSTED
                        if artifact_size > self.budget.max_artifact_bytes
                        or artifact_size > self.budget.max_disk_bytes
                        else CompetitorOutcome.MALFORMED_OUTPUT,
                        "artifact size or digest did not satisfy the bounded output contract",
                        provenance,
                    )
                if _source_digest(source_resolved) != before_digest:
                    provenance = self._provenance(
                        source,
                        command,
                        version=executable_version,
                        seed=seed,
                        environment=environment,
                        artifact_path=None,
                        artifact_sha256=artifact_digest,
                        artifact_size_bytes=artifact_size,
                        telemetry=telemetry,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    return CompetitorRunResult(
                        CompetitorOutcome.SOURCE_MUTATED,
                        "competitor changed the read-only source checkpoint",
                        provenance,
                    )
                try:
                    published = publication.publish()
                except CheckpointDestinationError as error:
                    provenance = self._provenance(
                        source,
                        command,
                        version=executable_version,
                        seed=seed,
                        environment=environment,
                        artifact_path=None,
                        artifact_sha256=artifact_digest,
                        artifact_size_bytes=artifact_size,
                        telemetry=telemetry,
                        stdout=stdout,
                        stderr=stderr,
                    )
                    return CompetitorRunResult(
                        CompetitorOutcome.PUBLICATION_FAILED,
                        str(error),
                        provenance,
                    )
                provenance = self._provenance(
                    source,
                    command,
                    version=executable_version,
                    seed=seed,
                    environment=environment,
                    artifact_path=str(published),
                    artifact_sha256=artifact_digest,
                    artifact_size_bytes=artifact_size,
                    telemetry=telemetry,
                    stdout=stdout,
                    stderr=stderr,
                )
                result = CompetitorRunResult(
                    CompetitorOutcome.SUCCESS,
                    "artifact-bound competitor execution succeeded",
                    provenance,
                )
                self._verified_operations.add(operation)
                return result
        except CheckpointDestinationError as error:
            raise CompetitorAdapterError(str(error)) from error

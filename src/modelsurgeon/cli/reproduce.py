"""Integrity-checked reconstruction and replay of one persisted experiment run."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol, cast, runtime_checkable

import typer

from modelsurgeon.experiments.artifacts import ContentAddressedArtifactStore
from modelsurgeon.experiments.hardware import HardwareInventory, collect_hardware_inventory
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.reproducibility import (
    REPRODUCIBILITY_ARTIFACT_ROLE,
    GitRevision,
    LockDigest,
    collect_git_revision,
    digest_lock_file,
    load_reproducibility_manifest,
)
from modelsurgeon.experiments.store import ExperimentMetadataStore

REPRODUCTION_RESULT_SCHEMA_VERSION = 1


class ReproductionCommandError(RuntimeError):
    """Raised when a persisted run cannot be reconstructed or replayed safely."""


@dataclass(frozen=True, slots=True)
class ReproductionMismatch:
    code: str
    path: str
    expected: object
    actual: object

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code,
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True, slots=True)
class ReproductionPlan:
    run_id: str
    candidate_id: str
    manifest_id: str
    manifest_digest: str
    command: tuple[str, ...]
    resolved_config: dict[str, object]
    inputs: dict[str, object]
    original_metrics: tuple[tuple[str, float], ...]
    tolerances: tuple[tuple[str, float, float], ...]
    mismatches: tuple[ReproductionMismatch, ...]

    @property
    def executable(self) -> bool:
        return not self.mismatches

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "reproduction_plan",
            "schema_version": REPRODUCTION_RESULT_SCHEMA_VERSION,
            "original_run_id": self.run_id,
            "original_candidate_id": self.candidate_id,
            "manifest_id": self.manifest_id,
            "manifest_digest": self.manifest_digest,
            "exact_inputs": self.inputs,
            "resolved_config": self.resolved_config,
            "exact_command": list(self.command),
            "original_metrics": dict(self.original_metrics),
            "metric_tolerances": {
                name: {"absolute": absolute, "relative": relative}
                for name, absolute, relative in self.tolerances
            },
            "environment_mismatches": [item.to_record() for item in self.mismatches],
            "executable": self.executable,
        }


@dataclass(frozen=True, slots=True)
class MetricComparison:
    metric: str
    original: float | None
    reproduced: float | None
    absolute_delta: float | None
    allowed_delta: float | None
    passed: bool
    detail: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "original": self.original,
            "reproduced": self.reproduced,
            "absolute_delta": self.absolute_delta,
            "allowed_delta": self.allowed_delta,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    run_id: str
    candidate_id: str
    manifest_id: str
    command: tuple[str, ...]
    comparisons: tuple[MetricComparison, ...]

    @property
    def passed(self) -> bool:
        return bool(self.comparisons) and all(item.passed for item in self.comparisons)

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": REPRODUCTION_RESULT_SCHEMA_VERSION,
            "original_run_id": self.run_id,
            "original_candidate_id": self.candidate_id,
            "manifest_id": self.manifest_id,
            "exact_command": list(self.command),
            "comparisons": [item.to_record() for item in self.comparisons],
        }

    @property
    def reproduction_id(self) -> str:
        digest = hashlib.sha256(
            canonical_identity_json(self._identity_record()).encode("utf-8")
        ).hexdigest()
        return f"reproduction_{digest}"

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "reproduction_result",
            **self._identity_record(),
            "reproduction_id": self.reproduction_id,
            "passed": self.passed,
        }


@runtime_checkable
class ReproductionExecutor(Protocol):
    """Explicitly supplied trusted adapter that reruns a reconstructed recipe."""

    def execute(self, plan: ReproductionPlan) -> Mapping[str, float]: ...


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ReproductionCommandError(f"{path} must be an object")
    return cast(dict[str, object], value)


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReproductionCommandError(f"{path} must be a non-empty string")
    return value


def _hardware_identity(value: Mapping[str, object]) -> dict[str, object]:
    """Drop intentionally volatile capacity readings while retaining platform identity."""

    hardware = dict(value)
    memory = _object(hardware.get("memory"), "hardware.memory")
    disk = _object(hardware.get("disk"), "hardware.disk")
    hardware["memory"] = {"total_bytes": memory.get("total_bytes")}
    hardware["disk"] = {"total_bytes": disk.get("total_bytes")}
    hardware["warnings"] = []
    return hardware


def _mismatch(
    result: list[ReproductionMismatch],
    code: str,
    path: str,
    expected: object,
    actual: object,
) -> None:
    if expected != actual:
        result.append(ReproductionMismatch(code, path, expected, actual))


def prepare_reproduction(
    run_id: str,
    *,
    metadata_path: str | Path,
    artifact_root: str | Path,
    repository_root: str | Path,
    lock_path: str | Path,
    current_git: GitRevision | None = None,
    current_lock: LockDigest | None = None,
    current_hardware: HardwareInventory | None = None,
) -> ReproductionPlan:
    """Resolve one run and verify all immutable recipe and environment evidence."""

    if not run_id.startswith("run_"):
        raise ReproductionCommandError("RUN_ID must use the canonical run_ prefix")
    metadata_file = Path(metadata_path)
    artifact_directory = Path(artifact_root)
    if not metadata_file.is_file() or metadata_file.is_symlink():
        raise ReproductionCommandError(
            "metadata path must be an existing regular SQLite file"
        )
    if not artifact_directory.is_dir() or artifact_directory.is_symlink():
        raise ReproductionCommandError(
            "artifact root must be an existing non-symlink directory"
        )
    if not (artifact_directory / "sha256").is_dir():
        raise ReproductionCommandError("artifact root has no SHA-256 content store")
    artifacts = ContentAddressedArtifactStore(artifact_directory)
    with ExperimentMetadataStore(metadata_file) as metadata:
        run = metadata.get_run(run_id)
        if run is None:
            raise ReproductionCommandError(f"run not found: {run_id}")
        stored_input = metadata.get_input(run.input_id)
        if stored_input is None:
            raise ReproductionCommandError("run input metadata is missing")
        candidates = metadata.list_run_candidates(run_id)
        manifest_links = [
            (candidate, reference)
            for candidate in candidates
            for reference in metadata.list_artifact_references(candidate.candidate_id)
            if reference.role == REPRODUCIBILITY_ARTIFACT_ROLE
        ]
        if len(manifest_links) != 1:
            raise ReproductionCommandError(
                "RUN_ID must resolve to exactly one reproducibility manifest; "
                f"found {len(manifest_links)}"
            )
        candidate, reference = manifest_links[0]
        original_metrics = tuple(
            sorted(
                (
                    f"{item.phase.value}:{item.name}",
                    item.value,
                )
                for item in metadata.list_metrics(candidate.candidate_id)
                if item.state == "measured" and item.value is not None
            )
        )
        if not original_metrics:
            raise ReproductionCommandError(
                "run has no measured metrics to compare during reproduction"
            )

    artifact = artifacts.get(reference.digest)
    raw = load_reproducibility_manifest(artifact.data_path.read_text(encoding="utf-8"))
    manifest_id = _string(raw.get("manifest_id"), "manifest_id")
    resolved_config = _object(raw.get("resolved_config"), "resolved_config")
    command_raw = raw.get("command")
    if not isinstance(command_raw, list) or not all(
        isinstance(item, str) and item for item in command_raw
    ):
        raise ReproductionCommandError("manifest command must be a non-empty string array")
    command = tuple(command_raw)
    if not command:
        raise ReproductionCommandError("manifest command cannot be empty")
    tolerance_raw = raw.get("metric_tolerances")
    if not isinstance(tolerance_raw, list):
        raise ReproductionCommandError("manifest metric tolerances must be an array")
    tolerances: list[tuple[str, float, float]] = []
    for index, value in enumerate(tolerance_raw):
        item = _object(value, f"metric_tolerances[{index}]")
        try:
            name = _string(item.get("metric"), f"metric_tolerances[{index}].metric")
            absolute_value = item["absolute"]
            relative_value = item["relative"]
        except (KeyError, TypeError, ValueError) as error:
            raise ReproductionCommandError("manifest metric tolerance is malformed") from error
        if not isinstance(absolute_value, (int, float)) or isinstance(absolute_value, bool):
            raise ReproductionCommandError("manifest absolute metric tolerance is malformed")
        if not isinstance(relative_value, (int, float)) or isinstance(relative_value, bool):
            raise ReproductionCommandError("manifest relative metric tolerance is malformed")
        absolute = float(absolute_value)
        relative = float(relative_value)
        if any(not math.isfinite(number) or number < 0 for number in (absolute, relative)):
            raise ReproductionCommandError("manifest metric tolerance is invalid")
        tolerances.append((name, absolute, relative))
    if tuple(name for name, _, _ in tolerances) != tuple(
        sorted({name for name, _, _ in tolerances})
    ):
        raise ReproductionCommandError("manifest metric tolerances are not canonical")

    git = current_git or collect_git_revision(repository_root)
    lock = current_lock or digest_lock_file(lock_path)
    hardware = current_hardware or collect_hardware_inventory(artifact_root)
    expected_git = _object(raw.get("git"), "git")
    expected_lock = _object(raw.get("lock"), "lock")
    expected_hardware = _object(raw.get("hardware"), "hardware")
    expected_model = _object(raw.get("model"), "model")
    expected_dataset = _object(raw.get("dataset"), "dataset")
    expected_versions = _object(raw.get("versions"), "versions")
    expected_seeds = _object(raw.get("seeds"), "seeds")
    mismatches: list[ReproductionMismatch] = []
    _mismatch(mismatches, "run_identity", "manifest.run_id", run_id, raw.get("run_id"))
    _mismatch(
        mismatches,
        "experiment_identity",
        "manifest.experiment_id",
        run.experiment_id,
        raw.get("experiment_id"),
    )
    _mismatch(
        mismatches,
        "config_digest",
        "versions.config_digest",
        stored_input.config_digest,
        expected_versions.get("config_digest"),
    )
    resolved_digest = hashlib.sha256(
        canonical_identity_json(resolved_config).encode("utf-8")
    ).hexdigest()
    _mismatch(
        mismatches,
        "resolved_config_digest",
        "resolved_config",
        stored_input.config_digest,
        resolved_digest,
    )
    stored_model = {
        "identifier": stored_input.model_identifier,
        "revision": stored_input.model_revision,
        "family": stored_input.model_family,
        "format": stored_input.model_format,
        "parameter_count": stored_input.model_parameter_count,
        "quantization": stored_input.model_quantization,
    }
    stored_dataset = {
        "identifier": stored_input.dataset_identifier,
        "revision": stored_input.dataset_revision,
        "split": stored_input.dataset_split,
        "manifest_id": stored_input.dataset_manifest_id,
        "tokenizer": stored_input.tokenizer,
        "tokenizer_revision": stored_input.tokenizer_revision,
    }
    _mismatch(mismatches, "model_identity", "model", stored_model, expected_model)
    _mismatch(mismatches, "dataset_identity", "dataset", stored_dataset, expected_dataset)
    _mismatch(
        mismatches,
        "attempt_identity",
        "manifest.attempt_id",
        run.attempt_id,
        raw.get("attempt_id"),
    )
    _mismatch(
        mismatches,
        "manifest_reference",
        "reference.manifest_id",
        manifest_id,
        reference.metadata.get("manifest_id"),
    )
    _mismatch(
        mismatches,
        "manifest_reference",
        "reference.run_id",
        run_id,
        reference.metadata.get("run_id"),
    )
    _mismatch(mismatches, "git_revision", "git.commit", expected_git.get("commit"), git.commit)
    _mismatch(mismatches, "git_cleanliness", "git.clean", True, git.clean)
    _mismatch(
        mismatches,
        "lock_digest",
        "lock.hexadecimal",
        expected_lock.get("hexadecimal"),
        lock.hexadecimal,
    )
    _mismatch(
        mismatches,
        "lock_name",
        "lock.name",
        expected_lock.get("name"),
        lock.name,
    )
    _mismatch(
        mismatches,
        "hardware_environment",
        "hardware",
        _hardware_identity(expected_hardware),
        _hardware_identity(hardware.to_record()),
    )
    if raw.get("reproducible") is not True or raw.get("issues") != []:
        mismatches.append(
            ReproductionMismatch(
                "manifest_not_reproducible",
                "manifest.reproducible",
                True,
                raw.get("reproducible"),
            )
        )
    inputs: dict[str, object] = {
        "model": expected_model,
        "dataset": expected_dataset,
        "versions": expected_versions,
        "seeds": expected_seeds,
        "mutation": run.mutation,
        "outcome": run.outcome,
    }
    return ReproductionPlan(
        run_id,
        candidate.candidate_id,
        manifest_id,
        reference.digest,
        command,
        resolved_config,
        inputs,
        original_metrics,
        tuple(tolerances),
        tuple(sorted(mismatches, key=lambda item: (item.code, item.path))),
    )


def load_reproduction_executor(specification: str) -> ReproductionExecutor:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ReproductionCommandError("executor must use module:factory syntax")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
        executor = factory()
    except Exception as error:
        raise ReproductionCommandError(
            f"cannot load reproduction executor {specification!r}"
        ) from error
    if not isinstance(executor, ReproductionExecutor):
        raise ReproductionCommandError("executor does not implement execute(plan)")
    return executor


def run_reproduction(
    plan: ReproductionPlan,
    executor: ReproductionExecutor,
) -> ReproductionResult:
    if plan.mismatches:
        raise ReproductionCommandError(
            "environment does not match the original run; inspect --dry-run output"
        )
    try:
        measured = dict(executor.execute(plan))
    except Exception as error:
        raise ReproductionCommandError("reproduction executor failed") from error
    if not all(
        isinstance(name, str)
        and name
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for name, value in measured.items()
    ):
        raise ReproductionCommandError("executor returned malformed metrics")
    reproduced = {name: float(value) for name, value in measured.items()}
    originals = dict(plan.original_metrics)
    tolerances = {
        name: (absolute, relative) for name, absolute, relative in plan.tolerances
    }
    comparisons: list[MetricComparison] = []
    for name in sorted(originals):
        original = originals[name]
        value = reproduced.get(name)
        absolute, relative = tolerances.get(name, (0.0, 0.0))
        allowed = max(absolute, abs(original) * relative)
        if value is None:
            comparisons.append(
                MetricComparison(name, original, None, None, allowed, False, "metric missing")
            )
            continue
        delta = abs(value - original)
        comparisons.append(
            MetricComparison(name, original, value, delta, allowed, delta <= allowed)
        )
    for name in sorted(set(reproduced) - set(originals)):
        comparisons.append(
            MetricComparison(
                name,
                None,
                reproduced[name],
                None,
                None,
                False,
                "unexpected metric",
            )
        )
    return ReproductionResult(
        plan.run_id,
        plan.candidate_id,
        plan.manifest_id,
        plan.command,
        tuple(comparisons),
    )


def _write_result(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_identity_json(value) + "\n")
    except FileExistsError as error:
        raise ReproductionCommandError(f"output already exists: {path}") from error


def reproduce_command(
    run_id: Annotated[str, typer.Argument(help="Canonical persisted RUN_ID")],
    metadata: Annotated[
        Path,
        typer.Option("--metadata", help="Experiment metadata SQLite database"),
    ],
    artifacts: Annotated[
        Path,
        typer.Option("--artifacts", help="Content-addressed artifact-store root"),
    ],
    lock: Annotated[
        Path,
        typer.Option("--lock", help="Dependency lock file captured by the run"),
    ] = Path("uv.lock"),
    repository: Annotated[
        Path,
        typer.Option("--repository", help="Git repository whose revision must match"),
    ] = Path("."),
    executor: Annotated[
        str | None,
        typer.Option("--executor", help="Trusted replay adapter as module:factory"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print exact inputs, command, and mismatches only"),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Save the canonical result without overwriting"),
    ] = None,
) -> None:
    """Reconstruct and optionally rerun one immutable experiment recipe."""

    try:
        plan = prepare_reproduction(
            run_id,
            metadata_path=metadata,
            artifact_root=artifacts,
            repository_root=repository,
            lock_path=lock,
        )
        if dry_run:
            record = plan.to_record()
        else:
            if executor is None:
                raise ReproductionCommandError("--executor is required unless --dry-run is used")
            record = run_reproduction(plan, load_reproduction_executor(executor)).to_record()
        if output is not None:
            _write_result(output, record)
        typer.echo(canonical_identity_json(record))
        if not dry_run and record["passed"] is not True:
            raise typer.Exit(1)
    except ReproductionCommandError as error:
        typer.echo(f"reproduce error: {error}", err=True)
        raise typer.Exit(2) from error

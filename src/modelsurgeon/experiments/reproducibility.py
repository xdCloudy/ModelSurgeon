"""Immutable, content-addressable experiment reproducibility manifests."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

from modelsurgeon.experiments.artifacts import (
    ContentAddressedArtifactStore,
    PublishedArtifactReference,
)
from modelsurgeon.experiments.hardware import HardwareInventory
from modelsurgeon.experiments.identity import (
    canonical_identity_json,
    canonicalize_identity_value,
)
from modelsurgeon.experiments.schema import (
    DatasetTarget,
    ExperimentRecord,
    ModelTarget,
    SeedContext,
    VersionContext,
)
from modelsurgeon.experiments.store import ExperimentMetadataStore, PersistedExperiment

REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION: Literal[2] = 2
REPRODUCIBILITY_ARTIFACT_ROLE = "reproducibility_manifest"
LOCK_DIGEST_ALGORITHM = "sha256"


class ReproducibilityError(ValueError):
    """Raised when reproducibility evidence is malformed or cannot be linked safely."""


class ReproducibilityIssueCode(StrEnum):
    MISSING_GIT_REVISION = "missing_git_revision"
    DIRTY_GIT_WORKTREE = "dirty_git_worktree"
    MISSING_CONFIG_DIGEST = "missing_config_digest"
    MISSING_MODEL_REVISION = "missing_model_revision"
    MISSING_DATASET_REVISION = "missing_dataset_revision"
    MISSING_DATASET_MANIFEST = "missing_dataset_manifest"
    MISSING_TOKENIZER_REVISION = "missing_tokenizer_revision"
    MISSING_TOOL_REVISION = "missing_tool_revision"
    MISSING_EVALUATOR_REVISION = "missing_evaluator_revision"
    MISSING_LOCK_DIGEST = "missing_lock_digest"


@dataclass(frozen=True, slots=True)
class MetricTolerance:
    """Allowed absolute and relative drift for one phase-qualified metric."""

    metric: str
    absolute: float = 0.0
    relative: float = 0.0

    def __post_init__(self) -> None:
        if not self.metric or ":" not in self.metric:
            raise ReproducibilityError(
                "metric tolerance names must use phase:name syntax"
            )
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.absolute, self.relative)
        ):
            raise ReproducibilityError("metric tolerances must be finite and non-negative")

    def to_record(self) -> dict[str, str | float]:
        return {
            "metric": self.metric,
            "absolute": self.absolute,
            "relative": self.relative,
        }


@dataclass(frozen=True, slots=True)
class GitRevision:
    commit: str | None
    clean: bool

    def __post_init__(self) -> None:
        if self.commit is not None:
            if self.commit != self.commit.lower():
                raise ReproducibilityError("git commit must use canonical lowercase hexadecimal")
            if len(self.commit) not in {40, 64} or any(
                character not in "0123456789abcdef" for character in self.commit
            ):
                raise ReproducibilityError("git commit must be a full SHA-1 or SHA-256 object ID")

    def to_record(self) -> dict[str, object]:
        return {"commit": self.commit, "clean": self.clean}


@dataclass(frozen=True, slots=True)
class LockDigest:
    name: str
    hexadecimal: str | None
    algorithm: str = LOCK_DIGEST_ALGORITHM

    def __post_init__(self) -> None:
        if not self.name:
            raise ReproducibilityError("dependency lock identity requires a non-empty name")
        if self.algorithm != LOCK_DIGEST_ALGORITHM:
            raise ReproducibilityError(f"unsupported dependency lock digest {self.algorithm!r}")
        invalid_hexadecimal = self.hexadecimal is not None and (
            len(self.hexadecimal) != 64
            or any(character not in "0123456789abcdef" for character in self.hexadecimal)
        )
        if invalid_hexadecimal:
            raise ReproducibilityError(
                "lock SHA-256 must be 64 lowercase hexadecimal characters"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "hexadecimal": self.hexadecimal,
        }


@dataclass(frozen=True, slots=True)
class ReproducibilityIssue:
    code: ReproducibilityIssueCode
    path: str
    detail: str

    def __post_init__(self) -> None:
        if not self.path or not self.detail:
            raise ReproducibilityError("reproducibility issues require path and detail")

    def to_record(self) -> dict[str, str]:
        return {"code": self.code.value, "path": self.path, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class ReproducibilityManifest:
    run_id: str
    experiment_id: str
    attempt_id: str
    git: GitRevision
    model: ModelTarget
    dataset: DatasetTarget
    versions: VersionContext
    seeds: SeedContext
    hardware: HardwareInventory
    lock: LockDigest
    resolved_config: Mapping[str, object]
    command: tuple[str, ...]
    metric_tolerances: tuple[MetricTolerance, ...] = ()
    schema_version: Literal[2] = REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION:
            raise ReproducibilityError(
                f"unsupported reproducibility manifest schema {self.schema_version}"
            )
        if not self.run_id or not self.experiment_id or not self.attempt_id:
            raise ReproducibilityError("manifest requires run, experiment, and attempt identities")
        if not self.resolved_config:
            raise ReproducibilityError("manifest requires the complete resolved configuration")
        canonical_config = canonical_identity_json(self.resolved_config)
        config_digest = hashlib.sha256(canonical_config.encode("utf-8")).hexdigest()
        if config_digest != self.versions.config_digest:
            raise ReproducibilityError(
                "resolved configuration does not match the recorded config digest"
            )
        if not self.command or any(not item for item in self.command):
            raise ReproducibilityError("manifest requires an exact non-empty command")
        tolerance_names = tuple(item.metric for item in self.metric_tolerances)
        if tolerance_names != tuple(sorted(set(tolerance_names))):
            raise ReproducibilityError(
                "metric tolerances must use unique canonical metric names"
            )

    @property
    def issues(self) -> tuple[ReproducibilityIssue, ...]:
        issues: list[ReproducibilityIssue] = []
        checks = (
            (
                not self.git.commit,
                ReproducibilityIssueCode.MISSING_GIT_REVISION,
                "git.commit",
                "an exact source commit is required",
            ),
            (
                not self.versions.config_digest,
                ReproducibilityIssueCode.MISSING_CONFIG_DIGEST,
                "versions.config_digest",
                "resolved configuration digest is required",
            ),
            (
                not self.model.revision,
                ReproducibilityIssueCode.MISSING_MODEL_REVISION,
                "model.revision",
                "exact model revision is required",
            ),
            (
                not self.dataset.revision,
                ReproducibilityIssueCode.MISSING_DATASET_REVISION,
                "dataset.revision",
                "exact dataset revision is required",
            ),
            (
                not self.dataset.manifest_id,
                ReproducibilityIssueCode.MISSING_DATASET_MANIFEST,
                "dataset.manifest_id",
                "calibration dataset manifest identity is required",
            ),
            (
                not self.dataset.tokenizer_revision,
                ReproducibilityIssueCode.MISSING_TOKENIZER_REVISION,
                "dataset.tokenizer_revision",
                "exact tokenizer revision is required",
            ),
            (
                not self.versions.tool_revision,
                ReproducibilityIssueCode.MISSING_TOOL_REVISION,
                "versions.tool_revision",
                "tool revision is required",
            ),
            (
                not self.versions.evaluator_version,
                ReproducibilityIssueCode.MISSING_EVALUATOR_REVISION,
                "versions.evaluator_version",
                "evaluator revision is required",
            ),
            (
                not self.lock.hexadecimal,
                ReproducibilityIssueCode.MISSING_LOCK_DIGEST,
                "lock.hexadecimal",
                "dependency lock digest is required",
            ),
        )
        for failed, code, path, detail in checks:
            if failed:
                issues.append(ReproducibilityIssue(code, path, detail))
        if not self.git.clean:
            issues.append(
                ReproducibilityIssue(
                    ReproducibilityIssueCode.DIRTY_GIT_WORKTREE,
                    "git.clean",
                    "uncommitted source changes prevent exact reconstruction from the commit",
                )
            )
        return tuple(sorted(issues, key=lambda item: (item.code.value, item.path)))

    @property
    def reproducible(self) -> bool:
        return not self.issues

    def require_reproducible(self) -> None:
        if self.reproducible:
            return
        first = self.issues[0]
        raise ReproducibilityError(
            f"run is not reproducible: {first.code.value}: {first.path}: {first.detail}"
        )

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "attempt_id": self.attempt_id,
            "git": self.git.to_record(),
            "model": self.model.to_record(),
            "dataset": self.dataset.to_record(),
            "versions": self.versions.to_record(),
            "seeds": self.seeds.to_record(),
            "hardware": self.hardware.to_record(),
            "lock": self.lock.to_record(),
            "resolved_config": canonicalize_identity_value(self.resolved_config),
            "command": list(self.command),
            "metric_tolerances": [item.to_record() for item in self.metric_tolerances],
        }

    @property
    def manifest_id(self) -> str:
        payload = canonical_identity_json(self._identity_record()).encode("utf-8")
        return f"repro_{hashlib.sha256(payload).hexdigest()}"

    def to_record(self) -> dict[str, object]:
        return {
            **self._identity_record(),
            "manifest_id": self.manifest_id,
            "reproducible": self.reproducible,
            "issues": [item.to_record() for item in self.issues],
        }

    def canonical_json(self) -> str:
        return canonical_identity_json(self.to_record())


def capture_reproducibility_manifest(
    record: ExperimentRecord,
    *,
    git: GitRevision,
    lock: LockDigest,
    resolved_config: Mapping[str, object],
    command: tuple[str, ...],
    metric_tolerances: tuple[MetricTolerance, ...] = (),
) -> ReproducibilityManifest:
    """Freeze all reproducibility evidence already associated with one run."""

    measured_metrics = {
        f"{phase}:{metric.name}"
        for phase, metrics in (
            ("baseline", record.baseline_metrics),
            ("post", record.post_metrics),
            ("delta", record.delta_metrics),
        )
        for metric in metrics
        if metric.value is not None
    }
    unknown_tolerances = {
        item.metric for item in metric_tolerances
    } - measured_metrics
    if unknown_tolerances:
        raise ReproducibilityError(
            "metric tolerances reference unmeasured metrics: "
            f"{sorted(unknown_tolerances)}"
        )
    canonical_config = cast(
        dict[str, object],
        canonicalize_identity_value(resolved_config),
    )

    return ReproducibilityManifest(
        record.run_id,
        record.experiment_id,
        record.attempt_id,
        git,
        record.model,
        record.dataset,
        record.versions,
        record.seeds,
        record.hardware,
        lock,
        canonical_config,
        command,
        metric_tolerances,
    )


def digest_lock_file(path: str | Path, *, name: str | None = None) -> LockDigest:
    """Stream a dependency lock file into its canonical SHA-256 identity."""

    resolved = Path(path)
    if not resolved.is_file() or resolved.is_symlink():
        raise ReproducibilityError("dependency lock path must be a regular file")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return LockDigest(name or resolved.name, digest.hexdigest())


def collect_git_revision(repository_root: str | Path = ".") -> GitRevision:
    """Collect the exact repository commit and whether tracked/untracked changes exist."""

    root = Path(repository_root).resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip().lower()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ReproducibilityError("failed to collect git revision evidence") from error
    return GitRevision(revision, clean=not bool(status.strip()))


def publish_reproducibility_manifest(
    manifest: ReproducibilityManifest,
    persisted: PersistedExperiment,
    *,
    artifact_store: ContentAddressedArtifactStore,
    metadata_store: ExperimentMetadataStore,
) -> PublishedArtifactReference:
    """Publish one immutable manifest and link it to the persisted run candidate."""

    if persisted.run_id != manifest.run_id:
        raise ReproducibilityError("persisted run does not match reproducibility manifest")
    candidate = metadata_store.get_candidate(persisted.candidate_id)
    if candidate is None or candidate.run_id != manifest.run_id:
        raise ReproducibilityError("reproducibility manifest requires its persisted run candidate")
    payload = (manifest.canonical_json() + "\n").encode("utf-8")
    return artifact_store.publish_for_candidate(
        metadata_store,
        persisted.candidate_id,
        role=REPRODUCIBILITY_ARTIFACT_ROLE,
        payload=payload,
        reference_metadata={
            "manifest_id": manifest.manifest_id,
            "run_id": manifest.run_id,
            "experiment_id": manifest.experiment_id,
            "reproducible": manifest.reproducible,
            "schema_version": manifest.schema_version,
        },
    )


def load_reproducibility_manifest(payload: str) -> dict[str, object]:
    """Parse a stored manifest payload for integrity-oriented consumers."""

    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ReproducibilityError("reproducibility manifest is not valid JSON") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ReproducibilityError("reproducibility manifest root must be an object")
    expected = {
        "schema_version",
        "run_id",
        "experiment_id",
        "attempt_id",
        "git",
        "model",
        "dataset",
        "versions",
        "seeds",
        "hardware",
        "lock",
        "resolved_config",
        "command",
        "metric_tolerances",
        "manifest_id",
        "reproducible",
        "issues",
    }
    if set(value) != expected:
        raise ReproducibilityError(
            "reproducibility manifest has missing or unknown fields"
        )
    if value["schema_version"] != REPRODUCIBILITY_MANIFEST_SCHEMA_VERSION:
        raise ReproducibilityError("reproducibility manifest schema is unsupported")
    manifest_id = value["manifest_id"]
    if not isinstance(manifest_id, str) or not manifest_id.startswith("repro_"):
        raise ReproducibilityError("reproducibility manifest identity is invalid")
    identity = {key: value[key] for key in expected - {"manifest_id", "reproducible", "issues"}}
    expected_id = "repro_" + hashlib.sha256(
        canonical_identity_json(identity).encode("utf-8")
    ).hexdigest()
    if manifest_id != expected_id:
        raise ReproducibilityError("reproducibility manifest identity does not match content")
    return value

"""Locked replay environments, deterministic cursors, and schema migration evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final

REPLAY_ENVIRONMENT_SCHEMA_VERSION: Final[int] = 1


class ReplayEnvironmentError(ValueError):
    """Raised when a replay recipe or environment cannot be verified safely."""


class ReplayEnvironmentKind(StrEnum):
    NATIVE = "native"
    DOCKER = "docker"
    NIX_EXPLORATORY = "nix_exploratory"


class ReplayStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayEnvironmentError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ReplayEnvironmentError(f"{label} must be a lowercase SHA-256")
    return result


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReplayEnvironmentError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ReplayEnvironmentLock:
    kind: ReplayEnvironmentKind
    identifier: str
    definition_digest: str
    toolchain_revision: str
    architecture: str
    hardware_class: str
    gpu_passthrough: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("environment identifier", self.identifier),
            ("toolchain revision", self.toolchain_revision),
            ("architecture", self.architecture),
            ("hardware class", self.hardware_class),
            ("GPU passthrough mode", self.gpu_passthrough),
        ):
            _text(value, label)
        _digest(self.definition_digest, "environment definition digest")
        if self.capabilities != tuple(sorted(set(self.capabilities))) or any(
            not capability.strip() for capability in self.capabilities
        ):
            raise ReplayEnvironmentError("environment capabilities must be canonical")
        if self.kind is ReplayEnvironmentKind.DOCKER and self.gpu_passthrough == "none":
            raise ReplayEnvironmentError("Docker replay must declare its GPU passthrough boundary")

    @property
    def lock_id(self) -> str:
        payload = {
            "schema_version": REPLAY_ENVIRONMENT_SCHEMA_VERSION,
            "kind": self.kind.value,
            "identifier": self.identifier,
            "definition_digest": self.definition_digest,
            "toolchain_revision": self.toolchain_revision,
            "architecture": self.architecture,
            "hardware_class": self.hardware_class,
            "gpu_passthrough": self.gpu_passthrough,
            "capabilities": list(self.capabilities),
        }
        return "environment_lock_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": REPLAY_ENVIRONMENT_SCHEMA_VERSION,
            "lock_id": self.lock_id,
            "kind": self.kind.value,
            "identifier": self.identifier,
            "definition_digest": self.definition_digest,
            "toolchain_revision": self.toolchain_revision,
            "architecture": self.architecture,
            "hardware_class": self.hardware_class,
            "gpu_passthrough": self.gpu_passthrough,
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True, slots=True)
class ReplayEnvironmentObservation:
    kind: ReplayEnvironmentKind
    identifier: str
    definition_digest: str
    toolchain_revision: str
    architecture: str
    hardware_class: str
    gpu_passthrough: str
    capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        _text(self.identifier, "observed environment identifier")
        _digest(self.definition_digest, "observed environment definition digest")
        for label, value in (
            ("observed toolchain revision", self.toolchain_revision),
            ("observed architecture", self.architecture),
            ("observed hardware class", self.hardware_class),
            ("observed GPU passthrough", self.gpu_passthrough),
        ):
            _text(value, label)
        if self.capabilities != tuple(sorted(set(self.capabilities))):
            raise ReplayEnvironmentError("observed capabilities must be canonical")


@dataclass(frozen=True, slots=True)
class ReplayMismatch:
    code: str
    path: str
    expected: object
    actual: object
    blocking: bool

    def __post_init__(self) -> None:
        _text(self.code, "replay mismatch code")
        _text(self.path, "replay mismatch path")

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code,
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
            "blocking": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class ReplayMetricTolerance:
    metric: str
    absolute: float
    relative: float

    def __post_init__(self) -> None:
        _text(self.metric, "replay metric")
        for label, value in (
            ("absolute tolerance", self.absolute),
            ("relative tolerance", self.relative),
        ):
            if not math.isfinite(value) or value < 0:
                raise ReplayEnvironmentError(f"{label} must be finite and non-negative")

    def to_record(self) -> dict[str, object]:
        return {"metric": self.metric, "absolute": self.absolute, "relative": self.relative}


@dataclass(frozen=True, slots=True)
class ReplayRecipe:
    bundle_id: str
    environment_lock_id: str
    command: tuple[str, ...]
    seeds: tuple[int, ...]
    input_digests: tuple[str, ...]
    schema_versions: tuple[tuple[str, int], ...]
    metric_tolerances: tuple[ReplayMetricTolerance, ...]

    def __post_init__(self) -> None:
        _text(self.bundle_id, "replay bundle ID")
        _text(self.environment_lock_id, "environment lock ID")
        if not self.command or any(not item.strip() for item in self.command):
            raise ReplayEnvironmentError("replay recipes require an exact command")
        if not self.seeds or self.seeds != tuple(sorted(set(self.seeds))):
            raise ReplayEnvironmentError("replay seeds must be unique and sorted")
        if any(isinstance(seed, bool) or seed < 0 for seed in self.seeds):
            raise ReplayEnvironmentError("replay seeds must be unsigned")
        if not self.input_digests or self.input_digests != tuple(sorted(set(self.input_digests))):
            raise ReplayEnvironmentError("replay inputs must be canonical digests")
        for digest in self.input_digests:
            _digest(digest, "replay input digest")
        names = tuple(name for name, _ in self.schema_versions)
        if names != tuple(sorted(set(names))) or any(
            version <= 0 for _, version in self.schema_versions
        ):
            raise ReplayEnvironmentError("replay schema versions must be canonical")
        tolerance_names = tuple(item.metric for item in self.metric_tolerances)
        if tolerance_names != tuple(sorted(set(tolerance_names))):
            raise ReplayEnvironmentError("replay tolerances must be canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "environment_lock_id": self.environment_lock_id,
            "command": list(self.command),
            "seeds": list(self.seeds),
            "input_digests": list(self.input_digests),
            "schema_versions": {name: version for name, version in self.schema_versions},
            "metric_tolerances": [item.to_record() for item in self.metric_tolerances],
        }


@dataclass(frozen=True, slots=True)
class ReplayCursor:
    steps: tuple[str, ...]
    completed: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.steps or self.steps != tuple(dict.fromkeys(self.steps)):
            raise ReplayEnvironmentError("replay cursor steps must be non-empty and unique")
        if any(step not in self.steps for step in self.completed):
            raise ReplayEnvironmentError("replay cursor completed step is unknown")
        if self.completed != self.steps[: len(self.completed)]:
            raise ReplayEnvironmentError("replay cursor completion must be an ordered prefix")

    @property
    def next_step(self) -> str | None:
        return self.steps[len(self.completed)] if len(self.completed) < len(self.steps) else None

    @property
    def cursor_id(self) -> str:
        payload = {"steps": list(self.steps), "completed": list(self.completed)}
        return "replay_cursor_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def advance(self, step: str) -> ReplayCursor:
        if step != self.next_step:
            raise ReplayEnvironmentError("replay cursor cannot skip or repeat a step")
        return replace(self, completed=(*self.completed, step))

    def to_record(self) -> dict[str, object]:
        return {
            "cursor_id": self.cursor_id,
            "steps": list(self.steps),
            "completed": list(self.completed),
            "next_step": self.next_step,
        }


@dataclass(frozen=True, slots=True)
class ReplayMigrationCopy:
    original_bundle_digest: str
    input_schema_version: int
    output_schema_version: int
    input_digest: str
    output_digest: str
    migration_id: str

    def __post_init__(self) -> None:
        _digest(self.original_bundle_digest, "original bundle digest")
        _digest(self.input_digest, "migration input digest")
        _digest(self.output_digest, "migration output digest")
        if (
            self.input_schema_version <= 0
            or self.output_schema_version <= self.input_schema_version
        ):
            raise ReplayEnvironmentError("migration versions must increase")
        _text(self.migration_id, "migration ID")

    def to_record(self) -> dict[str, object]:
        return {
            "original_bundle_digest": self.original_bundle_digest,
            "input_schema_version": self.input_schema_version,
            "output_schema_version": self.output_schema_version,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "migration_id": self.migration_id,
        }


@dataclass(frozen=True, slots=True)
class ReplayMigrationChain:
    copies: tuple[ReplayMigrationCopy, ...]

    def __post_init__(self) -> None:
        if not self.copies:
            raise ReplayEnvironmentError("migration chains require at least one copy")
        original = self.copies[0].original_bundle_digest
        for previous, current in zip(self.copies, self.copies[1:], strict=False):
            if current.original_bundle_digest != original:
                raise ReplayEnvironmentError("migration chain changed the original bundle digest")
            if current.input_schema_version != previous.output_schema_version:
                raise ReplayEnvironmentError("migration copies are not an ordered schema chain")
            if current.input_digest != previous.output_digest:
                raise ReplayEnvironmentError("migration copies do not retain the prior output hash")

    @property
    def final_digest(self) -> str:
        return self.copies[-1].output_digest

    def to_record(self) -> dict[str, object]:
        return {
            "copies": [item.to_record() for item in self.copies],
            "final_digest": self.final_digest,
        }


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    status: ReplayStatus
    recipe: ReplayRecipe
    environment_lock: ReplayEnvironmentLock
    cursor: ReplayCursor
    mismatches: tuple[ReplayMismatch, ...]
    migrations: ReplayMigrationChain | None = None
    reason: str | None = None

    @property
    def executable(self) -> bool:
        return self.status is ReplayStatus.READY and not self.mismatches

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": REPLAY_ENVIRONMENT_SCHEMA_VERSION,
            "status": self.status.value,
            "executable": self.executable,
            "recipe": self.recipe.to_record(),
            "environment_lock": self.environment_lock.to_record(),
            "cursor": self.cursor.to_record(),
            "mismatches": [item.to_record() for item in self.mismatches],
            "migrations": None if self.migrations is None else self.migrations.to_record(),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ReplayMetricComparison:
    metric: str
    expected: float | None
    actual: float | None
    allowed_delta: float | None
    passed: bool
    reason: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "expected": self.expected,
            "actual": self.actual,
            "allowed_delta": self.allowed_delta,
            "passed": self.passed,
            "reason": self.reason,
        }


def diagnose_replay_environment(
    expected: ReplayEnvironmentLock,
    actual: ReplayEnvironmentObservation,
) -> tuple[ReplayMismatch, ...]:
    """Diagnose lock and hardware differences before any expensive execution."""

    mismatches: list[ReplayMismatch] = []
    checks = (
        ("kind", expected.kind.value, actual.kind.value, True),
        ("identifier", expected.identifier, actual.identifier, True),
        ("definition_digest", expected.definition_digest, actual.definition_digest, True),
        ("toolchain_revision", expected.toolchain_revision, actual.toolchain_revision, True),
        ("architecture", expected.architecture, actual.architecture, True),
        ("hardware_class", expected.hardware_class, actual.hardware_class, True),
        ("gpu_passthrough", expected.gpu_passthrough, actual.gpu_passthrough, True),
    )
    for path, expected_value, actual_value, blocking in checks:
        if expected_value != actual_value:
            mismatches.append(
                ReplayMismatch("environment_mismatch", path, expected_value, actual_value, blocking)
            )
    missing = tuple(sorted(set(expected.capabilities) - set(actual.capabilities)))
    if missing:
        mismatches.append(
            ReplayMismatch(
                "missing_capability",
                "capabilities",
                list(expected.capabilities),
                list(actual.capabilities),
                True,
            )
        )
    return tuple(mismatches)


def prepare_replay_plan(
    recipe: ReplayRecipe,
    expected_environment: ReplayEnvironmentLock,
    actual_environment: ReplayEnvironmentObservation,
    cursor: ReplayCursor,
    migrations: ReplayMigrationChain | None = None,
) -> ReplayPlan:
    """Build a deterministic preflight plan and block mismatched environments."""

    mismatches = diagnose_replay_environment(expected_environment, actual_environment)
    status = ReplayStatus.BLOCKED if mismatches else ReplayStatus.READY
    reason = "environment mismatch blocks execution" if mismatches else None
    return ReplayPlan(status, recipe, expected_environment, cursor, mismatches, migrations, reason)


def compare_replay_metrics(
    expected: dict[str, float],
    actual: dict[str, float],
    tolerances: tuple[ReplayMetricTolerance, ...],
) -> tuple[ReplayMetricComparison, ...]:
    """Compare exact replay metrics with declared absolute/relative tolerances."""

    tolerance_map = {item.metric: item for item in tolerances}
    comparisons: list[ReplayMetricComparison] = []
    for metric in sorted(set(expected) | set(actual)):
        if metric not in expected or metric not in actual:
            comparisons.append(
                ReplayMetricComparison(
                    metric,
                    expected.get(metric),
                    actual.get(metric),
                    None,
                    False,
                    "metric is missing",
                )
            )
            continue
        expected_value = expected[metric]
        actual_value = actual[metric]
        if not all(math.isfinite(value) for value in (expected_value, actual_value)):
            raise ReplayEnvironmentError("replay metrics must be finite")
        tolerance = tolerance_map.get(metric)
        if tolerance is None:
            comparisons.append(
                ReplayMetricComparison(
                    metric,
                    expected_value,
                    actual_value,
                    0.0,
                    expected_value == actual_value,
                    "no tolerance declared",
                )
            )
            continue
        allowed = tolerance.absolute + tolerance.relative * abs(expected_value)
        delta = abs(actual_value - expected_value)
        comparisons.append(
            ReplayMetricComparison(metric, expected_value, actual_value, allowed, delta <= allowed)
        )
    return tuple(comparisons)

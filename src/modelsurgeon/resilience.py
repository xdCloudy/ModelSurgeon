"""Bounded fault injection, subprocess termination, and recovery evidence."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

CRASH_CONSISTENCY_SCHEMA_VERSION: Final[int] = 1


class ResilienceError(ValueError):
    """Raised when a fault scenario or recovery invariant is unsafe."""


class FaultPoint(StrEnum):
    SOURCE = "source"
    STAGING = "staging"
    CHECKSUM = "checksum"
    FSYNC = "fsync"
    RENAME = "rename"
    DISK_FULL = "disk_full"
    OOM = "oom"
    CANCELLATION = "cancellation"
    PROCESS_DEATH = "process_death"
    TIMEOUT = "timeout"
    DEADLOCK = "deadlock"


class RecoveryOutcome(StrEnum):
    RECOVERED = "recovered"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


FAULT_RECOVERY_ACTIONS: Final[dict[FaultPoint, str]] = {
    FaultPoint.SOURCE: "reject_without_publication",
    FaultPoint.STAGING: "remove_owned_staging_and_preserve_source",
    FaultPoint.CHECKSUM: "remove_owned_staging_and_preserve_source",
    FaultPoint.FSYNC: "remove_owned_staging_and_preserve_source",
    FaultPoint.RENAME: "revalidate_destination_then_resume_or_reject",
    FaultPoint.DISK_FULL: "remove_owned_staging_and_resume_after_capacity_check",
    FaultPoint.OOM: "remove_owned_staging_and_retain_failure",
    FaultPoint.CANCELLATION: "rollback_owned_transaction",
    FaultPoint.PROCESS_DEATH: "recover_owned_staging_on_restart",
    FaultPoint.TIMEOUT: "terminate_child_tree_then_recover_owned_staging",
    FaultPoint.DEADLOCK: "terminate_child_tree_then_recover_owned_staging",
}


@dataclass(frozen=True, slots=True)
class FaultScenario:
    seed: int
    point: FaultPoint
    grace_seconds: float = 0.5
    max_log_bytes: int = 64 * 1024
    owned_staging_prefix: str = ".modelsurgeon."

    def __post_init__(self) -> None:
        if self.seed < 0 or self.grace_seconds <= 0 or self.max_log_bytes <= 0:
            raise ResilienceError("fault scenario limits must be positive")
        if (
            not self.owned_staging_prefix
            or "/" in self.owned_staging_prefix
            or "\\" in self.owned_staging_prefix
        ):
            raise ResilienceError("staging ownership prefix must be a path component")

    @property
    def scenario_id(self) -> str:
        payload = {
            "seed": self.seed,
            "point": self.point.value,
            "action": FAULT_RECOVERY_ACTIONS[self.point],
        }
        return "fault_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "point": self.point.value,
            "recovery_action": FAULT_RECOVERY_ACTIONS[self.point],
            "grace_seconds": self.grace_seconds,
            "max_log_bytes": self.max_log_bytes,
            "owned_staging_prefix": self.owned_staging_prefix,
        }


@dataclass(frozen=True, slots=True)
class RecoveryEvidence:
    scenario_id: str
    outcome: RecoveryOutcome
    source_unchanged: bool
    accepted_unchanged: bool
    removed_paths: tuple[str, ...]
    logs: str
    elapsed_seconds: float
    detail: str

    def __post_init__(self) -> None:
        if not self.scenario_id or self.elapsed_seconds < 0:
            raise ResilienceError("recovery evidence identity and elapsed time are required")
        if len(self.logs.encode("utf-8")) > 64 * 1024:
            raise ResilienceError("recovery logs exceed the evidence limit")
        if self.outcome is RecoveryOutcome.RECOVERED and (
            not self.source_unchanged or not self.accepted_unchanged
        ):
            raise ResilienceError("recovered evidence must preserve source and accepted artifacts")

    def to_record(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "outcome": self.outcome.value,
            "source_unchanged": self.source_unchanged,
            "accepted_unchanged": self.accepted_unchanged,
            "removed_paths": list(self.removed_paths),
            "logs": self.logs,
            "elapsed_seconds": self.elapsed_seconds,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class FaultMatrixReport:
    scenarios: tuple[FaultScenario, ...]
    evidence: tuple[RecoveryEvidence, ...]
    schema_version: int = CRASH_CONSISTENCY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CRASH_CONSISTENCY_SCHEMA_VERSION:
            raise ResilienceError("unsupported crash consistency schema version")
        if {scenario.scenario_id for scenario in self.scenarios} != {
            item.scenario_id for item in self.evidence
        }:
            raise ResilienceError("fault matrix scenarios and evidence do not reconcile")

    @property
    def outcome(self) -> RecoveryOutcome:
        if any(item.outcome is RecoveryOutcome.UNKNOWN for item in self.evidence):
            return RecoveryOutcome.UNKNOWN
        if any(item.outcome is RecoveryOutcome.REJECTED for item in self.evidence):
            return RecoveryOutcome.REJECTED
        if any(item.outcome is RecoveryOutcome.UNSUPPORTED for item in self.evidence):
            return RecoveryOutcome.UNSUPPORTED
        return RecoveryOutcome.RECOVERED

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "outcome": self.outcome.value,
            "scenarios": [item.to_record() for item in self.scenarios],
            "evidence": [item.to_record() for item in self.evidence],
        }


def generate_fault_matrix(*, seed: int = 408) -> tuple[FaultScenario, ...]:
    """Generate one deterministic scenario for every documented fault point."""
    return tuple(FaultScenario(seed + index, point) for index, point in enumerate(FaultPoint))


def bounded_subprocess(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    grace_seconds: float = 0.5,
    max_log_bytes: int = 64 * 1024,
    cwd: str | Path | None = None,
) -> RecoveryEvidence:
    """Run an argument-array child and terminate its process tree on timeout."""
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise ResilienceError("subprocess command must contain non-empty argument strings")
    if timeout_seconds <= 0 or grace_seconds <= 0 or max_log_bytes <= 0:
        raise ResilienceError("subprocess bounds must be positive")
    started = time.perf_counter()
    process_group_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        list(command),
        cwd=None if cwd is None else Path(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=False,
        start_new_session=os.name != "nt",
        creationflags=process_group_flag if os.name == "nt" else 0,
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
        clipped = output[:max_log_bytes]
        outcome = RecoveryOutcome.RECOVERED if process.returncode == 0 else RecoveryOutcome.REJECTED
        detail = f"returncode={process.returncode}"
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            getpgid = getattr(os, "getpgid", None)
            killpg = getattr(os, "killpg", None)
            if getpgid is None or killpg is None:
                process.terminate()
            else:
                killpg(getpgid(process.pid), signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
        clipped = output[:max_log_bytes]
        outcome = RecoveryOutcome.UNKNOWN
        detail = f"child tree timed out after {timeout_seconds}s and was terminated"
    elapsed = time.perf_counter() - started
    return RecoveryEvidence(
        "subprocess",
        outcome,
        True,
        True,
        (),
        clipped.decode("utf-8", errors="replace"),
        elapsed,
        detail,
    )


def verify_recovery_invariants(
    before: dict[str, str],
    after: dict[str, str],
    *,
    source_paths: Sequence[str],
    accepted_paths: Sequence[str],
    owned_staging_prefix: str,
) -> tuple[bool, bool, tuple[str, ...]]:
    """Verify that only owned staging siblings were removed after a crash."""
    if not owned_staging_prefix or "/" in owned_staging_prefix or "\\" in owned_staging_prefix:
        raise ResilienceError("staging ownership prefix must be a path component")
    source_unchanged = all(before.get(path) == after.get(path) for path in source_paths)
    accepted_unchanged = all(before.get(path) == after.get(path) for path in accepted_paths)
    removed = tuple(sorted(set(before) - set(after)))
    unexpected = tuple(
        path for path in removed if not Path(path).name.startswith(owned_staging_prefix)
    )
    if unexpected:
        raise ResilienceError(f"recovery removed paths outside owned staging: {unexpected}")
    return source_unchanged, accepted_unchanged, removed

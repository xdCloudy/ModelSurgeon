"""Typed, bounded scheduling for local and explicitly registered workers."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.experiments.hardware import HardwareInventory

WORKER_SCHEDULER_SCHEMA_VERSION = 1


class WorkerSchedulerError(ValueError):
    """Raised when a worker, task, lease, or publication is unsafe."""


class WorkerOperation(StrEnum):
    EVALUATE = "evaluate"
    MUTATE = "mutate"
    PROFILE = "profile"
    PACKAGE = "package"


class WorkerTransport(StrEnum):
    LOCAL = "local"
    EXPLICIT_REMOTE = "explicit_remote"


class WorkerTaskOutcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class WorkerTaskState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    EXHAUSTED = "exhausted"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerSchedulerError(f"{label} must be non-empty text")
    return value


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if not text.startswith("sha256:") or len(text) != len("sha256:") + 64:
        raise WorkerSchedulerError(f"{label} must be a sha256 content digest")
    try:
        int(text[7:], 16)
    except ValueError as error:
        raise WorkerSchedulerError(f"{label} must be a sha256 content digest") from error
    return text


def _nonnegative(value: int, label: str) -> None:
    if isinstance(value, bool) or value < 0:
        raise WorkerSchedulerError(f"{label} must be a non-negative integer")


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise WorkerSchedulerError(f"{label} must be a positive integer")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class WorkerResourceCapacity:
    """Advertised capacity used for placement; unknown capacity is never guessed."""

    ram_bytes: int
    vram_bytes: int
    disk_bytes: int
    cpu_cores: int
    gpu_count: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.ram_bytes, "RAM capacity"),
            (self.vram_bytes, "VRAM capacity"),
            (self.disk_bytes, "disk capacity"),
            (self.cpu_cores, "CPU capacity"),
            (self.gpu_count, "GPU capacity"),
        ):
            _nonnegative(value, label)
        if self.cpu_cores == 0:
            raise WorkerSchedulerError("worker must advertise at least one CPU core")
        if self.gpu_count == 0 and self.vram_bytes != 0:
            raise WorkerSchedulerError("VRAM requires at least one GPU")

    def to_record(self) -> dict[str, int]:
        return {
            "ram_bytes": self.ram_bytes,
            "vram_bytes": self.vram_bytes,
            "disk_bytes": self.disk_bytes,
            "cpu_cores": self.cpu_cores,
            "gpu_count": self.gpu_count,
        }


@dataclass(frozen=True, slots=True)
class WorkerCapabilityProfile:
    """Versioned worker capabilities and immutable hardware lineage."""

    worker_id: str
    transport: WorkerTransport
    operations: tuple[WorkerOperation, ...]
    capacity: WorkerResourceCapacity
    hardware: HardwareInventory
    worker_revision: str
    max_concurrency: int = 1
    schema_version: int = WORKER_SCHEDULER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.worker_id, "worker ID")
        _text(self.worker_revision, "worker revision")
        if self.schema_version != WORKER_SCHEDULER_SCHEMA_VERSION:
            raise WorkerSchedulerError("unsupported worker profile schema")
        _positive(self.max_concurrency, "worker concurrency")
        if not self.operations:
            raise WorkerSchedulerError("worker must advertise at least one operation")
        if len(set(self.operations)) != len(self.operations):
            raise WorkerSchedulerError("worker operations must be unique")
        if tuple(sorted(self.operations, key=lambda item: item.value)) != self.operations:
            raise WorkerSchedulerError("worker operations must be canonically ordered")

    @property
    def profile_id(self) -> str:
        digest = hashlib.sha256(_canonical(self.to_record()).encode()).hexdigest()
        return f"worker_profile_{digest}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "worker_id": self.worker_id,
            "transport": self.transport.value,
            "operations": [item.value for item in self.operations],
            "capacity": self.capacity.to_record(),
            "hardware": self.hardware.to_record(),
            "worker_revision": self.worker_revision,
            "max_concurrency": self.max_concurrency,
        }


@dataclass(frozen=True, slots=True)
class WorkerRegistration:
    profile: WorkerCapabilityProfile
    credential_digest: str
    registered_at_ns: int

    def __post_init__(self) -> None:
        _digest(self.credential_digest, "worker credential digest")
        _nonnegative(self.registered_at_ns, "registration timestamp")

    def to_record(self) -> dict[str, object]:
        return {
            "profile": self.profile.to_record(),
            "profile_id": self.profile.profile_id,
            "credential_digest": self.credential_digest,
            "registered_at_ns": self.registered_at_ns,
        }


@dataclass(frozen=True, slots=True)
class ContentAddressedInput:
    digest: str
    size_bytes: int
    role: str

    def __post_init__(self) -> None:
        _digest(self.digest, "input digest")
        _nonnegative(self.size_bytes, "input size")
        _text(self.role, "input role")

    def to_record(self) -> dict[str, object]:
        return {"digest": self.digest, "size_bytes": self.size_bytes, "role": self.role}


@dataclass(frozen=True, slots=True)
class WorkerTaskSpec:
    """A typed operation; no shell command or arbitrary callable crosses this boundary."""

    campaign_id: str
    operation: WorkerOperation
    inputs: tuple[ContentAddressedInput, ...]
    ram_bytes: int
    vram_bytes: int
    disk_bytes: int
    cpu_cores: int
    priority: int = 0
    seed: int = 0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        _text(self.campaign_id, "campaign ID")
        if not isinstance(self.operation, WorkerOperation):
            raise WorkerSchedulerError("task operation must be allowlisted")
        if not self.inputs:
            raise WorkerSchedulerError("tasks require at least one content-addressed input")
        if len({item.digest for item in self.inputs}) != len(self.inputs):
            raise WorkerSchedulerError("task inputs must have unique content digests")
        for value, label in (
            (self.ram_bytes, "task RAM"),
            (self.vram_bytes, "task VRAM"),
            (self.disk_bytes, "task disk"),
        ):
            _nonnegative(value, label)
        _positive(self.cpu_cores, "task CPU")
        if self.priority < 0:
            raise WorkerSchedulerError("task priority must be non-negative")
        _nonnegative(self.seed, "task seed")
        _positive(self.max_attempts, "task attempts")

    @property
    def task_id(self) -> str:
        digest = hashlib.sha256(_canonical(self.to_record()).encode()).hexdigest()
        return f"worker_task_{digest}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": WORKER_SCHEDULER_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "operation": self.operation.value,
            "inputs": [item.to_record() for item in self.inputs],
            "ram_bytes": self.ram_bytes,
            "vram_bytes": self.vram_bytes,
            "disk_bytes": self.disk_bytes,
            "cpu_cores": self.cpu_cores,
            "priority": self.priority,
            "seed": self.seed,
            "max_attempts": self.max_attempts,
        }


@dataclass(frozen=True, slots=True)
class CampaignBudget:
    max_tasks: int
    max_concurrent_tasks: int

    def __post_init__(self) -> None:
        _positive(self.max_tasks, "campaign task budget")
        _positive(self.max_concurrent_tasks, "campaign concurrency budget")


@dataclass(frozen=True, slots=True)
class WorkerLease:
    task_id: str
    worker_id: str
    lease_token: str
    attempt: int
    acquired_at_ns: int
    heartbeat_at_ns: int
    expires_at_ns: int

    def expired(self, now_ns: int) -> bool:
        return self.expires_at_ns <= now_ns


@dataclass(frozen=True, slots=True)
class WorkerTaskResult:
    task_id: str
    outcome: WorkerTaskOutcome
    worker_id: str
    output_digest: str | None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.task_id, "result task ID")
        _text(self.worker_id, "result worker ID")
        if self.outcome is WorkerTaskOutcome.SUPPORTED and self.output_digest is None:
            raise WorkerSchedulerError("supported results require a content-addressed output")
        if self.output_digest is not None:
            _digest(self.output_digest, "output digest")
        if self.evidence != tuple(sorted(set(self.evidence))):
            raise WorkerSchedulerError("result evidence must be sorted and unique")


@dataclass(frozen=True, slots=True)
class WorkerAssignment:
    task_id: str
    worker_id: str
    lease: WorkerLease
    task: WorkerTaskSpec


@dataclass(frozen=True, slots=True)
class WorkerTaskSnapshot:
    task: WorkerTaskSpec
    state: WorkerTaskState
    attempts: int
    assignment: WorkerAssignment | None
    result: WorkerTaskResult | None


@dataclass
class _TaskEntry:
    task: WorkerTaskSpec
    state: WorkerTaskState = WorkerTaskState.QUEUED
    attempts: int = 0
    assignment: WorkerAssignment | None = None
    result: WorkerTaskResult | None = None


class WorkerScheduler:
    """Deterministic in-process scheduler for local and explicitly registered workers."""

    def __init__(
        self,
        *,
        lease_duration_ns: int,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        _positive(lease_duration_ns, "lease duration")
        self.lease_duration_ns = lease_duration_ns
        self._token_factory = token_factory or (lambda: uuid.uuid4().hex)
        self._workers: dict[str, WorkerRegistration] = {}
        self._tasks: dict[str, _TaskEntry] = {}
        self._budgets: dict[str, CampaignBudget] = {}
        self._submitted: dict[str, int] = {}
        self._active_by_campaign: dict[str, int] = {}

    def register(
        self,
        profile: WorkerCapabilityProfile,
        *,
        credential: str,
        now_ns: int,
    ) -> WorkerRegistration:
        _text(credential, "worker credential")
        _nonnegative(now_ns, "registration timestamp")
        credential_digest = "sha256:" + hashlib.sha256(credential.encode()).hexdigest()
        existing = self._workers.get(profile.worker_id)
        if existing is not None and existing.credential_digest != credential_digest:
            raise WorkerSchedulerError("worker registration credential mismatch")
        registration = WorkerRegistration(profile, credential_digest, now_ns)
        self._workers[profile.worker_id] = registration
        return registration

    def set_campaign_budget(self, campaign_id: str, budget: CampaignBudget) -> None:
        _text(campaign_id, "campaign ID")
        existing = self._budgets.get(campaign_id)
        if existing is not None and existing != budget:
            raise WorkerSchedulerError("campaign budget cannot change after registration")
        self._budgets[campaign_id] = budget

    def submit(self, task: WorkerTaskSpec) -> str:
        budget = self._budgets.get(task.campaign_id)
        if budget is None:
            raise WorkerSchedulerError("campaign budget must be registered before tasks")
        existing = self._tasks.get(task.task_id)
        if existing is not None:
            if existing.task != task:
                raise WorkerSchedulerError("task ID collision with different task content")
            return task.task_id
        submitted = self._submitted.get(task.campaign_id, 0)
        if submitted >= budget.max_tasks:
            raise WorkerSchedulerError("campaign task budget exceeded")
        self._tasks[task.task_id] = _TaskEntry(task)
        self._submitted[task.campaign_id] = submitted + 1
        return task.task_id

    def _available(self, profile: WorkerCapabilityProfile, task: WorkerTaskSpec) -> bool:
        if task.operation not in profile.operations:
            return False
        if task.ram_bytes > profile.capacity.ram_bytes:
            return False
        if task.vram_bytes > profile.capacity.vram_bytes:
            return False
        if task.disk_bytes > profile.capacity.disk_bytes:
            return False
        if task.cpu_cores > profile.capacity.cpu_cores:
            return False
        if task.vram_bytes > 0 and profile.capacity.gpu_count == 0:
            return False
        active_entries = tuple(
            entry
            for entry in self._tasks.values()
            if entry.assignment is not None
            and entry.assignment.worker_id == profile.worker_id
            and entry.state is WorkerTaskState.RUNNING
        )
        if len(active_entries) >= profile.max_concurrency:
            return False
        return all(
            sum(getattr(entry.task, field) for entry in active_entries) + getattr(task, field)
            <= getattr(profile.capacity, capacity_field)
            for field, capacity_field in (
                ("ram_bytes", "ram_bytes"),
                ("vram_bytes", "vram_bytes"),
                ("disk_bytes", "disk_bytes"),
                ("cpu_cores", "cpu_cores"),
            )
        )

    def _reap_expired(self, now_ns: int) -> None:
        for entry in self._tasks.values():
            assignment = entry.assignment
            if entry.state is not WorkerTaskState.RUNNING or assignment is None:
                continue
            if not assignment.lease.expired(now_ns):
                continue
            entry.assignment = None
            campaign = entry.task.campaign_id
            self._active_by_campaign[campaign] = max(
                0, self._active_by_campaign.get(campaign, 0) - 1
            )
            if entry.attempts >= entry.task.max_attempts:
                entry.state = WorkerTaskState.EXHAUSTED
            else:
                entry.state = WorkerTaskState.QUEUED

    def schedule(self, *, now_ns: int) -> tuple[WorkerAssignment, ...]:
        _nonnegative(now_ns, "schedule timestamp")
        self._reap_expired(now_ns)
        assignments: list[WorkerAssignment] = []
        queued = sorted(
            (entry for entry in self._tasks.values() if entry.state is WorkerTaskState.QUEUED),
            key=lambda entry: (-entry.task.priority, entry.task.task_id),
        )
        profiles = tuple(
            sorted(
                (registration.profile for registration in self._workers.values()),
                key=lambda item: item.worker_id,
            )
        )
        for entry in queued:
            budget = self._budgets[entry.task.campaign_id]
            if (
                self._active_by_campaign.get(entry.task.campaign_id, 0)
                >= budget.max_concurrent_tasks
            ):
                continue
            for profile in profiles:
                if not self._available(profile, entry.task):
                    continue
                token = self._token_factory()
                _text(token, "lease token")
                entry.attempts += 1
                lease = WorkerLease(
                    entry.task.task_id,
                    profile.worker_id,
                    token,
                    entry.attempts,
                    now_ns,
                    now_ns,
                    now_ns + self.lease_duration_ns,
                )
                assignment = WorkerAssignment(
                    entry.task.task_id, profile.worker_id, lease, entry.task
                )
                entry.assignment = assignment
                entry.state = WorkerTaskState.RUNNING
                self._active_by_campaign[entry.task.campaign_id] = (
                    self._active_by_campaign.get(entry.task.campaign_id, 0) + 1
                )
                assignments.append(assignment)
                break
        return tuple(assignments)

    def heartbeat(self, lease_token: str, *, now_ns: int) -> WorkerLease:
        _text(lease_token, "lease token")
        _nonnegative(now_ns, "heartbeat timestamp")
        entry = self._entry_for_token(lease_token)
        lease = entry.assignment.lease  # type: ignore[union-attr]
        if lease.expired(now_ns):
            raise WorkerSchedulerError("expired lease cannot heartbeat")
        if now_ns < lease.heartbeat_at_ns:
            raise WorkerSchedulerError("heartbeat timestamp moved backwards")
        renewed = WorkerLease(
            lease.task_id,
            lease.worker_id,
            lease.lease_token,
            lease.attempt,
            lease.acquired_at_ns,
            now_ns,
            now_ns + self.lease_duration_ns,
        )
        entry.assignment = WorkerAssignment(
            entry.task.task_id, lease.worker_id, renewed, entry.task
        )
        return renewed

    def complete(
        self, result: WorkerTaskResult, *, lease_token: str, now_ns: int
    ) -> WorkerTaskResult:
        _nonnegative(now_ns, "completion timestamp")
        entry = self._tasks.get(result.task_id)
        if entry is None:
            raise WorkerSchedulerError("unknown task result")
        if entry.result is not None:
            if entry.result != result:
                raise WorkerSchedulerError("task result was already published")
            return entry.result
        assignment = entry.assignment
        if entry.state is not WorkerTaskState.RUNNING or assignment is None:
            raise WorkerSchedulerError("task has no active lease")
        if assignment.lease.lease_token != lease_token:
            raise WorkerSchedulerError("lease is not current")
        if assignment.lease.expired(now_ns):
            raise WorkerSchedulerError("expired lease cannot complete")
        if result.worker_id != assignment.worker_id:
            raise WorkerSchedulerError("result worker does not own lease")
        entry.result = result
        entry.assignment = None
        entry.state = WorkerTaskState.COMPLETED
        campaign = entry.task.campaign_id
        self._active_by_campaign[campaign] = max(0, self._active_by_campaign.get(campaign, 0) - 1)
        return result

    def _entry_for_token(self, lease_token: str) -> _TaskEntry:
        for entry in self._tasks.values():
            if entry.assignment is not None and entry.assignment.lease.lease_token == lease_token:
                return entry
        raise WorkerSchedulerError("lease is not current")

    def snapshot(self, task_id: str) -> WorkerTaskSnapshot:
        entry = self._tasks.get(task_id)
        if entry is None:
            raise WorkerSchedulerError("unknown task")
        return WorkerTaskSnapshot(
            entry.task, entry.state, entry.attempts, entry.assignment, entry.result
        )

    def registrations(self) -> tuple[WorkerRegistration, ...]:
        return tuple(self._workers[key] for key in sorted(self._workers))


__all__ = [
    "WORKER_SCHEDULER_SCHEMA_VERSION",
    "CampaignBudget",
    "ContentAddressedInput",
    "WorkerAssignment",
    "WorkerCapabilityProfile",
    "WorkerLease",
    "WorkerOperation",
    "WorkerRegistration",
    "WorkerResourceCapacity",
    "WorkerScheduler",
    "WorkerSchedulerError",
    "WorkerTaskOutcome",
    "WorkerTaskResult",
    "WorkerTaskSnapshot",
    "WorkerTaskSpec",
    "WorkerTaskState",
    "WorkerTransport",
]

"""Bounded, resumable recovery from CUDA and host out-of-memory failures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from modelsurgeon.experiments.gpu_cleanup import ExperimentGPUCleanup, GPUCleanupReport
from modelsurgeon.experiments.state_machine import (
    CandidateState,
    CandidateWorkStage,
    ExperimentStateError,
    ExperimentStateMachine,
)

OOM_RECOVERY_VERSION = "1"


class OOMRecoveryError(RuntimeError):
    """Raised when OOM recovery inputs or persisted state are unsafe."""


class OOMKind(StrEnum):
    CUDA = "cuda"
    HOST = "host"


class OOMAdaptationAction(StrEnum):
    REDUCE_BATCH = "reduce_batch"
    REDUCE_CHUNK = "reduce_chunk"
    ENABLE_OFFLOAD = "enable_offload"


@dataclass(frozen=True, slots=True)
class OOMAttemptConfig:
    batch_size: int
    chunk_size: int
    offload_enabled: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_size", self.batch_size),
            ("chunk_size", self.chunk_size),
        ):
            if isinstance(value, bool) or value <= 0:
                raise OOMRecoveryError(f"{name} must be a positive integer")

    def to_record(self) -> dict[str, int | bool]:
        return {
            "batch_size": self.batch_size,
            "chunk_size": self.chunk_size,
            "offload_enabled": self.offload_enabled,
        }


@dataclass(frozen=True, slots=True)
class OOMRetryPolicy:
    max_retries: int = 3
    min_batch_size: int = 1
    min_chunk_size: int = 1
    allow_offload: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.max_retries, bool) or self.max_retries < 0:
            raise OOMRecoveryError("max_retries must be a non-negative integer")
        for name, value in (
            ("min_batch_size", self.min_batch_size),
            ("min_chunk_size", self.min_chunk_size),
        ):
            if isinstance(value, bool) or value <= 0:
                raise OOMRecoveryError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class OOMRecoveryEvent:
    attempt: int
    kind: OOMKind
    exception_type: str
    action: OOMAdaptationAction | None
    before: OOMAttemptConfig
    after: OOMAttemptConfig | None

    def __post_init__(self) -> None:
        if self.attempt <= 0 or not self.exception_type:
            raise OOMRecoveryError("OOM recovery events require an attempt and exception type")
        if (self.action is None) != (self.after is None):
            raise OOMRecoveryError("OOM recovery action and adapted config must be recorded together")

    def to_record(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "kind": self.kind.value,
            "exception_type": self.exception_type,
            "action": None if self.action is None else self.action.value,
            "before": self.before.to_record(),
            "after": None if self.after is None else self.after.to_record(),
        }


@dataclass(frozen=True, slots=True)
class OOMRecoveryResult[T]:
    version: str
    succeeded: bool
    attempts: int
    value: T | None
    final_config: OOMAttemptConfig
    events: tuple[OOMRecoveryEvent, ...]
    cleanup_reports: tuple[GPUCleanupReport, ...]
    exhausted_kind: OOMKind | None = None

    def __post_init__(self) -> None:
        if self.version != OOM_RECOVERY_VERSION:
            raise OOMRecoveryError(f"unsupported OOM recovery version {self.version}")
        if self.attempts <= 0:
            raise OOMRecoveryError("OOM recovery must record at least one attempt")
        if self.succeeded == (self.exhausted_kind is not None):
            raise OOMRecoveryError("OOM recovery success/exhaustion fields are inconsistent")
        if len(self.cleanup_reports) != self.attempts:
            raise OOMRecoveryError("each OOM recovery attempt must record one cleanup report")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "succeeded": self.succeeded,
            "attempts": self.attempts,
            "final_config": self.final_config.to_record(),
            "events": [event.to_record() for event in self.events],
            "cleanup_reports": [report.to_record() for report in self.cleanup_reports],
            "exhausted_kind": (
                None if self.exhausted_kind is None else self.exhausted_kind.value
            ),
        }


class OOMClassifier(Protocol):
    def __call__(self, error: BaseException) -> OOMKind | None: ...


class CleanupFactory(Protocol):
    def __call__(self) -> ExperimentGPUCleanup: ...


def classify_oom(error: BaseException) -> OOMKind | None:
    """Classify known host/CUDA OOM signals without importing a framework."""

    if isinstance(error, MemoryError):
        return OOMKind.HOST
    exception_name = type(error).__name__.lower()
    module_name = type(error).__module__.lower()
    message = str(error).lower()
    cuda_markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
    )
    if any(marker in message for marker in cuda_markers):
        return OOMKind.CUDA
    if "outofmemory" in exception_name and ("torch" in module_name or "cuda" in message):
        return OOMKind.CUDA
    host_markers = (
        "cannot allocate memory",
        "failed to allocate memory",
        "memory allocation failed",
    )
    if any(marker in message for marker in host_markers):
        return OOMKind.HOST
    return None


def _halve(value: int, minimum: int) -> int:
    return max(minimum, (value + 1) // 2)


def _validate_policy_for_config(config: OOMAttemptConfig, policy: OOMRetryPolicy) -> None:
    if policy.min_batch_size > config.batch_size:
        raise OOMRecoveryError("min_batch_size cannot exceed the initial batch size")
    if policy.min_chunk_size > config.chunk_size:
        raise OOMRecoveryError("min_chunk_size cannot exceed the initial chunk size")


def adapt_oom_config(
    config: OOMAttemptConfig,
    policy: OOMRetryPolicy,
) -> tuple[OOMAdaptationAction, OOMAttemptConfig] | None:
    """Choose exactly one deterministic lower-memory adaptation."""

    _validate_policy_for_config(config, policy)
    if config.batch_size > policy.min_batch_size:
        adapted = OOMAttemptConfig(
            _halve(config.batch_size, policy.min_batch_size),
            config.chunk_size,
            config.offload_enabled,
        )
        return OOMAdaptationAction.REDUCE_BATCH, adapted
    if config.chunk_size > policy.min_chunk_size:
        adapted = OOMAttemptConfig(
            config.batch_size,
            _halve(config.chunk_size, policy.min_chunk_size),
            config.offload_enabled,
        )
        return OOMAdaptationAction.REDUCE_CHUNK, adapted
    if policy.allow_offload and not config.offload_enabled:
        return (
            OOMAdaptationAction.ENABLE_OFFLOAD,
            OOMAttemptConfig(config.batch_size, config.chunk_size, True),
        )
    return None


def _active_state(stage: CandidateWorkStage) -> CandidateState:
    if stage is CandidateWorkStage.MUTATION:
        return CandidateState.RUNNING
    return CandidateState.EVALUATING


def _detail(
    prefix: str,
    attempt: int,
    kind: OOMKind,
    config: OOMAttemptConfig,
    action: OOMAdaptationAction | None = None,
) -> str:
    parts = [
        prefix,
        f"attempt={attempt}",
        f"kind={kind.value}",
        f"batch={config.batch_size}",
        f"chunk={config.chunk_size}",
        f"offload={int(config.offload_enabled)}",
    ]
    if action is not None:
        parts.append(f"action={action.value}")
    return ":".join(parts)


def run_with_oom_recovery[T](
    state_machine: ExperimentStateMachine,
    candidate_id: str,
    stage: CandidateWorkStage,
    initial_config: OOMAttemptConfig,
    policy: OOMRetryPolicy,
    cleanup_factory: CleanupFactory,
    operation: Callable[[OOMAttemptConfig, ExperimentGPUCleanup], T],
    *,
    classifier: OOMClassifier = classify_oom,
    lease_heartbeat: Callable[[], object] | None = None,
) -> OOMRecoveryResult[T]:
    """Execute one candidate stage with bounded OOM adaptation and isolated exhaustion."""

    expected = _active_state(stage)
    current = state_machine.current(candidate_id)
    if current is not expected:
        actual = "<none>" if current is None else current.value
        raise ExperimentStateError(
            f"OOM recovery for {stage.value} requires {expected.value} state, found {actual}"
        )
    _validate_policy_for_config(initial_config, policy)

    config = initial_config
    events: list[OOMRecoveryEvent] = []
    cleanups: list[GPUCleanupReport] = []
    attempt = 0
    while True:
        attempt += 1
        if lease_heartbeat is not None:
            lease_heartbeat()
        boundary = cleanup_factory()
        try:
            with boundary:
                value = operation(config, boundary)
        except BaseException as error:
            report = boundary.last_report
            if report is None:
                raise OOMRecoveryError("attempt cleanup did not produce a report") from error
            cleanups.append(report)
            kind = classifier(error)
            if kind is None:
                raise

            state_machine.transition(
                candidate_id,
                CandidateState.RECOVERABLE_OOM,
                _detail("oom", attempt, kind, config),
            )
            adaptation = adapt_oom_config(config, policy)
            retry_allowed = len(events) < policy.max_retries
            if not retry_allowed or adaptation is None:
                events.append(
                    OOMRecoveryEvent(
                        attempt,
                        kind,
                        type(error).__name__,
                        None,
                        config,
                        None,
                    )
                )
                state_machine.transition(
                    candidate_id,
                    CandidateState.FAILED,
                    _detail("oom-exhausted", attempt, kind, config),
                )
                return OOMRecoveryResult(
                    OOM_RECOVERY_VERSION,
                    False,
                    attempt,
                    None,
                    config,
                    tuple(events),
                    tuple(cleanups),
                    kind,
                )

            action, adapted = adaptation
            events.append(
                OOMRecoveryEvent(
                    attempt,
                    kind,
                    type(error).__name__,
                    action,
                    config,
                    adapted,
                )
            )
            if lease_heartbeat is not None:
                lease_heartbeat()
            state_machine.transition(
                candidate_id,
                expected,
                _detail("oom-retry", attempt, kind, adapted, action),
            )
            config = adapted
            continue

        report = boundary.last_report
        if report is None:
            raise OOMRecoveryError("successful attempt cleanup did not produce a report")
        cleanups.append(report)
        return OOMRecoveryResult(
            OOM_RECOVERY_VERSION,
            True,
            attempt,
            value,
            config,
            tuple(events),
            tuple(cleanups),
        )

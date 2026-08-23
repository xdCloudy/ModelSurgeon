"""Atomic resumable state transitions for experiment mutation candidates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.experiments.store import (
    ExperimentMetadataStore,
    ExperimentStoreError,
    StoredStateEvent,
)


class ExperimentStateError(RuntimeError):
    """Raised when persisted candidate state is invalid or cannot transition safely."""


class CandidateState(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    EVALUATING = "evaluating"
    INTERRUPTED = "interrupted"
    RECOVERABLE_OOM = "recoverable-oom"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


class CandidateWorkStage(StrEnum):
    MUTATION = "mutation"
    EVALUATION = "evaluation"


@dataclass(frozen=True, slots=True)
class CandidateRecoveryPlan:
    candidate_id: str
    current_state: CandidateState
    next_stage: CandidateWorkStage | None
    resume: bool
    terminal: bool


_TERMINAL_STATES = frozenset(
    {
        CandidateState.SUCCEEDED,
        CandidateState.REJECTED,
        CandidateState.FAILED,
    }
)

_DIRECT_TRANSITIONS: dict[CandidateState, frozenset[CandidateState]] = {
    CandidateState.PLANNED: frozenset(
        {
            CandidateState.RUNNING,
            CandidateState.REJECTED,
            CandidateState.FAILED,
        }
    ),
    CandidateState.RUNNING: frozenset(
        {
            CandidateState.EVALUATING,
            CandidateState.INTERRUPTED,
            CandidateState.RECOVERABLE_OOM,
            CandidateState.REJECTED,
            CandidateState.FAILED,
        }
    ),
    CandidateState.EVALUATING: frozenset(
        {
            CandidateState.INTERRUPTED,
            CandidateState.RECOVERABLE_OOM,
            CandidateState.SUCCEEDED,
            CandidateState.REJECTED,
            CandidateState.FAILED,
        }
    ),
    CandidateState.INTERRUPTED: frozenset(
        {
            CandidateState.RUNNING,
            CandidateState.EVALUATING,
            CandidateState.FAILED,
        }
    ),
    CandidateState.RECOVERABLE_OOM: frozenset(
        {
            CandidateState.RUNNING,
            CandidateState.EVALUATING,
            CandidateState.FAILED,
        }
    ),
    CandidateState.SUCCEEDED: frozenset(),
    CandidateState.REJECTED: frozenset(),
    CandidateState.FAILED: frozenset(),
}


def _decode_state(value: str) -> CandidateState:
    try:
        return CandidateState(value)
    except ValueError as error:
        raise ExperimentStateError(f"unknown persisted candidate state {value!r}") from error


def _active_stage_before_recovery(states: tuple[CandidateState, ...]) -> CandidateWorkStage:
    for state in reversed(states):
        if state is CandidateState.EVALUATING:
            return CandidateWorkStage.EVALUATION
        if state is CandidateState.RUNNING:
            return CandidateWorkStage.MUTATION
        if state is CandidateState.PLANNED:
            return CandidateWorkStage.MUTATION
    raise ExperimentStateError("recovery state has no preceding active stage")


def _atomic_append(
    store: ExperimentMetadataStore,
    candidate_id: str,
    expected: CandidateState | None,
    target: CandidateState,
    detail: str | None,
) -> StoredStateEvent:
    if detail is not None and not detail:
        raise ExperimentStateError("state transition detail cannot be blank")
    try:
        with store._write() as connection:
            candidate = connection.execute(
                "SELECT 1 FROM experiment_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if candidate is None:
                raise ExperimentStateError(f"unknown experiment candidate {candidate_id}")
            row = connection.execute(
                """
                SELECT sequence, state
                FROM experiment_state_events
                WHERE candidate_id = ?
                ORDER BY sequence DESC
                LIMIT 1
                """,
                (candidate_id,),
            ).fetchone()
            current = None if row is None else _decode_state(str(row["state"]))
            if current is not expected:
                actual = "<none>" if current is None else current.value
                wanted = "<none>" if expected is None else expected.value
                raise ExperimentStateError(
                    f"candidate state changed concurrently: expected {wanted}, found {actual}"
                )
            sequence = 0 if row is None else int(row["sequence"]) + 1
            cursor = connection.execute(
                """
                INSERT INTO experiment_state_events(candidate_id, sequence, state, detail)
                VALUES (?, ?, ?, ?)
                """,
                (candidate_id, sequence, target.value, detail),
            )
            if cursor.lastrowid is None:
                raise ExperimentStateError("SQLite did not return a state event ID")
            return StoredStateEvent(
                int(cursor.lastrowid),
                candidate_id,
                sequence,
                target.value,
                detail,
            )
    except ExperimentStoreError as error:
        raise ExperimentStateError(str(error)) from error


class ExperimentStateMachine:
    """Validate state transitions and derive idempotent restart work from event history."""

    def __init__(self, store: ExperimentMetadataStore) -> None:
        self.store = store

    def history(self, candidate_id: str) -> tuple[CandidateState, ...]:
        events = self.store.list_states(candidate_id)
        return tuple(_decode_state(event.state) for event in events)

    def current(self, candidate_id: str) -> CandidateState | None:
        states = self.history(candidate_id)
        return None if not states else states[-1]

    def initialize(self, candidate_id: str, detail: str | None = None) -> StoredStateEvent:
        if self.current(candidate_id) is not None:
            raise ExperimentStateError("candidate state machine is already initialized")
        return _atomic_append(self.store, candidate_id, None, CandidateState.PLANNED, detail)

    def transition(
        self,
        candidate_id: str,
        target: CandidateState,
        detail: str | None = None,
    ) -> StoredStateEvent:
        states = self.history(candidate_id)
        if not states:
            raise ExperimentStateError("candidate state machine is not initialized")
        current = states[-1]
        if target not in _DIRECT_TRANSITIONS[current]:
            raise ExperimentStateError(
                f"invalid candidate state transition {current.value} -> {target.value}"
            )
        if current in {CandidateState.INTERRUPTED, CandidateState.RECOVERABLE_OOM} and target in {
            CandidateState.RUNNING,
            CandidateState.EVALUATING,
        }:
            stage = _active_stage_before_recovery(states[:-1])
            required = (
                CandidateState.RUNNING
                if stage is CandidateWorkStage.MUTATION
                else CandidateState.EVALUATING
            )
            if target is not required:
                raise ExperimentStateError(
                    f"recovery must return to {required.value}, not {target.value}"
                )
        return _atomic_append(self.store, candidate_id, current, target, detail)

    def recovery_plan(self, candidate_id: str) -> CandidateRecoveryPlan:
        states = self.history(candidate_id)
        if not states:
            raise ExperimentStateError("candidate state machine is not initialized")
        current = states[-1]
        if current in _TERMINAL_STATES:
            return CandidateRecoveryPlan(candidate_id, current, None, False, True)
        if current is CandidateState.PLANNED:
            return CandidateRecoveryPlan(
                candidate_id,
                current,
                CandidateWorkStage.MUTATION,
                False,
                False,
            )
        if current is CandidateState.RUNNING:
            return CandidateRecoveryPlan(
                candidate_id,
                current,
                CandidateWorkStage.MUTATION,
                True,
                False,
            )
        if current is CandidateState.EVALUATING:
            return CandidateRecoveryPlan(
                candidate_id,
                current,
                CandidateWorkStage.EVALUATION,
                True,
                False,
            )
        stage = _active_stage_before_recovery(states[:-1])
        return CandidateRecoveryPlan(candidate_id, current, stage, True, False)

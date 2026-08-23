"""Resumable candidate campaign scheduling over leases, OOM recovery, and tiered evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

from modelsurgeon.evaluation.baseline_cache import (
    BaselineArtifact,
    BaselineCache,
    BaselineCacheKey,
)
from modelsurgeon.evaluation.tiered import (
    TieredEvaluationBackend,
    TieredEvaluationConfig,
    TieredEvaluationReport,
    run_tiered_evaluation,
)
from modelsurgeon.experiments.candidates import MutationCandidate
from modelsurgeon.experiments.gpu_cleanup import ExperimentGPUCleanup
from modelsurgeon.experiments.identity import (
    canonical_identity_json,
    derive_candidate_identity,
)
from modelsurgeon.experiments.oom_recovery import (
    OOMAttemptConfig,
    OOMRecoveryResult,
    OOMRetryPolicy,
    run_with_oom_recovery,
)
from modelsurgeon.experiments.queue import ExperimentWorkQueue, WorkLease, WorkLeaseError
from modelsurgeon.experiments.schema import (
    EXPERIMENT_SCHEMA_VERSION,
    DatasetTarget,
    ModelTarget,
    QuantizationControl,
    SeedContext,
    VersionContext,
)
from modelsurgeon.experiments.state_machine import (
    CandidateState,
    CandidateWorkStage,
    ExperimentStateError,
    ExperimentStateMachine,
)
from modelsurgeon.experiments.store import ExperimentMetadataStore, ExperimentStoreError
from modelsurgeon.experiments.hardware import HardwareInventory

CAMPAIGN_RUNNER_VERSION = "1"
_CAMPAIGN_MUTATION_ID = "__campaign__"


class CampaignError(RuntimeError):
    """Raised when a campaign plan or persisted lifecycle is inconsistent."""


class CampaignOutcome(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"


_TERMINAL_OUTCOMES = frozenset(
    {CampaignOutcome.SUCCEEDED, CampaignOutcome.REJECTED, CampaignOutcome.FAILED}
)
_TERMINAL_STATES = frozenset(
    {CandidateState.SUCCEEDED, CandidateState.REJECTED, CandidateState.FAILED}
)


@dataclass(frozen=True, slots=True)
class MutationCheckpoint:
    checkpoint_id: str
    metadata: tuple[tuple[str, str | int | float | bool | None], ...] = ()

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise CampaignError("mutation checkpoints require an identity")
        keys = tuple(key for key, _ in self.metadata)
        if keys != tuple(sorted(set(keys))) or any(not key for key in keys):
            raise CampaignError("checkpoint metadata keys must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        return {"checkpoint_id": self.checkpoint_id, "metadata": dict(self.metadata)}

    @classmethod
    def from_record(cls, value: object) -> MutationCheckpoint:
        record = _json_object_value(value, "mutation checkpoint")
        if set(record) != {"checkpoint_id", "metadata"}:
            raise CampaignError("mutation checkpoint has missing or unknown fields")
        checkpoint_id = record["checkpoint_id"]
        metadata = record["metadata"]
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise CampaignError("mutation checkpoint identity is invalid")
        if not isinstance(metadata, dict) or not all(
            isinstance(key, str) and key for key in metadata
        ):
            raise CampaignError("mutation checkpoint metadata is invalid")
        entries: list[tuple[str, str | int | float | bool | None]] = []
        for key, item in metadata.items():
            if item is not None and not isinstance(item, (str, int, float, bool)):
                raise CampaignError("mutation checkpoint metadata values must be primitive")
            entries.append((key, item))
        return cls(checkpoint_id, tuple(sorted(entries)))


@dataclass(frozen=True, slots=True)
class CampaignContext:
    experiment_id: str
    run_id: str
    attempt_id: str
    model: ModelTarget
    dataset: DatasetTarget
    hardware: HardwareInventory
    versions: VersionContext
    seeds: SeedContext
    quantization_control: QuantizationControl | None = None

    def __post_init__(self) -> None:
        if not self.experiment_id.startswith("exp_") or not self.run_id.startswith("run_"):
            raise CampaignError("campaign context requires canonical experiment and run IDs")
        if not self.attempt_id:
            raise CampaignError("campaign attempt identity is required")

    @property
    def baseline_key(self) -> BaselineCacheKey:
        return BaselineCacheKey(
            self.model.revision,
            self.dataset.revision,
            self.dataset.tokenizer_revision,
            self.versions.evaluator_version,
        )


@dataclass(frozen=True, slots=True)
class CampaignRunnerConfig:
    worker_id: str
    lease_duration_ns: int = 60_000_000_000
    mutation_attempt: OOMAttemptConfig = OOMAttemptConfig(1, 1)
    evaluation_attempt: OOMAttemptConfig = OOMAttemptConfig(1, 1)
    oom_policy: OOMRetryPolicy = OOMRetryPolicy()
    evaluation: TieredEvaluationConfig = TieredEvaluationConfig()

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise CampaignError("campaign worker identity is required")
        if isinstance(self.lease_duration_ns, bool) or self.lease_duration_ns <= 0:
            raise CampaignError("campaign lease duration must be positive integer nanoseconds")


@dataclass(frozen=True, slots=True)
class CampaignCandidateStatus:
    candidate_id: str
    checkpoint: MutationCheckpoint | None
    evaluation: dict[str, object] | None
    recovery: dict[str, object]
    outcome: CampaignOutcome
    detail: str | None


@dataclass(frozen=True, slots=True)
class CampaignProgress:
    run_id: str
    total: int
    planned: int
    active: int
    interrupted: int
    succeeded: int
    rejected: int
    failed: int

    @property
    def completed(self) -> int:
        return self.succeeded + self.rejected + self.failed

    @property
    def remaining(self) -> int:
        return self.total - self.completed

    def to_record(self) -> dict[str, str | int]:
        return {
            "run_id": self.run_id,
            "total": self.total,
            "planned": self.planned,
            "active": self.active,
            "interrupted": self.interrupted,
            "succeeded": self.succeeded,
            "rejected": self.rejected,
            "failed": self.failed,
            "completed": self.completed,
            "remaining": self.remaining,
        }


@dataclass(frozen=True, slots=True)
class CandidateExecutionFailure:
    candidate_id: str
    exception_type: str
    message: str


@dataclass(frozen=True, slots=True)
class CampaignRunReport:
    version: str
    progress: CampaignProgress
    processed: tuple[str, ...]
    skipped_completed: tuple[str, ...]
    leased_elsewhere: tuple[str, ...]
    lease_lost: tuple[str, ...]
    failures: tuple[CandidateExecutionFailure, ...]

    def __post_init__(self) -> None:
        if self.version != CAMPAIGN_RUNNER_VERSION:
            raise CampaignError(f"unsupported campaign runner version {self.version}")


class CampaignWorker(Protocol):
    def compute_baseline(self) -> BaselineArtifact: ...

    def mutate(
        self,
        candidate: MutationCandidate,
        baseline: BaselineArtifact,
        config: OOMAttemptConfig,
        cleanup: ExperimentGPUCleanup,
        heartbeat: Callable[[], None],
    ) -> MutationCheckpoint: ...

    def evaluation_backend(
        self,
        candidate: MutationCandidate,
        checkpoint: MutationCheckpoint,
        baseline: BaselineArtifact,
        config: OOMAttemptConfig,
        cleanup: ExperimentGPUCleanup,
        heartbeat: Callable[[], None],
    ) -> TieredEvaluationBackend: ...


def _json_object(payload: str, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise CampaignError(f"stored {label} JSON is malformed") from error
    return _json_object_value(value, label)


def _json_object_value(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CampaignError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _input_id(context: CampaignContext) -> str:
    payload = {
        "model": context.model.to_record(),
        "dataset": context.dataset.to_record(),
        "config_digest": context.versions.config_digest,
    }
    encoded = canonical_identity_json(payload).encode("utf-8")
    return f"input_{hashlib.sha256(encoded).hexdigest()}"


def _plan_digest(candidates: tuple[MutationCandidate, ...]) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        encoded = canonical_identity_json(candidate.to_record()).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_candidates(
    context: CampaignContext,
    candidates: tuple[MutationCandidate, ...],
) -> None:
    ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(ids) != len(set(ids)):
        raise CampaignError("campaign candidate IDs must be unique")
    mutation_ids = tuple(candidate.mutation_id for candidate in candidates)
    if len(mutation_ids) != len(set(mutation_ids)):
        raise CampaignError("campaign mutation IDs must be unique")
    for candidate in candidates:
        expected = derive_candidate_identity(context.run_id, candidate.mutation_id).candidate_id
        if candidate.candidate_id != expected:
            raise CampaignError(
                f"candidate {candidate.candidate_id} does not belong to run {context.run_id}"
            )


def register_campaign(
    store: ExperimentMetadataStore,
    context: CampaignContext,
    candidates: tuple[MutationCandidate, ...],
) -> None:
    """Atomically register a deterministic plan before any candidate work is leased."""

    _validate_candidates(context, candidates)
    input_id = _input_id(context)
    plan_digest = _plan_digest(candidates)
    input_columns = (
        "input_id",
        "model_identifier",
        "model_revision",
        "model_family",
        "model_format",
        "model_parameter_count",
        "model_quantization",
        "dataset_identifier",
        "dataset_revision",
        "dataset_split",
        "dataset_manifest_id",
        "tokenizer",
        "tokenizer_revision",
        "config_digest",
    )
    input_values: tuple[object, ...] = (
        input_id,
        context.model.identifier,
        context.model.revision,
        context.model.family,
        context.model.format,
        context.model.parameter_count,
        context.model.quantization,
        context.dataset.identifier,
        context.dataset.revision,
        context.dataset.split,
        context.dataset.manifest_id,
        context.dataset.tokenizer,
        context.dataset.tokenizer_revision,
        context.versions.config_digest,
    )
    run_columns = (
        "run_id",
        "experiment_id",
        "attempt_id",
        "input_id",
        "mutation_id",
        "experiment_schema_version",
        "mutation_record_schema_version",
        "mutation_json",
        "outcome_json",
        "hardware_json",
        "versions_json",
        "seeds_json",
        "quantization_control_json",
    )
    run_values: tuple[object, ...] = (
        context.run_id,
        context.experiment_id,
        context.attempt_id,
        input_id,
        _CAMPAIGN_MUTATION_ID,
        EXPERIMENT_SCHEMA_VERSION,
        context.versions.mutation_record_schema_version,
        canonical_identity_json(
            {
                "kind": "campaign-plan",
                "plan_digest": plan_digest,
                "candidate_count": len(candidates),
            }
        ),
        canonical_identity_json({"kind": "pending"}),
        canonical_identity_json(context.hardware.to_record()),
        canonical_identity_json(context.versions.to_record()),
        canonical_identity_json(context.seeds.to_record()),
        None
        if context.quantization_control is None
        else canonical_identity_json(context.quantization_control.to_record()),
    )
    baseline_key_json = canonical_identity_json(context.baseline_key.to_record())

    try:
        with store._write() as connection:
            store._insert_or_verify(
                connection,
                table="experiment_inputs",
                key_column="input_id",
                key_value=input_id,
                columns=input_columns,
                values=input_values,
            )
            store._insert_or_verify(
                connection,
                table="experiment_runs",
                key_column="run_id",
                key_value=context.run_id,
                columns=run_columns,
                values=run_values,
            )
            store._insert_or_verify(
                connection,
                table="experiment_campaign_runs",
                key_column="run_id",
                key_value=context.run_id,
                columns=("run_id", "plan_digest", "baseline_key_json", "candidate_count"),
                values=(context.run_id, plan_digest, baseline_key_json, len(candidates)),
            )
            for order, candidate in enumerate(candidates):
                store._insert_or_verify(
                    connection,
                    table="experiment_candidates",
                    key_column="candidate_id",
                    key_value=candidate.candidate_id,
                    columns=(
                        "candidate_id",
                        "run_id",
                        "mutation_id",
                        "affected_components_json",
                        "candidate_order",
                    ),
                    values=(
                        candidate.candidate_id,
                        context.run_id,
                        candidate.mutation_id,
                        canonical_identity_json(
                            [str(item) for item in candidate.affected_components]
                        ),
                        order,
                    ),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO experiment_campaign_status(candidate_id)
                    VALUES (?)
                    """,
                    (candidate.candidate_id,),
                )
    except ExperimentStoreError as error:
        raise CampaignError(str(error)) from error


def campaign_status(
    store: ExperimentMetadataStore,
    candidate_id: str,
) -> CampaignCandidateStatus:
    with store.reader() as connection:
        row = connection.execute(
            "SELECT * FROM experiment_campaign_status WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
    if row is None:
        raise CampaignError(f"candidate {candidate_id} is not registered for a campaign")
    checkpoint = None
    if row["checkpoint_json"] is not None:
        checkpoint = MutationCheckpoint.from_record(
            _json_object(str(row["checkpoint_json"]), "checkpoint")
        )
    evaluation = None
    if row["evaluation_json"] is not None:
        evaluation = _json_object(str(row["evaluation_json"]), "evaluation")
    recovery = _json_object(str(row["recovery_json"]), "recovery")
    try:
        outcome = CampaignOutcome(str(row["outcome"]))
    except ValueError as error:
        raise CampaignError("stored campaign outcome is unknown") from error
    return CampaignCandidateStatus(
        candidate_id,
        checkpoint,
        evaluation,
        recovery,
        outcome,
        None if row["detail"] is None else str(row["detail"]),
    )


def persist_campaign_checkpoint(
    store: ExperimentMetadataStore,
    candidate_id: str,
    checkpoint: MutationCheckpoint,
) -> None:
    encoded = canonical_identity_json(checkpoint.to_record())
    with store._write() as connection:
        row = connection.execute(
            "SELECT checkpoint_json, outcome FROM experiment_campaign_status WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise CampaignError(f"unknown campaign candidate {candidate_id}")
        existing = None if row["checkpoint_json"] is None else str(row["checkpoint_json"])
        if existing is not None and existing != encoded:
            raise CampaignError("campaign checkpoint conflicts with immutable persisted checkpoint")
        if existing is None:
            connection.execute(
                "UPDATE experiment_campaign_status SET checkpoint_json = ? WHERE candidate_id = ?",
                (encoded, candidate_id),
            )


def persist_campaign_recovery(
    store: ExperimentMetadataStore,
    candidate_id: str,
    stage: CandidateWorkStage,
    recovery: OOMRecoveryResult[object],
) -> None:
    record = recovery.to_record()
    with store._write() as connection:
        row = connection.execute(
            "SELECT recovery_json FROM experiment_campaign_status WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise CampaignError(f"unknown campaign candidate {candidate_id}")
        root = _json_object(str(row["recovery_json"]), "recovery")
        current = root.get(stage.value, [])
        if not isinstance(current, list):
            raise CampaignError("stored campaign recovery history is malformed")
        if not current or current[-1] != record:
            current.append(record)
        root[stage.value] = current
        connection.execute(
            "UPDATE experiment_campaign_status SET recovery_json = ? WHERE candidate_id = ?",
            (canonical_identity_json(root), candidate_id),
        )


def persist_campaign_outcome(
    store: ExperimentMetadataStore,
    candidate_id: str,
    outcome: CampaignOutcome,
    *,
    evaluation: TieredEvaluationReport | None = None,
    detail: str | None = None,
) -> None:
    if outcome is CampaignOutcome.PENDING:
        raise CampaignError("terminal campaign persistence cannot write a pending outcome")
    evaluation_json = (
        None if evaluation is None else canonical_identity_json(evaluation.to_record())
    )
    with store._write() as connection:
        row = connection.execute(
            """
            SELECT outcome, evaluation_json, detail
            FROM experiment_campaign_status
            WHERE candidate_id = ?
            """,
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise CampaignError(f"unknown campaign candidate {candidate_id}")
        existing = CampaignOutcome(str(row["outcome"]))
        if existing in _TERMINAL_OUTCOMES:
            existing_evaluation = (
                None if row["evaluation_json"] is None else str(row["evaluation_json"])
            )
            existing_detail = None if row["detail"] is None else str(row["detail"])
            if (
                existing is not outcome
                or existing_evaluation != evaluation_json
                or existing_detail != detail
            ):
                raise CampaignError("terminal campaign outcome conflicts with persisted result")
            return
        connection.execute(
            """
            UPDATE experiment_campaign_status
            SET outcome = ?, evaluation_json = ?, detail = ?
            WHERE candidate_id = ?
            """,
            (outcome.value, evaluation_json, detail, candidate_id),
        )


def query_campaign_progress(store: ExperimentMetadataStore, run_id: str) -> CampaignProgress:
    with store.reader() as connection:
        rows = connection.execute(
            """
            SELECT c.candidate_id, s.outcome,
                   (
                       SELECT e.state
                       FROM experiment_state_events AS e
                       WHERE e.candidate_id = c.candidate_id
                       ORDER BY e.sequence DESC
                       LIMIT 1
                   ) AS state
            FROM experiment_candidates AS c
            JOIN experiment_campaign_status AS s ON s.candidate_id = c.candidate_id
            WHERE c.run_id = ?
            ORDER BY c.candidate_order, c.candidate_id
            """,
            (run_id,),
        ).fetchall()
    planned = active = interrupted = succeeded = rejected = failed = 0
    for row in rows:
        outcome = CampaignOutcome(str(row["outcome"]))
        state = None if row["state"] is None else CandidateState(str(row["state"]))
        if outcome is CampaignOutcome.SUCCEEDED or state is CandidateState.SUCCEEDED:
            succeeded += 1
        elif outcome is CampaignOutcome.REJECTED or state is CandidateState.REJECTED:
            rejected += 1
        elif outcome is CampaignOutcome.FAILED or state is CandidateState.FAILED:
            failed += 1
        elif state in {CandidateState.INTERRUPTED, CandidateState.RECOVERABLE_OOM}:
            interrupted += 1
        elif state in {CandidateState.RUNNING, CandidateState.EVALUATING}:
            active += 1
        else:
            planned += 1
    return CampaignProgress(
        run_id,
        len(rows),
        planned,
        active,
        interrupted,
        succeeded,
        rejected,
        failed,
    )


def _terminal_state(outcome: CampaignOutcome) -> CandidateState:
    return {
        CampaignOutcome.SUCCEEDED: CandidateState.SUCCEEDED,
        CampaignOutcome.REJECTED: CandidateState.REJECTED,
        CampaignOutcome.FAILED: CandidateState.FAILED,
    }[outcome]


def _reconcile_terminal(
    machine: ExperimentStateMachine,
    candidate_id: str,
    outcome: CampaignOutcome,
) -> None:
    target = _terminal_state(outcome)
    current = machine.current(candidate_id)
    if current is target:
        return
    if current in _TERMINAL_STATES:
        raise CampaignError(
            f"candidate {candidate_id} terminal state disagrees with persisted outcome"
        )
    if current is None:
        machine.initialize(candidate_id, "reconcile persisted campaign outcome")
        current = CandidateState.PLANNED
    if target is CandidateState.FAILED:
        machine.transition(candidate_id, target, "reconcile persisted failed result")
        return
    if target is CandidateState.REJECTED:
        machine.transition(candidate_id, target, "reconcile persisted rejected result")
        return
    while current is not CandidateState.EVALUATING:
        if current is CandidateState.PLANNED:
            machine.transition(candidate_id, CandidateState.RUNNING, "reconcile completed mutation")
        elif current is CandidateState.RUNNING:
            machine.transition(candidate_id, CandidateState.EVALUATING, "reconcile completed evaluation")
        elif current in {CandidateState.INTERRUPTED, CandidateState.RECOVERABLE_OOM}:
            recovery = machine.recovery_plan(candidate_id)
            resume = (
                CandidateState.RUNNING
                if recovery.next_stage is CandidateWorkStage.MUTATION
                else CandidateState.EVALUATING
            )
            machine.transition(candidate_id, resume, "reconcile persisted terminal result")
        else:
            raise CampaignError("cannot reconcile succeeded campaign result from current state")
        current = machine.current(candidate_id)
    machine.transition(candidate_id, CandidateState.SUCCEEDED, "reconcile persisted succeeded result")


class CampaignRunner:
    """Schedule one deterministic campaign while isolating candidate failures and restarts."""

    def __init__(
        self,
        store: ExperimentMetadataStore,
        baseline_cache: BaselineCache,
        context: CampaignContext,
        candidates: tuple[MutationCandidate, ...],
        worker: CampaignWorker,
        config: CampaignRunnerConfig,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        cleanup_factory: Callable[[], ExperimentGPUCleanup] | None = None,
    ) -> None:
        _validate_candidates(context, candidates)
        self.store = store
        self.baseline_cache = baseline_cache
        self.context = context
        self.candidates = candidates
        self.worker = worker
        self.config = config
        self.clock_ns = clock_ns
        self.cleanup_factory = cleanup_factory or ExperimentGPUCleanup
        self.machine = ExperimentStateMachine(store)
        self.queue = ExperimentWorkQueue(store, lease_duration_ns=config.lease_duration_ns)

    def _heartbeat(self, lease: WorkLease) -> Callable[[], None]:
        def heartbeat() -> None:
            self.queue.heartbeat(lease.lease_token, now_ns=self.clock_ns())

        return heartbeat

    def _record_recovery[T](
        self,
        candidate_id: str,
        stage: CandidateWorkStage,
        result: OOMRecoveryResult[T],
    ) -> None:
        persist_campaign_recovery(
            self.store,
            candidate_id,
            stage,
            cast(OOMRecoveryResult[object], result),
        )

    def _resume_state(self, candidate_id: str, status: CampaignCandidateStatus) -> CandidateState:
        current = self.machine.current(candidate_id)
        if current is None:
            self.machine.initialize(candidate_id)
            current = CandidateState.PLANNED
        if status.outcome in _TERMINAL_OUTCOMES:
            _reconcile_terminal(self.machine, candidate_id, status.outcome)
            resolved = self.machine.current(candidate_id)
            if resolved is None:
                raise CampaignError("terminal reconciliation lost candidate state")
            return resolved
        if current in {CandidateState.INTERRUPTED, CandidateState.RECOVERABLE_OOM}:
            recovery = self.machine.recovery_plan(candidate_id)
            target = (
                CandidateState.RUNNING
                if recovery.next_stage is CandidateWorkStage.MUTATION
                else CandidateState.EVALUATING
            )
            self.machine.transition(candidate_id, target, "resume campaign candidate")
            current = target
        if current is CandidateState.PLANNED:
            self.machine.transition(candidate_id, CandidateState.RUNNING)
            current = CandidateState.RUNNING
        if current is CandidateState.RUNNING and status.checkpoint is not None:
            self.machine.transition(
                candidate_id,
                CandidateState.EVALUATING,
                "checkpoint persisted before prior interruption",
            )
            current = CandidateState.EVALUATING
        return current

    def _run_candidate(
        self,
        candidate: MutationCandidate,
        baseline: BaselineArtifact,
        lease: WorkLease,
    ) -> None:
        status = campaign_status(self.store, candidate.candidate_id)
        current = self._resume_state(candidate.candidate_id, status)
        heartbeat = self._heartbeat(lease)
        checkpoint = status.checkpoint

        if current is CandidateState.RUNNING:
            mutation = run_with_oom_recovery(
                self.machine,
                candidate.candidate_id,
                CandidateWorkStage.MUTATION,
                self.config.mutation_attempt,
                self.config.oom_policy,
                self.cleanup_factory,
                lambda attempt, cleanup: self.worker.mutate(
                    candidate,
                    baseline,
                    attempt,
                    cleanup,
                    heartbeat,
                ),
                lease_heartbeat=heartbeat,
            )
            self._record_recovery(
                candidate.candidate_id,
                CandidateWorkStage.MUTATION,
                mutation,
            )
            if not mutation.succeeded:
                persist_campaign_outcome(
                    self.store,
                    candidate.candidate_id,
                    CampaignOutcome.FAILED,
                    detail="mutation OOM recovery exhausted",
                )
                return
            checkpoint = mutation.value
            if checkpoint is None:
                raise CampaignError("mutation stage returned no persistent checkpoint")
            persist_campaign_checkpoint(self.store, candidate.candidate_id, checkpoint)
            self.machine.transition(candidate.candidate_id, CandidateState.EVALUATING)
            current = CandidateState.EVALUATING

        if current is not CandidateState.EVALUATING:
            if current in _TERMINAL_STATES:
                return
            raise CampaignError(f"candidate reached unsupported campaign state {current.value}")
        if checkpoint is None:
            checkpoint = campaign_status(self.store, candidate.candidate_id).checkpoint
        if checkpoint is None:
            raise CampaignError("evaluation resume requires a persisted mutation checkpoint")

        evaluation = run_with_oom_recovery(
            self.machine,
            candidate.candidate_id,
            CandidateWorkStage.EVALUATION,
            self.config.evaluation_attempt,
            self.config.oom_policy,
            self.cleanup_factory,
            lambda attempt, cleanup: run_tiered_evaluation(
                self.worker.evaluation_backend(
                    candidate,
                    checkpoint,
                    baseline,
                    attempt,
                    cleanup,
                    heartbeat,
                ),
                self.config.evaluation,
            ),
            lease_heartbeat=heartbeat,
        )
        self._record_recovery(
            candidate.candidate_id,
            CandidateWorkStage.EVALUATION,
            evaluation,
        )
        if not evaluation.succeeded:
            persist_campaign_outcome(
                self.store,
                candidate.candidate_id,
                CampaignOutcome.FAILED,
                detail="evaluation OOM recovery exhausted",
            )
            return
        report = evaluation.value
        if report is None:
            raise CampaignError("evaluation stage returned no tiered report")
        outcome = CampaignOutcome.SUCCEEDED if report.accepted else CampaignOutcome.REJECTED
        persist_campaign_outcome(
            self.store,
            candidate.candidate_id,
            outcome,
            evaluation=report,
        )
        self.machine.transition(
            candidate.candidate_id,
            CandidateState.SUCCEEDED if report.accepted else CandidateState.REJECTED,
        )

    def run(self) -> CampaignRunReport:
        register_campaign(self.store, self.context, self.candidates)
        skipped_completed: list[str] = []
        pending: list[MutationCandidate] = []
        for candidate in self.candidates:
            status = campaign_status(self.store, candidate.candidate_id)
            current = self.machine.current(candidate.candidate_id)
            if status.outcome in _TERMINAL_OUTCOMES:
                _reconcile_terminal(self.machine, candidate.candidate_id, status.outcome)
                skipped_completed.append(candidate.candidate_id)
            elif current in _TERMINAL_STATES:
                skipped_completed.append(candidate.candidate_id)
            else:
                pending.append(candidate)

        if not pending:
            return CampaignRunReport(
                CAMPAIGN_RUNNER_VERSION,
                query_campaign_progress(self.store, self.context.run_id),
                (),
                tuple(skipped_completed),
                (),
                (),
                (),
            )

        baseline = self.baseline_cache.get_or_compute(
            self.context.baseline_key,
            self.worker.compute_baseline,
        )
        processed: list[str] = []
        leased_elsewhere: list[str] = []
        lease_lost: list[str] = []
        failures: list[CandidateExecutionFailure] = []
        for candidate in pending:
            now = self.clock_ns()
            lease = self.queue.claim(
                candidate.candidate_id,
                attempt_id=self.context.attempt_id,
                worker_id=self.config.worker_id,
                now_ns=now,
            )
            if lease is None:
                leased_elsewhere.append(candidate.candidate_id)
                continue
            try:
                self._run_candidate(candidate, baseline, lease)
                processed.append(candidate.candidate_id)
            except WorkLeaseError:
                lease_lost.append(candidate.candidate_id)
                continue
            except Exception as error:
                failures.append(
                    CandidateExecutionFailure(
                        candidate.candidate_id,
                        type(error).__name__,
                        str(error),
                    )
                )
                current = self.machine.current(candidate.candidate_id)
                if current not in _TERMINAL_STATES:
                    try:
                        self.machine.transition(
                            candidate.candidate_id,
                            CandidateState.FAILED,
                            f"campaign-error:{type(error).__name__}",
                        )
                    except ExperimentStateError:
                        pass
                try:
                    persist_campaign_outcome(
                        self.store,
                        candidate.candidate_id,
                        CampaignOutcome.FAILED,
                        detail=f"{type(error).__name__}: {error}",
                    )
                except CampaignError:
                    pass
            finally:
                current = self.machine.current(candidate.candidate_id)
                if current in _TERMINAL_STATES:
                    try:
                        self.queue.complete(lease.lease_token, now_ns=self.clock_ns())
                    except WorkLeaseError:
                        if candidate.candidate_id not in lease_lost:
                            lease_lost.append(candidate.candidate_id)

        return CampaignRunReport(
            CAMPAIGN_RUNNER_VERSION,
            query_campaign_progress(self.store, self.context.run_id),
            tuple(processed),
            tuple(skipped_completed),
            tuple(leased_elsewhere),
            tuple(lease_lost),
            tuple(failures),
        )

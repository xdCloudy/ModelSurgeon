"""Single-mutation experiment orchestration for the command-line interface."""

from __future__ import annotations

import importlib
import json
import math
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from modelsurgeon.evaluation.tiered import TieredEvaluationReport
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.contracts import (
    MUTATION_SCHEMA_VERSION,
    MutationKind,
    MutationPlan,
    MutationPrimitive,
    MutationRequest,
    MutationTransaction,
    TransactionState,
)
from modelsurgeon.surgery.serialization import (
    MutationOutcome,
    MutationOutcomeStatus,
    MutationProvenance,
    MutationRunRecord,
)

EXPERIMENT_COMMAND_VERSION = "1"


class ExperimentCommandError(ValueError):
    """Raised when an experiment request or runtime cannot satisfy the CLI contract."""


@dataclass(frozen=True, slots=True)
class ResolvedExperiment:
    plan: MutationPlan
    provenance: MutationProvenance


@dataclass(frozen=True, slots=True)
class SingleMutationExperimentResult:
    dry_run: bool
    run_record: MutationRunRecord
    evaluation: TieredEvaluationReport | None
    version: str = EXPERIMENT_COMMAND_VERSION

    def __post_init__(self) -> None:
        if self.dry_run != (self.evaluation is None):
            raise ExperimentCommandError(
                "dry-run state must agree with the presence of evaluation results"
            )
        if self.dry_run and self.run_record.outcome is not None:
            raise ExperimentCommandError("dry runs cannot contain mutation outcomes")
        if not self.dry_run:
            outcome = self.run_record.outcome
            if outcome is None or outcome.status is not MutationOutcomeStatus.ROLLED_BACK:
                raise ExperimentCommandError(
                    "single-mutation experiments must finish with a rolled-back outcome"
                )

    def to_record(self, *, redact_local_paths: bool = True) -> dict[str, object]:
        return {
            "record_type": "single_mutation_experiment",
            "version": self.version,
            "dry_run": self.dry_run,
            "run": self.run_record.to_record(redact_local_paths=redact_local_paths),
            "evaluation": (
                None if self.evaluation is None else self.evaluation.to_record()
            ),
        }

    def canonical_json(self, *, redact_local_paths: bool = True) -> str:
        return json.dumps(
            self.to_record(redact_local_paths=redact_local_paths),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


@runtime_checkable
class SingleMutationExperimentRuntime(Protocol):
    """Adapter/model-specific operations required by the CLI experiment orchestrator."""

    def resolve(self, request: MutationRequest) -> ResolvedExperiment: ...

    def transaction(
        self,
        plan: MutationPlan,
    ) -> AbstractContextManager[MutationTransaction]: ...

    def mutation_scope(
        self,
        plan: MutationPlan,
        transaction: MutationTransaction,
    ) -> AbstractContextManager[object]: ...

    def evaluate(self, plan: MutationPlan) -> TieredEvaluationReport: ...

    def rolled_back_outcome(
        self,
        plan: MutationPlan,
        evaluation: TieredEvaluationReport,
    ) -> MutationOutcome: ...


def _primitive(value: object, name: str) -> MutationPrimitive:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentCommandError(f"{name} must be finite")
        return value
    raise ExperimentCommandError(f"{name} must be a JSON primitive")


def parse_mutation_request_json(payload: str) -> MutationRequest:
    """Parse the canonical `MutationRequest.to_record()` JSON representation."""

    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ExperimentCommandError("mutation request is not valid JSON") from error
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise ExperimentCommandError("mutation request must be a JSON object")
    expected = {"schema_version", "kind", "targets", "parameters"}
    if set(raw) != expected:
        raise ExperimentCommandError("mutation request has missing or unknown fields")
    schema_version = raw["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != MUTATION_SCHEMA_VERSION
    ):
        raise ExperimentCommandError("unsupported mutation request schema version")
    try:
        kind = MutationKind(raw["kind"])
    except (TypeError, ValueError) as error:
        raise ExperimentCommandError("mutation request kind is unknown") from error
    targets_raw = raw["targets"]
    if not isinstance(targets_raw, list) or not targets_raw:
        raise ExperimentCommandError("mutation request targets must be a non-empty array")
    try:
        targets = tuple(sorted(ComponentId.parse(value) for value in targets_raw))
    except (TypeError, ValueError) as error:
        raise ExperimentCommandError("mutation request contains an invalid target") from error
    parameters_raw = raw["parameters"]
    if not isinstance(parameters_raw, dict) or not all(
        isinstance(key, str) and key for key in parameters_raw
    ):
        raise ExperimentCommandError("mutation request parameters must be an object")
    parameters = tuple(
        sorted(
            (key, _primitive(value, f"parameter {key}"))
            for key, value in parameters_raw.items()
        )
    )
    return MutationRequest(kind, targets, parameters)


def read_mutation_request(path: str | Path) -> MutationRequest:
    try:
        payload = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ExperimentCommandError(f"cannot read mutation request: {error}") from error
    return parse_mutation_request_json(payload)


def load_experiment_runtime(specification: str) -> SingleMutationExperimentRuntime:
    """Load a zero-argument runtime factory from `module:attribute`."""

    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ExperimentCommandError("runtime must use module:factory syntax")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    except (ImportError, AttributeError) as error:
        raise ExperimentCommandError(
            f"cannot load experiment runtime factory {specification!r}"
        ) from error
    if not callable(factory):
        raise ExperimentCommandError("experiment runtime attribute must be callable")
    try:
        runtime = factory()
    except Exception as error:
        raise ExperimentCommandError("experiment runtime factory failed") from error
    if not isinstance(runtime, SingleMutationExperimentRuntime):
        raise ExperimentCommandError(
            "experiment runtime does not implement the required runtime contract"
        )
    return runtime


def _rollback_open_transaction(transaction: MutationTransaction) -> None:
    if transaction.state in {TransactionState.PREPARED, TransactionState.APPLIED}:
        transaction.rollback()
    if transaction.state is not TransactionState.ROLLED_BACK:
        raise ExperimentCommandError(
            f"experiment transaction ended in unsafe state {transaction.state.value}"
        )


def run_single_mutation_experiment(
    request: MutationRequest,
    runtime: SingleMutationExperimentRuntime,
    *,
    dry_run: bool = False,
) -> SingleMutationExperimentResult:
    """Resolve one mutation, evaluate it under a transaction, and always roll it back."""

    resolved = runtime.resolve(request)
    if resolved.plan.request != request:
        raise ExperimentCommandError("runtime resolved a plan for a different mutation request")
    if dry_run:
        return SingleMutationExperimentResult(
            True,
            MutationRunRecord(resolved.plan, resolved.provenance),
            None,
        )

    evaluation: TieredEvaluationReport
    with runtime.transaction(resolved.plan) as transaction:
        try:
            with runtime.mutation_scope(resolved.plan, transaction):
                evaluation = runtime.evaluate(resolved.plan)
        finally:
            _rollback_open_transaction(transaction)

    outcome = runtime.rolled_back_outcome(resolved.plan, evaluation)
    if outcome.status is not MutationOutcomeStatus.ROLLED_BACK:
        raise ExperimentCommandError("runtime outcome must describe a rolled-back experiment")
    run_record = MutationRunRecord(resolved.plan, resolved.provenance, outcome)
    return SingleMutationExperimentResult(False, run_record, evaluation)


def write_experiment_result(
    path: str | Path,
    result: SingleMutationExperimentResult,
    *,
    redact_local_paths: bool = True,
) -> None:
    """Persist one result without overwriting an existing user artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = result.canonical_json(redact_local_paths=redact_local_paths) + "\n"
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise ExperimentCommandError(
            f"experiment result destination already exists: {destination}"
        ) from error
    except OSError as error:
        raise ExperimentCommandError(f"cannot write experiment result: {error}") from error

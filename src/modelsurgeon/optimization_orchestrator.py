"""Bounded, resumable execution around the typed optimization coordinators.

The orchestrator owns workflow state only. Model-specific work is supplied by a
trusted runtime adapter; no default adapter fabricates measurements or artifacts.
Every stage is content-addressed by the immutable plan and stage name, and state
is replaced atomically after each transition.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from modelsurgeon.experiments.coordinator import (
    ApprovedCampaignPlan,
    AutonomousCampaignCoordinator,
    MeasuredCandidateEvidence,
    PromotionOutcome,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.optimization_package import diff_plans, plan_digest
from modelsurgeon.optimization import OptimizeOutcome, OptimizePlan
from modelsurgeon.surgery.contracts import TransactionState

ORCHESTRATOR_SCHEMA_VERSION = 2


class OptimizeOrchestratorError(RuntimeError):
    """Raised when an optimization workflow cannot safely advance or resume."""


class OptimizeStage(StrEnum):
    PROFILE = "profile"
    CAPABILITY = "capability"
    BASELINE = "baseline"
    CANDIDATE_GENERATION = "candidate_generation"
    ACTIVE_SEARCH = "active_search"
    SURGERY = "surgery"
    REPAIR = "repair"
    QUANTIZATION = "quantization"
    DEPLOYMENT_BENCHMARK = "deployment_benchmark"
    PARETO = "pareto"
    REPORT = "report"


class WorkflowStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowOutcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


STAGE_ORDER: tuple[OptimizeStage, ...] = (
    OptimizeStage.PROFILE,
    OptimizeStage.CAPABILITY,
    OptimizeStage.BASELINE,
    OptimizeStage.CANDIDATE_GENERATION,
    OptimizeStage.ACTIVE_SEARCH,
    OptimizeStage.SURGERY,
    OptimizeStage.REPAIR,
    OptimizeStage.QUANTIZATION,
    OptimizeStage.DEPLOYMENT_BENCHMARK,
    OptimizeStage.PARETO,
    OptimizeStage.REPORT,
)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OptimizeOrchestratorError(f"{label} must be non-empty text")
    return value


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if not text.startswith("sha256:") or len(text) != 71:
        raise OptimizeOrchestratorError(f"{label} must be a SHA-256 content digest")
    try:
        int(text[7:], 16)
    except ValueError as error:
        raise OptimizeOrchestratorError(f"{label} must be a SHA-256 content digest") from error
    return text


def _canonical(value: object) -> str:
    return canonical_identity_json(value)


@dataclass(frozen=True, slots=True)
class StageResult:
    """Immutable result returned by a trusted runtime for one stage."""

    outcome: WorkflowOutcome
    evidence_id: str
    detail: str
    measured: bool = False
    complete: bool = False
    constraints_passed: bool = False
    artifact_digest: str | None = None
    candidate_id: str | None = None
    candidate_state_id: str | None = None
    evaluation_id: str | None = None
    transaction_state: TransactionState | None = None
    artifact_immutable: bool = False
    alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.evidence_id, "stage evidence ID")
        _text(self.detail, "stage detail")
        if self.artifact_digest is not None:
            _digest(self.artifact_digest, "stage artifact digest")
        for value, label, prefix in (
            (self.candidate_id, "candidate ID", "candidate_"),
            (self.candidate_state_id, "candidate state ID", "state_"),
            (self.evaluation_id, "evaluation ID", "evaluation_"),
        ):
            if value is not None and not value.startswith(prefix):
                raise OptimizeOrchestratorError(f"{label} must start with {prefix}")
        if self.alternatives != tuple(sorted(set(self.alternatives))):
            raise OptimizeOrchestratorError("stage alternatives must be sorted and unique")
        if self.outcome is WorkflowOutcome.SUPPORTED and not self.complete:
            raise OptimizeOrchestratorError("supported stages require complete evidence")
        if self.artifact_immutable and self.artifact_digest is None:
            raise OptimizeOrchestratorError("immutable artifact evidence requires a digest")

    def to_record(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "evidence_id": self.evidence_id,
            "detail": self.detail,
            "measured": self.measured,
            "complete": self.complete,
            "constraints_passed": self.constraints_passed,
            "artifact_digest": self.artifact_digest,
            "candidate_id": self.candidate_id,
            "candidate_state_id": self.candidate_state_id,
            "evaluation_id": self.evaluation_id,
            "transaction_state": (
                None if self.transaction_state is None else self.transaction_state.value
            ),
            "artifact_immutable": self.artifact_immutable,
            "alternatives": list(self.alternatives),
        }


@dataclass(frozen=True, slots=True)
class StageRecord:
    """Durable stage cursor entry; completed entries are never executed twice."""

    stage: OptimizeStage
    stage_id: str
    dependencies: tuple[OptimizeStage, ...]
    status: StageStatus = StageStatus.PENDING
    attempts: int = 0
    result: StageResult | None = None

    def __post_init__(self) -> None:
        _text(self.stage_id, "stage ID")
        if self.stage_id != f"stage_{self.stage.value}":
            raise OptimizeOrchestratorError("stage ID is not deterministic")
        if self.dependencies != tuple(STAGE_ORDER[: STAGE_ORDER.index(self.stage)]):
            raise OptimizeOrchestratorError("stage dependencies do not match workflow DAG")
        if self.attempts < 0:
            raise OptimizeOrchestratorError("stage attempts cannot be negative")
        if self.status is StageStatus.COMPLETED and self.result is None:
            raise OptimizeOrchestratorError("completed stages require a result")

    def to_record(self) -> dict[str, object]:
        return {
            "stage": self.stage.value,
            "stage_id": self.stage_id,
            "dependencies": [item.value for item in self.dependencies],
            "status": self.status.value,
            "attempts": self.attempts,
            "result": None if self.result is None else self.result.to_record(),
        }


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    code: str
    approved: bool
    recorded_by: str
    plan_id: str
    plan_digest: str
    diff_id: str
    expires_at: str
    operator_context: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _text(self.code, "approval code")
        _text(self.recorded_by, "approval recorder")
        _text(self.plan_id, "approval plan ID")
        if len(self.plan_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.plan_digest
        ):
            raise OptimizeOrchestratorError("approval plan digest must be a lowercase SHA-256")
        _text(self.diff_id, "approval diff ID")
        try:
            expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise OptimizeOrchestratorError(
                "approval expiry must be an ISO-8601 timestamp"
            ) from error
        if expiry.tzinfo is None:
            raise OptimizeOrchestratorError("approval expiry must include a timezone")
        if self.operator_context != tuple(sorted(self.operator_context)):
            raise OptimizeOrchestratorError("approval operator context must be canonical")

    @property
    def approval_id(self) -> str:
        digest = hashlib.sha256(
            _canonical(
                {
                    "code": self.code,
                    "approved": self.approved,
                    "recorded_by": self.recorded_by,
                    "plan_id": self.plan_id,
                    "plan_digest": self.plan_digest,
                    "diff_id": self.diff_id,
                    "expires_at": self.expires_at,
                    "operator_context": dict(self.operator_context),
                }
            ).encode()
        ).hexdigest()
        return f"approval_{digest}"

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code,
            "approved": self.approved,
            "recorded_by": self.recorded_by,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "diff_id": self.diff_id,
            "expires_at": self.expires_at,
            "operator_context": {key: value for key, value in self.operator_context},
            "approval_id": self.approval_id,
        }

    @property
    def active(self) -> bool:
        return datetime.now(UTC) < datetime.fromisoformat(
            self.expires_at.replace("Z", "+00:00")
        ).astimezone(UTC)


@dataclass(frozen=True, slots=True)
class OverrideRecord:
    name: str
    value: str
    approved: bool
    approval_id: str | None

    def __post_init__(self) -> None:
        _text(self.name, "override name")
        _text(self.value, "override value")
        if self.approved and self.approval_id is None:
            raise OptimizeOrchestratorError("approved overrides require an approval ID")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "approved": self.approved,
            "approval_id": self.approval_id,
        }


@dataclass(frozen=True, slots=True)
class OptimizeRun:
    """Complete persisted orchestration state and retained evidence."""

    run_id: str
    plan_id: str
    plan_digest: str
    plan_record: Mapping[str, object]
    source_artifact_digest: str
    status: WorkflowStatus
    outcome: WorkflowOutcome
    cursor: int
    stages: tuple[StageRecord, ...]
    approvals: tuple[ApprovalRecord, ...]
    overrides: tuple[OverrideRecord, ...]
    accepted_artifact_digest: str | None
    alternatives: tuple[str, ...]
    reasons: tuple[str, ...]
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.run_id, "run ID")
        _text(self.plan_id, "plan ID")
        if len(self.plan_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.plan_digest
        ):
            raise OptimizeOrchestratorError("run plan digest must be a lowercase SHA-256")
        if (
            not isinstance(self.plan_record, Mapping)
            or self.plan_record.get("plan_id") != self.plan_id
        ):
            raise OptimizeOrchestratorError("run must retain its canonical plan record")
        if plan_digest(self.plan_record) != self.plan_digest:
            raise OptimizeOrchestratorError("retained plan record does not match its digest")
        _digest(self.source_artifact_digest, "source artifact digest")
        if self.schema_version != ORCHESTRATOR_SCHEMA_VERSION:
            raise OptimizeOrchestratorError("unsupported orchestrator schema version")
        if self.cursor < 0 or self.cursor > len(STAGE_ORDER):
            raise OptimizeOrchestratorError("workflow cursor is out of bounds")
        if tuple(item.stage for item in self.stages) != STAGE_ORDER:
            raise OptimizeOrchestratorError("workflow stages are incomplete or out of order")
        if self.accepted_artifact_digest == self.source_artifact_digest:
            raise OptimizeOrchestratorError("accepted artifact must differ from source")
        if self.accepted_artifact_digest is not None:
            _digest(self.accepted_artifact_digest, "accepted artifact digest")
        if self.alternatives != tuple(sorted(set(self.alternatives))):
            raise OptimizeOrchestratorError("workflow alternatives must be sorted and unique")
        if any(not item.strip() for item in self.reasons):
            raise OptimizeOrchestratorError("workflow reasons must be non-empty")

    @property
    def terminal(self) -> bool:
        return self.status in {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED}

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "autonomous_optimize_run",
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "plan_record": dict(self.plan_record),
            "source_artifact_digest": self.source_artifact_digest,
            "status": self.status.value,
            "outcome": self.outcome.value,
            "cursor": self.cursor,
            "stages": [item.to_record() for item in self.stages],
            "approvals": [item.to_record() for item in self.approvals],
            "overrides": [item.to_record() for item in self.overrides],
            "accepted_artifact_digest": self.accepted_artifact_digest,
            "alternatives": list(self.alternatives),
            "reasons": list(self.reasons),
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


@dataclass(frozen=True, slots=True)
class StageContext:
    run: OptimizeRun
    plan: OptimizePlan
    stage: OptimizeStage
    completed_results: Mapping[OptimizeStage, StageResult]
    overrides: Mapping[str, str]


@runtime_checkable
class OptimizeRuntime(Protocol):
    """Trusted model-specific implementation behind the orchestration boundary."""

    def run_stage(self, context: StageContext) -> StageResult: ...


class OptimizeInterrupted(Exception):
    """Raised by a runtime to request a durable pause before retrying a stage."""


class PreflightRuntime:
    """Safe default runtime: validates the workflow but never claims measurements."""

    def run_stage(self, context: StageContext) -> StageResult:
        if context.stage in {
            OptimizeStage.PROFILE,
            OptimizeStage.CAPABILITY,
            OptimizeStage.BASELINE,
            OptimizeStage.CANDIDATE_GENERATION,
        }:
            return StageResult(
                WorkflowOutcome.SUPPORTED,
                f"preflight_{context.stage.value}",
                "stage inputs and bounds validated; no model artifact was changed",
                complete=True,
            )
        return StageResult(
            WorkflowOutcome.UNKNOWN,
            f"runtime_required_{context.stage.value}",
            "no trusted model runtime is registered for this execution stage",
            complete=True,
        )


class OptimizeStateStore:
    """Atomic single-run JSON store with strict identity matching on resume."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().absolute().resolve(strict=False)

    def load(self) -> OptimizeRun:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OptimizeOrchestratorError("orchestrator state is missing or corrupt") from error
        return _run_from_record(raw)

    def save(self, run: OptimizeRun) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.partial")
        temporary.write_text(run.canonical_json() + "\n", encoding="utf-8", newline="\n")
        try:
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()


def _source_digest(plan: OptimizePlan) -> str:
    payload = {"plan_id": plan.plan_id, "source_model": plan.resolved_config.get("model", {})}
    return "sha256:" + hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _run_id(plan: OptimizePlan) -> str:
    return (
        "optimize_run_"
        + hashlib.sha256(
            _canonical(
                {"schema_version": ORCHESTRATOR_SCHEMA_VERSION, "plan_id": plan.plan_id}
            ).encode()
        ).hexdigest()
    )


def _new_run(plan: OptimizePlan) -> OptimizeRun:
    stages = tuple(
        StageRecord(
            stage,
            f"stage_{stage.value}",
            tuple(STAGE_ORDER[:index]),
        )
        for index, stage in enumerate(STAGE_ORDER)
    )
    return OptimizeRun(
        _run_id(plan),
        plan.plan_id,
        plan_digest(plan),
        plan.to_record(),
        _source_digest(plan),
        WorkflowStatus.RUNNING,
        WorkflowOutcome.UNKNOWN,
        0,
        stages,
        (),
        (),
        None,
        (),
        (),
    )


def _record_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise OptimizeOrchestratorError(f"stored {label} must be an object")
    return value


def _stored_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise OptimizeOrchestratorError(f"stored {label} must be an integer")
    return value


def _stored_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise OptimizeOrchestratorError(f"stored {label} must be a boolean")
    return value


def _stored_strings(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise OptimizeOrchestratorError(f"stored {label} must be a string array")
    return tuple(value)


def _run_from_record(value: object) -> OptimizeRun:
    raw = _record_mapping(value, "run")
    if raw.get("record_type") != "autonomous_optimize_run":
        raise OptimizeOrchestratorError("stored state is not an autonomous optimize run")
    if raw.get("schema_version") != ORCHESTRATOR_SCHEMA_VERSION:
        raise OptimizeOrchestratorError("stored orchestrator schema is unsupported")
    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list):
        raise OptimizeOrchestratorError("stored stages must be an array")
    stages: list[StageRecord] = []
    for item in stages_raw:
        record = _record_mapping(item, "stage")
        try:
            stage = OptimizeStage(str(record["stage"]))
            status = StageStatus(str(record["status"]))
        except (KeyError, ValueError) as error:
            raise OptimizeOrchestratorError("stored stage identity or status is invalid") from error
        result_raw = record.get("result")
        result = None if result_raw is None else _stage_result_from_record(result_raw)
        dependencies_raw = record.get("dependencies")
        if not isinstance(dependencies_raw, list):
            raise OptimizeOrchestratorError("stored stage dependencies must be an array")
        try:
            dependencies = tuple(OptimizeStage(str(item)) for item in dependencies_raw)
            attempts = _stored_int(record["attempts"], "stage attempts")
        except (KeyError, TypeError, ValueError) as error:
            raise OptimizeOrchestratorError("stored stage fields are invalid") from error
        stages.append(
            StageRecord(
                stage,
                str(record.get("stage_id")),
                dependencies,
                status,
                attempts,
                result,
            )
        )
    approvals_raw = raw.get("approvals", [])
    overrides_raw = raw.get("overrides", [])
    if not isinstance(approvals_raw, list) or not isinstance(overrides_raw, list):
        raise OptimizeOrchestratorError("stored approvals and overrides must be arrays")
    approvals = tuple(
        ApprovalRecord(
            str(item["code"]),
            _stored_bool(item["approved"], "approval status"),
            str(item["recorded_by"]),
            str(item["plan_id"]),
            str(item["plan_digest"]),
            str(item["diff_id"]),
            str(item["expires_at"]),
            tuple(
                (str(key), str(value))
                for key, value in sorted(
                    _record_mapping(item.get("operator_context", {}), "operator context").items()
                )
            ),
        )
        for item in (_record_mapping(item, "approval") for item in approvals_raw)
    )
    overrides = tuple(
        OverrideRecord(
            str(item["name"]),
            str(item["value"]),
            _stored_bool(item["approved"], "override status"),
            None if item.get("approval_id") is None else str(item["approval_id"]),
        )
        for item in (_record_mapping(item, "override") for item in overrides_raw)
    )
    try:
        return OptimizeRun(
            str(raw["run_id"]),
            str(raw["plan_id"]),
            str(raw["plan_digest"]),
            _record_mapping(raw["plan_record"], "plan record"),
            str(raw["source_artifact_digest"]),
            WorkflowStatus(str(raw["status"])),
            WorkflowOutcome(str(raw["outcome"])),
            _stored_int(raw["cursor"], "cursor"),
            tuple(stages),
            approvals,
            overrides,
            None
            if raw.get("accepted_artifact_digest") is None
            else str(raw["accepted_artifact_digest"]),
            _stored_strings(raw.get("alternatives", []), "alternatives"),
            _stored_strings(raw.get("reasons", []), "reasons"),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise OptimizeOrchestratorError("stored run fields are invalid") from error


def _stage_result_from_record(value: object) -> StageResult:
    raw = _record_mapping(value, "stage result")
    alternatives = raw.get("alternatives", [])
    if not isinstance(alternatives, list):
        raise OptimizeOrchestratorError("stored stage alternatives must be an array")
    transaction_raw = raw.get("transaction_state")
    try:
        return StageResult(
            WorkflowOutcome(str(raw["outcome"])),
            str(raw["evidence_id"]),
            str(raw["detail"]),
            _stored_bool(raw.get("measured", False), "measurement status"),
            _stored_bool(raw.get("complete", False), "completion status"),
            _stored_bool(raw.get("constraints_passed", False), "constraint status"),
            None if raw.get("artifact_digest") is None else str(raw["artifact_digest"]),
            None if raw.get("candidate_id") is None else str(raw["candidate_id"]),
            None if raw.get("candidate_state_id") is None else str(raw["candidate_state_id"]),
            None if raw.get("evaluation_id") is None else str(raw["evaluation_id"]),
            None if transaction_raw is None else TransactionState(str(transaction_raw)),
            _stored_bool(raw.get("artifact_immutable", False), "artifact immutability status"),
            tuple(str(item) for item in alternatives),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise OptimizeOrchestratorError("stored stage result is invalid") from error


def _approval_codes(plan: OptimizePlan) -> tuple[str, ...]:
    return tuple(item.code for item in plan.approvals if item.required)


def load_optimize_runtime(specification: str) -> OptimizeRuntime:
    """Load a runtime factory from ``module:factory`` and validate its protocol."""

    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise OptimizeOrchestratorError("runtime must use module:factory syntax")
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise OptimizeOrchestratorError(
            f"cannot load optimize runtime {specification!r}"
        ) from error
    if not callable(factory):
        raise OptimizeOrchestratorError("optimize runtime factory must be callable")
    try:
        runtime = factory()
    except Exception as error:
        raise OptimizeOrchestratorError("optimize runtime factory failed") from error
    if not isinstance(runtime, OptimizeRuntime):
        raise OptimizeOrchestratorError("optimize runtime does not implement run_stage(context)")
    return runtime


class OptimizeOrchestrator:
    """Execute one deterministic workflow while retaining every decision."""

    def __init__(self, plan: OptimizePlan, state_path: str | Path) -> None:
        if plan.outcome is not OptimizeOutcome.SUPPORTED:
            raise OptimizeOrchestratorError(
                f"cannot execute optimize plan with outcome {plan.outcome.value}"
            )
        self.plan = plan
        self.store = OptimizeStateStore(state_path)

    def _load_or_start(self, *, resume: bool) -> OptimizeRun:
        if self.store.path.exists():
            if not resume:
                raise OptimizeOrchestratorError("state exists; pass --resume to continue it")
            run = self.store.load()
            if (
                run.plan_id != self.plan.plan_id
                or run.plan_digest != plan_digest(self.plan)
                or run.source_artifact_digest != _source_digest(self.plan)
            ):
                changed = diff_plans(run.plan_record, self.plan)
                raise OptimizeOrchestratorError(
                    "resume state does not match the supplied optimize plan; "
                    f"material_diff={changed.material} diff_id={changed.diff_id} "
                    f"changed_paths={','.join(changed.changed_paths) or 'none'}"
                )
            return run
        if resume:
            raise OptimizeOrchestratorError("cannot resume missing optimize state")
        run = _new_run(self.plan)
        self.store.save(run)
        return run

    def _record_inputs(
        self,
        run: OptimizeRun,
        approvals: Sequence[str],
        overrides: Mapping[str, str],
        approval_expires_at: str | None,
        operator_id: str,
        operator_context: Mapping[str, str],
    ) -> OptimizeRun:
        required = set(_approval_codes(self.plan))
        existing = {item.code: item for item in run.approvals}
        current_plan_digest = plan_digest(self.plan)
        initial_diff_id = diff_plans(self.plan, self.plan).diff_id
        expiry = approval_expires_at or (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        context = tuple(sorted((str(key), str(value)) for key, value in operator_context.items()))
        for code in approvals:
            if code not in {item.code for item in self.plan.approvals} and not code.startswith(
                "override:"
            ):
                raise OptimizeOrchestratorError(f"unknown approval code: {code}")
            prior = existing.get(code)
            if prior is not None:
                if prior.plan_digest != current_plan_digest or prior.diff_id != initial_diff_id:
                    raise OptimizeOrchestratorError(
                        f"approval {code} is bound to a different plan diff"
                    )
                continue
            existing[code] = ApprovalRecord(
                code,
                True,
                operator_id,
                self.plan.plan_id,
                current_plan_digest,
                initial_diff_id,
                expiry,
                context,
            )
        recorded_approvals = tuple(existing[key] for key in sorted(existing))
        override_records: list[OverrideRecord] = list(run.overrides)
        for name, value in sorted(overrides.items()):
            approval = existing.get(f"override:{name}")
            candidate = OverrideRecord(
                name,
                value,
                approval is not None and approval.approved,
                None if approval is None else approval.approval_id,
            )
            if candidate not in override_records:
                override_records.append(candidate)
        missing = sorted(
            required
            - {
                item.code
                for item in recorded_approvals
                if item.approved
                and item.plan_id == self.plan.plan_id
                and item.plan_digest == current_plan_digest
                and item.active
            }
        )
        if missing:
            return replace(
                run,
                status=WorkflowStatus.PAUSED,
                outcome=WorkflowOutcome.UNKNOWN,
                reasons=tuple(
                    sorted(
                        set((*run.reasons, "required approvals are missing: " + ", ".join(missing)))
                    )
                ),
                approvals=recorded_approvals,
                overrides=tuple(sorted(override_records, key=lambda item: item.name)),
            )
        latest_overrides: dict[str, OverrideRecord] = {}
        for item in override_records:
            latest_overrides[item.name] = item
        unapproved_overrides = sorted(
            name for name, item in latest_overrides.items() if not item.approved
        )
        if unapproved_overrides:
            return replace(
                run,
                status=WorkflowStatus.PAUSED,
                outcome=WorkflowOutcome.UNKNOWN,
                reasons=tuple(
                    sorted(
                        set(
                            (
                                *run.reasons,
                                "overrides require explicit approval: "
                                + ", ".join(unapproved_overrides),
                            )
                        )
                    )
                ),
                approvals=recorded_approvals,
                overrides=tuple(sorted(override_records, key=lambda item: item.name)),
            )
        return replace(
            run,
            status=WorkflowStatus.RUNNING,
            approvals=recorded_approvals,
            overrides=tuple(sorted(override_records, key=lambda item: item.name)),
        )

    def run(
        self,
        runtime: OptimizeRuntime | None = None,
        *,
        resume: bool = False,
        approvals: Sequence[str] = (),
        overrides: Mapping[str, str] | None = None,
        approval_expires_at: str | None = None,
        operator_id: str = "cli",
        operator_context: Mapping[str, str] | None = None,
    ) -> OptimizeRun:
        run = self._load_or_start(resume=resume)
        run = self._record_inputs(
            run,
            approvals,
            {} if overrides is None else overrides,
            approval_expires_at,
            operator_id,
            {} if operator_context is None else operator_context,
        )
        self.store.save(run)
        if run.status is WorkflowStatus.PAUSED:
            return run
        if run.terminal:
            return run
        selected_runtime = runtime or PreflightRuntime()
        if not isinstance(selected_runtime, OptimizeRuntime):
            raise OptimizeOrchestratorError("runtime does not implement run_stage(context)")
        results = {
            item.stage: item.result
            for item in run.stages
            if item.status is StageStatus.COMPLETED and item.result is not None
        }
        for index, record in enumerate(run.stages):
            if record.status is StageStatus.COMPLETED:
                continue
            if record.dependencies and any(stage not in results for stage in record.dependencies):
                raise OptimizeOrchestratorError("workflow state has an incomplete stage dependency")
            running = replace(record, status=StageStatus.RUNNING, attempts=record.attempts + 1)
            run = replace(
                run,
                status=WorkflowStatus.RUNNING,
                cursor=index,
                stages=(*run.stages[:index], running, *run.stages[index + 1 :]),
            )
            self.store.save(run)
            context = StageContext(
                run,
                self.plan,
                record.stage,
                {key: value for key, value in results.items() if value is not None},
                {
                    name: item.value
                    for name, item in {
                        item.name: item for item in run.overrides if item.approved
                    }.items()
                },
            )
            try:
                result = selected_runtime.run_stage(context)
            except OptimizeInterrupted:
                paused = replace(
                    run,
                    status=WorkflowStatus.PAUSED,
                    outcome=WorkflowOutcome.UNKNOWN,
                    reasons=tuple(
                        sorted(set((*run.reasons, f"interrupted during {record.stage.value}")))
                    ),
                )
                self.store.save(paused)
                return paused
            except KeyboardInterrupt:
                paused = replace(
                    run,
                    status=WorkflowStatus.PAUSED,
                    outcome=WorkflowOutcome.UNKNOWN,
                    reasons=tuple(
                        sorted(set((*run.reasons, f"interrupted during {record.stage.value}")))
                    ),
                )
                self.store.save(paused)
                return paused
            if not isinstance(result, StageResult):
                raise OptimizeOrchestratorError("runtime returned a non-StageResult value")
            completed = replace(running, status=StageStatus.COMPLETED, result=result)
            results[record.stage] = result
            run = replace(
                run,
                stages=(*run.stages[:index], completed, *run.stages[index + 1 :]),
                cursor=index + 1,
            )
            self.store.save(run)
            if result.outcome is not WorkflowOutcome.SUPPORTED:
                terminal = replace(
                    run,
                    status=WorkflowStatus.COMPLETED,
                    outcome=result.outcome,
                    reasons=tuple(
                        sorted(set((*run.reasons, f"{record.stage.value}: {result.detail}")))
                    ),
                )
                self.store.save(terminal)
                return terminal
        final = results.get(OptimizeStage.PARETO)
        if final is None:
            raise OptimizeOrchestratorError("workflow completed without Pareto evidence")
        alternatives = final.alternatives or run.alternatives
        if not final.measured or not final.complete or not final.constraints_passed:
            terminal = replace(
                run,
                status=WorkflowStatus.COMPLETED,
                outcome=WorkflowOutcome.FAILED,
                alternatives=alternatives,
                reasons=tuple(
                    sorted(
                        set(
                            (
                                *run.reasons,
                                "no feasible measured candidate passed the final constraints",
                            )
                        )
                    )
                ),
            )
            self.store.save(terminal)
            return terminal
        if (
            final.candidate_id is None
            or final.candidate_state_id is None
            or final.evaluation_id is None
            or final.transaction_state is None
            or final.artifact_digest is None
        ):
            terminal = replace(
                run,
                status=WorkflowStatus.COMPLETED,
                outcome=WorkflowOutcome.FAILED,
                alternatives=alternatives,
                reasons=tuple(
                    sorted(
                        set(
                            (*run.reasons, "final evidence is incomplete; no artifact was promoted")
                        )
                    )
                ),
            )
            self.store.save(terminal)
            return terminal
        approved_plan = ApprovedCampaignPlan(
            self.plan.plan_id,
            run.source_artifact_digest,
            "objective_contract_" + self.plan.config_digest,
            "candidate_space_" + self.plan.config_digest,
            (final.candidate_id,),
            0,
            self.plan.budget.evaluations,
            "plan_review",
        )
        promotion = AutonomousCampaignCoordinator(approved_plan).promote(
            MeasuredCandidateEvidence(
                final.candidate_id,
                final.candidate_state_id,
                final.evaluation_id,
                final.artifact_digest,
                run.source_artifact_digest,
                final.measured,
                final.complete,
                final.constraints_passed,
                final.transaction_state,
                final.artifact_immutable,
                final.detail,
            )
        )
        if promotion.outcome is not PromotionOutcome.PROMOTED:
            terminal = replace(
                run,
                status=WorkflowStatus.COMPLETED,
                outcome=WorkflowOutcome.FAILED,
                alternatives=alternatives,
                reasons=tuple(
                    sorted(set((*run.reasons, f"promotion refused: {promotion.reason}")))
                ),
            )
            self.store.save(terminal)
            return terminal
        terminal = replace(
            run,
            status=WorkflowStatus.COMPLETED,
            outcome=WorkflowOutcome.SUPPORTED,
            accepted_artifact_digest=final.artifact_digest,
            alternatives=alternatives,
            reasons=tuple(
                sorted(
                    set((*run.reasons, "measured candidate promoted by the existing coordinator"))
                )
            ),
        )
        self.store.save(terminal)
        return terminal


__all__ = [
    "ORCHESTRATOR_SCHEMA_VERSION",
    "STAGE_ORDER",
    "ApprovalRecord",
    "OptimizeInterrupted",
    "OptimizeOrchestrator",
    "OptimizeOrchestratorError",
    "OptimizeRun",
    "OptimizeRuntime",
    "OptimizeStage",
    "OptimizeStateStore",
    "OverrideRecord",
    "PreflightRuntime",
    "StageContext",
    "StageRecord",
    "StageResult",
    "StageStatus",
    "WorkflowOutcome",
    "WorkflowStatus",
    "load_optimize_runtime",
]

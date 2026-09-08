"""Trusted execution adapter for the bounded conversational optimize slice.

The adapter is deliberately small: the conversational layer may submit an
already-confirmed :class:`SpecSubmission`, but planning, orchestration,
approval, measurement, and promotion remain owned by ModelSurgeon.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from modelsurgeon.config import (
    ObjectiveDirection as ConfigObjectiveDirection,
)
from modelsurgeon.config import (
    ObjectiveNormalization as ConfigObjectiveNormalization,
)
from modelsurgeon.config import (
    ObjectiveTermConfig,
    OptimizeMetric,
    Settings,
    TaskQualitySpecError,
    task_quality_from_spec,
)
from modelsurgeon.conversation.campaign import CanonicalCampaignRecorder
from modelsurgeon.conversation.campaign_state import (
    CampaignLifecycle,
    CampaignOutcome,
    CampaignState,
    CampaignStateError,
    CampaignStateStore,
)
from modelsurgeon.conversation.dispatcher import (
    ToolCancellationToken,
    ToolDispatcher,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionResponse,
)
from modelsurgeon.conversation.isolation import redact_secret_text
from modelsurgeon.conversation.tools import (
    DEFAULT_TOOL_CATALOG,
    JSONValue,
    ToolEvidenceStatus,
    ToolFailureCode,
    ToolOutcome,
    ToolRequest,
    ToolResult,
)
from modelsurgeon.experiments.optimization_package import ApprovalReuse, plan_digest
from modelsurgeon.first_party_optimize_runtime import build_first_party_optimize_runtime
from modelsurgeon.optimization import (
    OptimizeOutcome,
    OptimizePlan,
    OptimizePlanError,
    build_optimize_plan,
)
from modelsurgeon.optimization_orchestrator import (
    OptimizeInterrupted,
    OptimizeOrchestrator,
    OptimizeRuntime,
    OptimizeStage,
    StageContext,
    StageResult,
    WorkflowOutcome,
)

if TYPE_CHECKING:
    from modelsurgeon.search.objective_contract import ObjectiveContract
    from modelsurgeon.search.spec_preview import SpecPreview, SpecSubmission


class ChatExecutionError(ValueError):
    """Raised when a confirmed chat submission cannot cross the engine boundary."""

    def __init__(self, code: str, message: str, outcome: ToolOutcome = ToolOutcome.FAILED) -> None:
        self.code = code
        self.outcome = outcome
        super().__init__(message)


class ChatProgressStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"


@dataclass(frozen=True, slots=True)
class ChatProgressEvent:
    """One engine-owned stage transition suitable for streaming to a client."""

    sequence: int
    campaign_id: str
    plan_id: str
    stage: str
    status: ChatProgressStatus
    outcome: ToolOutcome
    detail: str
    evidence_id: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "chat_optimize_progress",
            "sequence": self.sequence,
            "campaign_id": self.campaign_id,
            "plan_id": self.plan_id,
            "stage": self.stage,
            "status": self.status.value,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True, slots=True)
class ChatExecutionRecord:
    """A chat-facing view over one typed tool result and engine progress."""

    operation: str
    outcome: ToolOutcome
    plan_id: str | None
    plan_digest: str | None
    campaign_id: str | None
    run_id: str | None
    artifact_id: str | None
    artifact_digest: str | None
    evidence_refs: tuple[str, ...]
    reasons: tuple[str, ...]
    progress: tuple[ChatProgressEvent, ...]
    tool_result: ToolResult

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "chat_optimize_execution",
            "operation": self.operation,
            "outcome": self.outcome.value,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "artifact_id": self.artifact_id,
            "artifact_digest": self.artifact_digest,
            "evidence_refs": list(self.evidence_refs),
            "reasons": list(self.reasons),
            "progress": [item.to_record() for item in self.progress],
            "tool_result": self.tool_result.to_record(),
        }


ProgressCallback = Callable[[ChatProgressEvent], None]


@dataclass(frozen=True, slots=True)
class _PreparedPlan:
    spec_digest: str
    plan: OptimizePlan | None
    outcome: ToolOutcome
    diagnostic: str
    plan_id: str
    plan_digest: str | None


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _identifier_digest(prefix: str, value: object) -> str:
    return prefix + _digest(value)[len("sha256:") :]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ChatExecutionError(
            "invalid_spec", f"{label} must be an object", ToolOutcome.UNSUPPORTED
        )
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ChatExecutionError(
            "invalid_spec", f"{label} must be non-empty text", ToolOutcome.UNSUPPORTED
        )
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChatExecutionError(
            "invalid_spec", f"{label} must be numeric", ToolOutcome.UNSUPPORTED
        )
    return float(value)


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _enum(value: object, enum_type: type[StrEnum], label: str) -> StrEnum:
    try:
        return enum_type(_text(value, label))
    except ValueError as error:
        raise ChatExecutionError(
            "unsupported_spec", f"{label} has an unsupported value", ToolOutcome.UNSUPPORTED
        ) from error


def _contract_from_submission(submission: SpecSubmission) -> ObjectiveContract:
    from modelsurgeon.search.objective_contract import (
        ConstraintDirection,
        HardConstraint,
        MetricUnit,
        ObjectiveApprovalPolicy,
        ObjectiveContract,
        ObjectiveDirection,
        ObjectiveNormalization,
        ObjectivePluginBinding,
        SoftObjective,
    )

    record = _mapping(submission.spec, "confirmed spec")
    if record.get("schema_version") != 1:
        raise ChatExecutionError(
            "unsupported_spec", "confirmed spec schema is not supported", ToolOutcome.UNSUPPORTED
        )
    constraints_raw = record.get("constraints")
    objectives_raw = record.get("objectives")
    if not isinstance(constraints_raw, list) or not isinstance(objectives_raw, list):
        raise ChatExecutionError(
            "invalid_spec",
            "confirmed spec must contain constraint and objective arrays",
            ToolOutcome.UNSUPPORTED,
        )
    constraints: list[HardConstraint] = []
    for item in constraints_raw:
        raw = _mapping(item, "hard constraint")
        constraints.append(
            HardConstraint(
                _text(raw.get("metric"), "constraint metric"),
                cast(
                    ConstraintDirection,
                    _enum(raw.get("direction"), ConstraintDirection, "constraint direction"),
                ),
                _number(raw.get("threshold"), "constraint threshold"),
                cast(MetricUnit, _enum(raw.get("unit"), MetricUnit, "constraint unit")),
                _text(raw.get("baseline", "absolute"), "constraint baseline"),
            )
        )
    objectives: list[SoftObjective] = []
    for item in objectives_raw:
        raw = _mapping(item, "soft objective")
        plugin_raw = raw.get("plugin")
        plugin = None
        if plugin_raw is not None:
            plugin_record = _mapping(plugin_raw, "objective plugin")
            plugin = ObjectivePluginBinding(
                _text(plugin_record.get("name"), "plugin name"),
                _text(plugin_record.get("plugin_id"), "plugin ID"),
                _text(plugin_record.get("capability"), "plugin capability"),
                _text(plugin_record.get("config_digest"), "plugin config digest"),
                _text(plugin_record.get("trust_mode", "subprocess"), "plugin trust mode"),
            )
        objectives.append(
            SoftObjective(
                _text(raw.get("metric"), "objective metric"),
                cast(
                    ObjectiveDirection,
                    _enum(raw.get("direction"), ObjectiveDirection, "objective direction"),
                ),
                cast(MetricUnit, _enum(raw.get("unit"), MetricUnit, "objective unit")),
                _number(raw.get("weight", 1.0), "objective weight"),
                cast(
                    ObjectiveNormalization,
                    _enum(
                        raw.get("normalization", "baseline_ratio"),
                        ObjectiveNormalization,
                        "objective normalization",
                    ),
                ),
                None
                if raw.get("baseline") is None
                else _number(raw["baseline"], "objective baseline"),
                None
                if raw.get("minimum") is None
                else _number(raw["minimum"], "objective minimum"),
                None
                if raw.get("maximum") is None
                else _number(raw["maximum"], "objective maximum"),
                plugin,
            )
        )
    approval_raw = _mapping(record.get("approval_policy", {}), "approval policy")
    allowed_plugins = approval_raw.get("allowed_plugin_names", [])
    if not isinstance(allowed_plugins, list) or any(
        not isinstance(item, str) for item in allowed_plugins
    ):
        raise ChatExecutionError(
            "invalid_spec", "approval policy plugin names are invalid", ToolOutcome.UNSUPPORTED
        )
    mode = _text(record.get("mode", "weighted"), "objective mode")
    if mode != "weighted":
        raise ChatExecutionError(
            "unsupported_spec",
            "only weighted objective contracts are supported by optimize",
            ToolOutcome.UNSUPPORTED,
        )
    try:
        contract = ObjectiveContract(
            tuple(constraints),
            tuple(objectives),
            approval_policy=ObjectiveApprovalPolicy(
                bool(approval_raw.get("require_execution_approval", True)),
                bool(approval_raw.get("require_custom_plugin_approval", True)),
                tuple(sorted(allowed_plugins)),
            ),
            schema_version=1,
        )
    except (TypeError, ValueError) as error:
        raise ChatExecutionError("invalid_spec", str(error), ToolOutcome.UNSUPPORTED) from error
    if contract.contract_id != record.get("contract_id"):
        raise ChatExecutionError(
            "invalid_spec",
            "confirmed spec identity does not match its contents",
            ToolOutcome.UNSUPPORTED,
        )
    return contract


_OBJECTIVE_METRICS: dict[str, OptimizeMetric] = {
    "quality": OptimizeMetric.QUALITY,
    "perplexity": OptimizeMetric.PERPLEXITY,
    "latency": OptimizeMetric.LATENCY,
    "latency_gain": OptimizeMetric.LATENCY,
    "parameter_count": OptimizeMetric.PARAMETER_COUNT,
    "peak_ram": OptimizeMetric.MEMORY,
    "peak_vram": OptimizeMetric.MEMORY,
    "file_size": OptimizeMetric.DISK_SIZE,
}


def _settings_for_contract(
    base: Settings,
    contract: ObjectiveContract,
    spec: Mapping[str, object] | None = None,
) -> Settings:
    from modelsurgeon.search.objective_contract import ConstraintDirection

    values = base.canonical_dict()
    if spec is not None:
        try:
            task_quality = task_quality_from_spec(spec)
        except TaskQualitySpecError as error:
            raise ChatExecutionError("invalid_spec", str(error), ToolOutcome.UNSUPPORTED) from error
        if task_quality is not None:
            dataset = task_quality.get("dataset")
            if not isinstance(dataset, str) or not Path(dataset).expanduser().is_file():
                raise ChatExecutionError(
                    "invalid_spec",
                    "task-quality benchmark dataset is not a readable local file",
                    ToolOutcome.UNSUPPORTED,
                )
            # The conversational intent currently identifies the benchmark
            # dataset, while the target config owns the bounded execution
            # parameters. Preserve those explicit config values when the
            # interpreter emitted its canonical defaults.
            configured_task_quality = cast(
                Mapping[str, object], values.get("task_quality", {})
            )
            if configured_task_quality.get("method") != "none":
                defaults = {
                    "dataset_revision": None,
                    "split": "test",
                    "max_new_tokens": 128,
                    "max_samples": None,
                }
                for key, default in defaults.items():
                    if (
                        task_quality.get(key) == default
                        and configured_task_quality.get(key) != default
                    ):
                        task_quality[key] = configured_task_quality[key]
            values["task_quality"] = task_quality
    constraints = dict(cast(Mapping[str, object], values["constraints"]))
    objectives = dict(cast(Mapping[str, object], values["objective"]))
    mapped_constraints: dict[str, str] = {
        "quality": "min_quality_retention_ratio",
        "perplexity": "max_perplexity_delta",
        "latency_gain": "min_latency_gain_ratio",
        "peak_ram": "max_ram_bytes",
        "peak_vram": "max_vram_bytes",
        "file_size": "max_disk_bytes",
    }
    for constraint in contract.constraints:
        field = mapped_constraints.get(constraint.metric)
        if field is None:
            raise ChatExecutionError(
                "unsupported_spec",
                f"hard constraint metric {constraint.metric!r} is not supported by optimize",
                ToolOutcome.UNSUPPORTED,
            )
        if constraint.baseline != "absolute":
            raise ChatExecutionError(
                "unsupported_spec",
                "relative hard constraints require a measured baseline at the engine boundary",
                ToolOutcome.UNSUPPORTED,
            )
        expected_direction = (
            ConstraintDirection.MINIMUM
            if constraint.metric in {"quality", "latency_gain"}
            else ConstraintDirection.MAXIMUM
        )
        if constraint.direction is not expected_direction:
            raise ChatExecutionError(
                "unsupported_spec",
                f"constraint direction for {constraint.metric!r} cannot be represented safely",
                ToolOutcome.UNSUPPORTED,
            )
        constraints[field] = constraint.threshold
    terms: list[dict[str, object]] = []
    optimize_metrics: list[OptimizeMetric] = []
    for objective in contract.objectives:
        metric = _OBJECTIVE_METRICS.get(objective.metric)
        if metric is None or objective.plugin is not None:
            raise ChatExecutionError(
                "unsupported_spec",
                f"objective metric {objective.metric!r} is not supported by optimize",
                ToolOutcome.UNSUPPORTED,
            )
        if objective.baseline is not None:
            raise ChatExecutionError(
                "unsupported_spec",
                "baseline-relative objectives require a measured baseline at the engine boundary",
                ToolOutcome.UNSUPPORTED,
            )
        optimize_metrics.append(metric)
        terms.append(
            ObjectiveTermConfig(
                metric=metric,
                direction=ConfigObjectiveDirection(objective.direction.value),
                weight=objective.weight,
                normalization=ConfigObjectiveNormalization(objective.normalization.value),
                minimum=objective.minimum,
                maximum=objective.maximum,
            ).model_dump(mode="json")
        )
    objectives["optimize"] = tuple(optimize_metrics)
    objectives["terms"] = terms
    values["constraints"] = constraints
    values["objective"] = objectives
    try:
        return Settings.model_validate(values)
    except ValueError as error:
        raise ChatExecutionError("invalid_spec", str(error), ToolOutcome.UNSUPPORTED) from error


def _outcome(value: OptimizeOutcome | WorkflowOutcome) -> ToolOutcome:
    return ToolOutcome(value.value)


def _campaign_id(plan_id: str) -> str:
    return _identifier_digest("campaign_", {"plan_id": plan_id})


def _artifact_id(digest: str | None) -> str | None:
    return None if digest is None else _identifier_digest("artifact_", digest)


class _ProgressRuntime:
    def __init__(
        self,
        runtime: OptimizeRuntime,
        token: ToolCancellationToken,
        callback: ProgressCallback,
        campaign_id: str,
        plan_id: str,
        pause_requested: Callable[[], bool] | None = None,
    ) -> None:
        self.runtime = runtime
        self.token = token
        self.callback = callback
        self.campaign_id = campaign_id
        self.plan_id = plan_id
        self.pause_requested = pause_requested or (lambda: False)
        self.sequence = 0

    def _emit(
        self,
        stage: OptimizeStage,
        status: ChatProgressStatus,
        outcome: ToolOutcome,
        detail: str,
        evidence_id: str | None = None,
    ) -> None:
        self.sequence += 1
        self.callback(
            ChatProgressEvent(
                self.sequence,
                self.campaign_id,
                self.plan_id,
                stage.value,
                status,
                outcome,
                detail,
                evidence_id,
            )
        )

    def run_stage(self, context: StageContext) -> StageResult:
        if self.token.cancelled:
            raise OptimizeInterrupted()
        if self.pause_requested():
            raise OptimizeInterrupted()
        self._emit(
            context.stage,
            ChatProgressStatus.STARTED,
            ToolOutcome.UNKNOWN,
            "stage started",
        )
        result = self.runtime.run_stage(context)
        if self.token.cancelled:
            raise OptimizeInterrupted()
        if self.pause_requested():
            raise OptimizeInterrupted()
        self._emit(
            context.stage,
            ChatProgressStatus.COMPLETED,
            _outcome(result.outcome),
            result.detail,
            result.evidence_id,
        )
        return result


class ChatOptimizeAdapter:
    """Bridge confirmed conversational specs to the stable optimize APIs."""

    def __init__(
        self,
        settings: Settings,
        *,
        state_path: str | Path | None = None,
        campaign_state_path: str | Path | None = None,
        preset: str = "balanced",
        hardware_profile: str = "cpu-small",
        quality_profile: str | None = None,
        runtime: OptimizeRuntime | None = None,
        approvals: tuple[str, ...] = (),
        approval_expires_at: str | None = None,
        approval_reuse: Mapping[str, ApprovalReuse | str] | None = None,
        operator_id: str = "chat",
        resume: bool = False,
    ) -> None:
        if not isinstance(settings, Settings):
            raise ChatExecutionError("settings", "chat optimize adapter requires Settings")
        if runtime is not None and not isinstance(runtime, OptimizeRuntime):
            raise ChatExecutionError("runtime", "chat optimize runtime is not trusted")
        self.settings = settings
        self.state_path = None if state_path is None else Path(state_path)
        self.campaign_state_path: Path | None
        if campaign_state_path is not None:
            self.campaign_state_path = Path(campaign_state_path)
        elif self.state_path is not None:
            self.campaign_state_path = self.state_path.with_suffix(".campaign.sqlite3")
        else:
            self.campaign_state_path = None
        self.preset = preset
        self.hardware_profile = hardware_profile
        self.quality_profile = quality_profile
        # Runtime construction is deferred until a confirmed plan exists.  This
        # keeps model loading outside interpretation/preview and makes chat use
        # the same first-party engine as direct CLI execution by default.
        self.runtime = runtime
        self.approvals = tuple(sorted(set(approvals)))
        self.approval_expires_at = approval_expires_at
        self.approval_reuse = None if approval_reuse is None else dict(approval_reuse)
        self.operator_id = operator_id
        self._resume_by_default = resume
        self._previews: dict[str, object] = {}
        self._prepared: dict[str, _PreparedPlan] = {}
        self._submissions: dict[str, SpecSubmission] = {}
        self._progress: dict[str, list[ChatProgressEvent]] = {}
        self._progress_callbacks: dict[str, ProgressCallback] = {}
        self._resume_by_request: dict[str, bool] = {}
        self._provider_context_by_spec: dict[str, Mapping[str, object]] = {}
        self._session_by_request: dict[str, str] = {}
        self._active_tokens: dict[str, ToolCancellationToken] = {}
        self._pause_events: dict[str, threading.Event] = {}
        self._lifecycle_lock = threading.RLock()
        self._dispatcher = ToolDispatcher(
            {
                "preview_plan": self._preview_handler,
                "execute_approved_plan": self._execute_handler,
            },
            approval_policy=self._approval_policy,
        )

    def reconnect(self, campaign_id: str, session_id: str) -> CampaignState:
        """Recover canonical state by trusted IDs, never by replaying chat."""

        if self.campaign_state_path is None:
            raise ChatExecutionError("reconnect_failed", "campaign state path is required")
        try:
            with CampaignStateStore(self.campaign_state_path) as store:
                return store.reconnect(campaign_id, session_id)
        except CampaignStateError as error:
            raise ChatExecutionError("reconnect_failed", str(error)) from error

    def pause(
        self,
        campaign_id: str,
        session_id: str,
        *,
        operation_id: str = "pause",
        detail: str = "paused by operator",
    ) -> CampaignState:
        """Request cooperative pause and persist the lifecycle transition."""

        with self._lifecycle_lock:
            event = self._pause_events.get(campaign_id)
            if event is not None:
                event.set()
        if self.campaign_state_path is None:
            raise ChatExecutionError("pause_failed", "campaign state path is required")
        try:
            with CampaignStateStore(self.campaign_state_path) as store:
                return store.pause(
                    campaign_id,
                    session_id,
                    operation_id=operation_id,
                    detail=detail,
                )
        except CampaignStateError as error:
            raise ChatExecutionError("pause_failed", str(error)) from error

    def resume(
        self,
        campaign_id: str,
        session_id: str,
        *,
        operation_id: str = "resume",
    ) -> CampaignState:
        """Move a paused campaign back to runnable state after validation."""

        if self.campaign_state_path is None:
            raise ChatExecutionError("resume_failed", "campaign state path is required")
        try:
            with CampaignStateStore(self.campaign_state_path) as store:
                return store.resume(campaign_id, session_id, operation_id=operation_id)
        except CampaignStateError as error:
            raise ChatExecutionError("resume_failed", str(error)) from error

    def cancel(
        self,
        campaign_id: str,
        session_id: str,
        *,
        operation_id: str = "cancel",
        detail: str = "cancelled by operator",
    ) -> CampaignState:
        """Cancel cooperatively and make the terminal lifecycle durable."""

        with self._lifecycle_lock:
            token = self._active_tokens.get(campaign_id)
            if token is not None:
                token.cancel()
        if self.campaign_state_path is None:
            raise ChatExecutionError("cancel_failed", "campaign state path is required")
        try:
            with CampaignStateStore(self.campaign_state_path) as store:
                return store.cancel(
                    campaign_id,
                    session_id,
                    operation_id=operation_id,
                    detail=detail,
                )
        except CampaignStateError as error:
            raise ChatExecutionError("cancel_failed", str(error)) from error

    def restart(
        self,
        campaign_id: str,
        session_id: str,
        *,
        operation_id: str = "restart",
    ) -> CampaignState:
        """Reconnect after process loss and mark unfinished work runnable."""

        if self.campaign_state_path is None:
            raise ChatExecutionError("restart_failed", "campaign state path is required")
        try:
            with CampaignStateStore(self.campaign_state_path) as store:
                return store.restart(campaign_id, session_id, operation_id=operation_id)
        except CampaignStateError as error:
            raise ChatExecutionError("restart_failed", str(error)) from error

    def preview(
        self,
        session_id: str,
        request_id: str,
        preview: object,
        *,
        provider_context: Mapping[str, object] | None = None,
    ) -> ChatExecutionRecord:
        from modelsurgeon.search.spec_preview import SpecPreview

        if not isinstance(preview, SpecPreview):
            raise ChatExecutionError("preview", "chat execution requires a typed spec preview")
        if not preview.executable or preview.spec_digest is None:
            raise ChatExecutionError(
                "preview_not_executable",
                "optimization is blocked until the interpreted spec is executable",
                ToolOutcome.UNSUPPORTED,
            )
        self._previews[preview.spec_digest] = preview
        if provider_context is not None:
            self._provider_context_by_spec[preview.spec_digest] = dict(provider_context)
        definition = DEFAULT_TOOL_CATALOG.definition("preview_plan")
        if definition is None:  # pragma: no cover - frozen catalog invariant
            raise ChatExecutionError("catalog", "preview tool is unavailable")
        request = ToolRequest.create(
            definition,
            {"intent_id": preview.intent_id, "spec_digest": preview.spec_digest},
        )
        dispatched = self._dispatcher.dispatch(request)
        return self._record("preview_plan", dispatched.result, ())

    def execute(
        self,
        session_id: str,
        request_id: str,
        preview: object,
        approval_id: str | None,
        *,
        cancellation: ToolCancellationToken | None = None,
        progress_callback: ProgressCallback | None = None,
        resume: bool | None = None,
        provider_context: Mapping[str, object] | None = None,
    ) -> ChatExecutionRecord:
        from modelsurgeon.search.spec_preview import SpecPreview

        if not isinstance(preview, SpecPreview):
            raise ChatExecutionError("preview", "chat execution requires a typed spec preview")
        if preview.spec_digest is None:
            raise ChatExecutionError(
                "preview_not_executable",
                "optimization requires an executable spec",
                ToolOutcome.UNSUPPORTED,
            )
        if preview.spec_digest not in self._prepared:
            self.preview(
                session_id,
                request_id,
                preview,
                provider_context=provider_context,
            )
        elif provider_context is not None:
            self._provider_context_by_spec[preview.spec_digest] = dict(provider_context)
        submission = preview.confirm(approval_id=approval_id)
        prepared = self._prepared.get(preview.spec_digest)
        if prepared is None or prepared.plan is None:
            return self._record(
                "execute_approved_plan",
                self._failure_result(
                    request_id,
                    ToolOutcome.UNSUPPORTED,
                    ToolFailureCode.UNSUPPORTED_CAPABILITY,
                    prepared.diagnostic
                    if prepared is not None
                    else "optimization plan is unavailable",
                ),
                (),
            )
        self._submissions[prepared.plan_id] = submission
        progress: list[ChatProgressEvent] = []
        self._progress[prepared.plan_id] = progress

        def callback(event: ChatProgressEvent) -> None:
            progress.append(event)
            if progress_callback is not None:
                progress_callback(event)

        definition = DEFAULT_TOOL_CATALOG.definition("execute_approved_plan")
        if definition is None:  # pragma: no cover - frozen catalog invariant
            raise ChatExecutionError("catalog", "execution tool is unavailable")
        digest = cast(str, prepared.plan_digest)
        request = ToolRequest.create(
            definition,
            {
                "plan_id": prepared.plan_id,
                "plan_digest": digest,
                "approval_id": submission.approval_id,
            },
            approval_id=submission.approval_id,
        )
        self._session_by_request[request.request_id] = session_id
        self._progress[request.request_id] = progress
        self._progress_callbacks[request.request_id] = callback
        selected_resume = self._resume_by_default if resume is None else resume
        if selected_resume:
            # A resumed workflow is an explicit new request against the
            # durable state, not a replay of the earlier paused response.
            self._dispatcher = ToolDispatcher(
                {
                    "preview_plan": self._preview_handler,
                    "execute_approved_plan": self._execute_handler,
                },
                approval_policy=self._approval_policy,
            )
        self._resume_by_request[request.request_id] = selected_resume
        try:
            dispatched = self._dispatcher.dispatch(request, cancellation=cancellation)
        finally:
            self._resume_by_request.pop(request.request_id, None)
            self._progress_callbacks.pop(request.request_id, None)
            self._session_by_request.pop(request.request_id, None)
        return self._record("execute_approved_plan", dispatched.result, tuple(progress))

    def _preview_handler(self, context: ToolExecutionContext) -> ToolExecutionResponse:
        spec_digest = _text(context.request.input.get("spec_digest"), "spec digest")
        preview = self._previews.get(spec_digest)
        if preview is None:
            raise ToolExecutionError(
                ToolFailureCode.INVALID_INPUT, "spec preview is not registered"
            )
        try:
            submission = self._submission_for_preview(preview)
            contract = _contract_from_submission(submission)
            settings = _settings_for_contract(self.settings, contract, submission.spec)
            plan = build_optimize_plan(
                settings,
                preset=self.preset,
                hardware_profile=self.hardware_profile,
                quality_profile=self.quality_profile,
                dry_run=False,
            )
            prepared = _PreparedPlan(
                spec_digest,
                plan,
                _outcome(plan.outcome),
                "plan resolved",
                plan.plan_id,
                "sha256:" + plan_digest(plan),
            )
        except ChatExecutionError as error:
            prepared = _PreparedPlan(
                spec_digest,
                None,
                error.outcome,
                str(error),
                _identifier_digest("optimize_plan_unavailable_", spec_digest),
                None,
            )
        except (OptimizePlanError, ValueError) as error:
            prepared = _PreparedPlan(
                spec_digest,
                None,
                ToolOutcome.FAILED,
                str(error),
                _identifier_digest("optimize_plan_failed_", spec_digest),
                None,
            )
        self._prepared[spec_digest] = prepared
        output: dict[str, JSONValue] = {
            "plan_id": prepared.plan_id,
            "spec_digest": spec_digest,
            "status": prepared.outcome.value,
            "diagnostic": prepared.diagnostic,
            "evidence_refs": ["plan_evidence_" + spec_digest[len("sha256:") :]],
        }
        return ToolExecutionResponse(
            output,
            ToolEvidenceStatus.CANONICAL,
            spec_digest,
            "plan_evidence_" + spec_digest[len("sha256:") :],
            observed_at=_now(),
        )

    def _execute_handler(self, context: ToolExecutionContext) -> ToolExecutionResponse:
        plan_id = _text(context.request.input.get("plan_id"), "plan ID")
        prepared = next((item for item in self._prepared.values() if item.plan_id == plan_id), None)
        if prepared is None or prepared.plan is None or prepared.plan_digest is None:
            raise ToolExecutionError(
                ToolFailureCode.INVALID_INPUT, "plan was not prepared by this session"
            )
        if context.request.input.get("plan_digest") != prepared.plan_digest:
            raise ToolExecutionError(
                ToolFailureCode.APPROVAL_MISMATCH, "plan digest is not current"
            )
        context.check_cancelled()
        if self.campaign_state_path is None:
            raise ToolExecutionError(
                ToolFailureCode.INVALID_INPUT, "execution campaign state path is required"
            )
        progress = self._progress.get(context.request.request_id, [])
        callback = self._progress_callbacks.get(context.request.request_id, progress.append)
        state_path = self.state_path
        if state_path is None:
            raise ToolExecutionError(
                ToolFailureCode.INVALID_INPUT, "execution state path is required"
            )
        selected_resume = self._resume_by_request.get(
            context.request.request_id, self._resume_by_default
        )
        recorder = CanonicalCampaignRecorder(
            self.campaign_state_path,
            session_id=self._session_for_request(context.request.request_id),
            plan=prepared.plan,
            preview=self._preview_for_spec(prepared.spec_digest),
            provider_context=self._provider_context_by_spec.get(prepared.spec_digest),
            approval_id=self._submissions[prepared.plan_id].approval_id,
            recorded_by=self.operator_id,
            approval_expires_at=self.approval_expires_at,
            approval_reuse=self.approval_reuse,
        )
        campaign_id = recorder.campaign_id
        try:
            with self._lifecycle_lock:
                pause_event = threading.Event()
                self._active_tokens[campaign_id] = context.cancellation
                self._pause_events[campaign_id] = pause_event
            selected_runtime = self.runtime or build_first_party_optimize_runtime(prepared.plan)
            runtime = _ProgressRuntime(
                selected_runtime,
                context.cancellation,
                callback,
                campaign_id,
                plan_id,
                pause_event.is_set,
            )
            if prepared.plan.outcome is not OptimizeOutcome.SUPPORTED:
                detail = redact_secret_text(
                    f"optimize plan is {prepared.plan.outcome.value}; execution is unsupported"
                )
                _state, evidence_id = recorder.retain_outcome(
                    detail,
                    outcome=CampaignOutcome.UNSUPPORTED,
                    lifecycle=CampaignLifecycle.COMPLETED,
                    inconclusive=False,
                )
                return self._terminal_response(
                    context,
                    recorder,
                    ToolOutcome.UNSUPPORTED,
                    evidence_id,
                    detail,
                )
            recorder.start()
            try:
                run = OptimizeOrchestrator(prepared.plan, state_path).run(
                    runtime,
                    resume=selected_resume,
                    approvals=self.approvals,
                    approval_expires_at=self.approval_expires_at,
                    operator_id=self.operator_id,
                    operator_context={"campaign_id": campaign_id},
                    approval_reuse=self.approval_reuse,
                )
            except Exception as error:
                detail = redact_secret_text(str(error))
                _state, evidence_id = recorder.retain_failure(detail)
                return self._terminal_response(
                    context,
                    recorder,
                    ToolOutcome.FAILED,
                    evidence_id,
                    detail,
                )
            if context.cancellation.cancelled:
                _state, evidence_id = recorder.retain_outcome(
                    "campaign cancelled by operator",
                    outcome=CampaignOutcome.UNKNOWN,
                    lifecycle=CampaignLifecycle.CANCELLED,
                    inconclusive=True,
                )
                return self._terminal_response(
                    context,
                    recorder,
                    ToolOutcome.CANCELLED,
                    evidence_id,
                    "campaign cancelled by operator",
                )
            _state, campaign_evidence_refs = recorder.retain_run(run)
            context.check_cancelled()
            evidence_refs = tuple(dict.fromkeys(campaign_evidence_refs))
            output: dict[str, JSONValue] = {
                "run_id": run.run_id,
                "campaign_id": campaign_id,
                "plan_id": run.plan_id,
                "outcome": run.outcome.value,
                "evidence_refs": list(evidence_refs),
            }
            artifact_id = _artifact_id(run.accepted_artifact_digest)
            if artifact_id is not None:
                output["artifact_ref"] = artifact_id
                output["artifact_digest"] = run.accepted_artifact_digest
            if run.reasons:
                output["reasons"] = list(run.reasons)
            context.transaction.commit()
            return ToolExecutionResponse(
                output,
                ToolEvidenceStatus.CANONICAL,
                _digest(run.to_record()),
                campaign_evidence_refs[-1],
                artifact_id=artifact_id,
                campaign_id=campaign_id,
                observed_at=_now(),
            )
        finally:
            with self._lifecycle_lock:
                self._active_tokens.pop(campaign_id, None)
                self._pause_events.pop(campaign_id, None)

    @staticmethod
    def _terminal_response(
        context: ToolExecutionContext,
        recorder: CanonicalCampaignRecorder,
        outcome: ToolOutcome,
        evidence_id: str,
        detail: str,
    ) -> ToolExecutionResponse:
        context.transaction.commit()
        return ToolExecutionResponse(
            {
                "run_id": recorder.run_id,
                "campaign_id": recorder.campaign_id,
                "plan_id": recorder.plan.plan_id,
                "outcome": outcome.value,
                "evidence_refs": [evidence_id],
                "reasons": [detail],
            },
            ToolEvidenceStatus.CANONICAL,
            recorder.source_model_digest,
            evidence_id,
            campaign_id=recorder.campaign_id,
            observed_at=_now(),
        )

    def _approval_policy(self, request: ToolRequest) -> bool:
        plan_id = request.input.get("plan_id")
        plan_digest_value = request.input.get("plan_digest")
        approval = request.input.get("approval_id")
        prepared = next((item for item in self._prepared.values() if item.plan_id == plan_id), None)
        submission = None if prepared is None else self._submissions.get(prepared.plan_id)
        if prepared is None or prepared.plan is None or submission is None:
            return False
        required = {item.code for item in prepared.plan.approvals if item.required}
        expiry_ok = True
        if self.approval_expires_at is not None:
            try:
                expiry = datetime.fromisoformat(self.approval_expires_at.replace("Z", "+00:00"))
                expiry_ok = expiry.tzinfo is not None and expiry.astimezone(UTC) > datetime.now(UTC)
            except ValueError:
                expiry_ok = False
        return (
            plan_digest_value == prepared.plan_digest
            and approval == request.approval_id == submission.approval_id
            and bool(request.approval_id)
            and required.issubset(self.approvals)
            and expiry_ok
        )

    def _submission_for_preview(self, preview: object) -> SpecSubmission:
        from modelsurgeon.search.spec_preview import SpecPreview

        if not isinstance(preview, SpecPreview) or preview.spec_digest is None:
            raise ChatExecutionError(
                "preview", "preview is not executable", ToolOutcome.UNSUPPORTED
            )
        return preview.confirm(approval_id="preview_only")

    def _preview_for_spec(self, spec_digest: str) -> SpecPreview:
        from modelsurgeon.search.spec_preview import SpecPreview

        preview = self._previews.get(spec_digest)
        if not isinstance(preview, SpecPreview):
            raise ChatExecutionError("preview", "spec preview is not registered")
        return preview

    def _session_for_request(self, request_id: str) -> str:
        session_id = self._session_by_request.get(request_id)
        if session_id is None:
            raise ChatExecutionError("session", "execution request has no trusted session identity")
        return session_id

    @staticmethod
    def _failure_result(
        request_id: str,
        outcome: ToolOutcome,
        code: ToolFailureCode,
        detail: str,
    ) -> ToolResult:
        from modelsurgeon.conversation.tools import ToolFailure, ToolProvenance

        request_digest = _digest({"request_id": request_id})
        return ToolResult(
            request_id,
            "execute_approved_plan",
            outcome,
            ToolProvenance(
                "modelsurgeon.conversation",
                "unknown",
                request_digest,
                evidence_status=ToolEvidenceStatus.UNAVAILABLE,
            ),
            failure=ToolFailure(code, detail, request_id),
        )

    def _record(
        self,
        operation: str,
        result: ToolResult | None,
        progress: tuple[ChatProgressEvent, ...],
    ) -> ChatExecutionRecord:
        if result is None:
            raise ChatExecutionError("missing_result", "optimizer adapter returned no typed result")
        output = {} if result.output is None else dict(result.output)
        outcome = result.outcome
        raw_outcome = output.get("outcome", output.get("status"))
        if result.outcome is ToolOutcome.SUPPORTED and isinstance(raw_outcome, str):
            try:
                outcome = ToolOutcome(raw_outcome)
            except ValueError:
                outcome = ToolOutcome.FAILED
        evidence_refs_raw = output.get("evidence_refs", [])
        evidence_refs = (
            tuple(item for item in evidence_refs_raw if isinstance(item, str))
            if isinstance(evidence_refs_raw, list)
            else ()
        )
        reasons_raw = output.get("reasons", [])
        reasons = (
            tuple(item for item in reasons_raw if isinstance(item, str))
            if isinstance(reasons_raw, list)
            else ()
        )
        return ChatExecutionRecord(
            operation,
            outcome,
            _optional_text(output.get("plan_id")),
            _optional_text(output.get("plan_digest")),
            (_optional_text(output.get("campaign_id")) or result.provenance.campaign_id),
            _optional_text(output.get("run_id")),
            (_optional_text(output.get("artifact_ref")) or result.provenance.artifact_id),
            (_optional_text(output.get("artifact_digest"))),
            evidence_refs,
            reasons,
            progress,
            result,
        )


__all__ = [
    "ChatExecutionError",
    "ChatExecutionRecord",
    "ChatOptimizeAdapter",
    "ChatProgressEvent",
    "ChatProgressStatus",
    "ProgressCallback",
]

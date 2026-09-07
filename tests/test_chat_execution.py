"""End-to-end contract tests for the chat-to-optimize execution adapter."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from modelsurgeon.config import (
    ConstraintConfig,
    ModelConfig,
    ObjectiveConfig,
    ObjectiveTermConfig,
    OptimizeMetric,
    Settings,
)
from modelsurgeon.config import (
    ObjectiveDirection as ConfigObjectiveDirection,
)
from modelsurgeon.config import (
    ObjectiveNormalization as ConfigObjectiveNormalization,
)
from modelsurgeon.conversation import (
    ChatOptimizeAdapter,
    ToolCancellationToken,
    ToolOutcome,
)
from modelsurgeon.optimization import OptimizeOutcome, build_optimize_plan
from modelsurgeon.optimization_orchestrator import (
    OptimizeInterrupted,
    OptimizeStage,
    StageContext,
    StageResult,
    WorkflowOutcome,
)
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    HardConstraint,
    MetricUnit,
    ObjectiveApprovalPolicy,
    ObjectiveContract,
    SoftObjective,
)
from modelsurgeon.search.objective_contract import (
    ObjectiveDirection as ContractObjectiveDirection,
)
from modelsurgeon.search.objective_contract import (
    ObjectiveNormalization as ContractObjectiveNormalization,
)
from modelsurgeon.search.spec_preview import SpecPreview, _digest, build_spec_preview
from modelsurgeon.surgery.contracts import TransactionState
from test_chat import _intent

_APPROVALS = ("artifact_write", "plan_review", "resource_budget", "source_model")


class _Runtime:
    def __init__(self, *, feasible: bool = True, interrupt_once: bool = False) -> None:
        self.feasible = feasible
        self.interrupt_once = interrupt_once
        self.calls: list[OptimizeStage] = []

    def run_stage(self, context: StageContext) -> StageResult:
        self.calls.append(context.stage)
        if self.interrupt_once and context.stage is OptimizeStage.SURGERY:
            self.interrupt_once = False
            raise OptimizeInterrupted()
        if context.stage is OptimizeStage.PARETO:
            return StageResult(
                WorkflowOutcome.SUPPORTED,
                "evaluation_chat_fixture",
                "measured final candidate evidence",
                measured=True,
                complete=True,
                constraints_passed=self.feasible,
                artifact_digest="sha256:" + "b" * 64,
                candidate_id="candidate_chat_fixture",
                candidate_state_id="state_chat_fixture",
                evaluation_id="evaluation_chat_fixture",
                transaction_state=TransactionState.COMMITTED,
                artifact_immutable=True,
                alternatives=("candidate_chat_alternative",),
            )
        return StageResult(
            WorkflowOutcome.SUPPORTED,
            f"evidence_{context.stage.value}",
            "bounded stage completed",
            complete=True,
        )


def _adapter(tmp_path: Path, runtime: _Runtime, *, resume: bool = False) -> ChatOptimizeAdapter:
    return ChatOptimizeAdapter(
        Settings(model=ModelConfig(path="models/tiny", revision="revision-1")),
        state_path=tmp_path / "chat-run.json",
        runtime=runtime,
        approvals=_APPROVALS,
        resume=resume,
    )


def _preview() -> SpecPreview:
    return build_spec_preview(_intent("retain quality and reduce latency"))


def test_chat_execution_matches_direct_plan_and_returns_canonical_ids(tmp_path: Path) -> None:
    runtime = _Runtime()
    adapter = _adapter(tmp_path, runtime)
    preview = _preview()

    planned = adapter.preview("chat-session", "chat-request", preview)
    executed = adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    assert planned.outcome is ToolOutcome.SUPPORTED
    assert planned.plan_id == executed.plan_id
    assert executed.outcome is ToolOutcome.SUPPORTED
    assert executed.campaign_id is not None and executed.campaign_id.startswith("campaign_")
    assert executed.run_id is not None and executed.run_id.startswith("optimize_run_")
    assert executed.artifact_id is not None and executed.artifact_id.startswith("artifact_")
    assert executed.evidence_refs
    assert executed.progress[0].status.value == "started"
    assert executed.progress[-1].status.value == "completed"
    assert executed.tool_result.provenance.evidence_status.value == "canonical"

    # The chat adapter and direct caller use the same stable plan identity and
    # therefore the same deterministic workflow/campaign identity.
    direct_plan = build_optimize_plan(
        Settings(
            model=ModelConfig(path="models/tiny", revision="revision-1"),
            constraints=ConstraintConfig(min_quality_retention_ratio=0.95),
            objective=ObjectiveConfig(
                optimize=(OptimizeMetric.LATENCY,),
                terms=(
                    ObjectiveTermConfig(
                        metric=OptimizeMetric.LATENCY,
                        direction=ConfigObjectiveDirection.MINIMIZE,
                        normalization=ConfigObjectiveNormalization.IDENTITY,
                    ),
                ),
            ),
        ),
        dry_run=False,
    )
    assert direct_plan.outcome is OptimizeOutcome.SUPPORTED
    assert planned.plan_id == direct_plan.plan_id


def test_chat_execution_rejects_missing_plan_approvals_without_runtime(tmp_path: Path) -> None:
    runtime = _Runtime()
    adapter = ChatOptimizeAdapter(
        Settings(model=ModelConfig(path="models/tiny", revision="revision-1")),
        state_path=tmp_path / "chat-run.json",
        runtime=runtime,
        approvals=(),
    )
    preview = _preview()
    adapter.preview("chat-session", "chat-request", preview)

    result = adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    assert result.outcome is ToolOutcome.REFUSED
    assert result.tool_result.failure is not None
    assert result.tool_result.failure.code.value == "approval_invalid"
    assert runtime.calls == []
    assert not (tmp_path / "chat-run.json").exists()


def test_chat_execution_rejects_relative_constraint_baselines(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, _Runtime())
    preview = _preview()
    assert preview.spec is not None
    relative_constraint = HardConstraint(
        "quality",
        ConstraintDirection.MINIMUM,
        0.95,
        MetricUnit.RATIO,
        "baseline:measured",
    )
    spec = dict(preview.spec)
    spec["constraints"] = [relative_constraint.to_record()]
    objective = cast(list[object], spec["objectives"])[0]
    assert isinstance(objective, dict)
    approval = cast(object, spec["approval_policy"])
    assert isinstance(approval, dict)
    contract = ObjectiveContract(
        (relative_constraint,),
        (
            SoftObjective(
                cast(str, objective["metric"]),
                ContractObjectiveDirection(cast(str, objective["direction"])),
                MetricUnit(cast(str, objective["unit"])),
                cast(float, objective["weight"]),
                ContractObjectiveNormalization(cast(str, objective["normalization"])),
                cast(float | None, objective["baseline"]),
                cast(float | None, objective["minimum"]),
                cast(float | None, objective["maximum"]),
            ),
        ),
        approval_policy=ObjectiveApprovalPolicy(
            cast(bool, approval["require_execution_approval"]),
            cast(bool, approval["require_custom_plugin_approval"]),
            tuple(cast(list[str], approval["allowed_plugin_names"])),
        ),
    )
    spec["contract_id"] = contract.contract_id
    invalid_preview = replace(preview, spec=spec, spec_digest=_digest(spec))

    result = adapter.preview("chat-session", "chat-request", invalid_preview)

    assert result.outcome is ToolOutcome.UNSUPPORTED
    assert result.tool_result.output is not None
    assert "relative hard constraints" in str(result.tool_result.output["diagnostic"])


def test_chat_execution_retains_failed_no_artifact_result(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, _Runtime(feasible=False))
    preview = _preview()
    adapter.preview("chat-session", "chat-request", preview)

    result = adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    assert result.outcome is ToolOutcome.FAILED
    assert result.artifact_id is None
    assert result.artifact_digest is None
    assert result.reasons
    assert result.tool_result.provenance.artifact_id is None


def test_chat_execution_pause_and_resume_keep_the_campaign_identity(tmp_path: Path) -> None:
    first_adapter = _adapter(tmp_path, _Runtime(interrupt_once=True))
    preview = _preview()
    first_adapter.preview("chat-session", "chat-request", preview)
    paused = first_adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    resumed_adapter = _adapter(tmp_path, _Runtime(), resume=True)
    resumed_adapter.preview("chat-session", "chat-request", preview)
    resumed = resumed_adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    assert paused.outcome is ToolOutcome.UNKNOWN
    assert resumed.outcome is ToolOutcome.SUPPORTED
    assert paused.campaign_id == resumed.campaign_id
    assert paused.artifact_id is None
    assert resumed.artifact_id is not None


def test_chat_execution_cancellation_is_typed_and_creates_no_state(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, _Runtime())
    preview = _preview()
    adapter.preview("chat-session", "chat-request", preview)
    cancellation = ToolCancellationToken()
    cancellation.cancel()

    result = adapter.execute(
        "chat-session", "chat-request", preview, "approval-chat", cancellation=cancellation
    )

    assert result.outcome is ToolOutcome.CANCELLED
    assert result.artifact_id is None
    assert not (tmp_path / "chat-run.json").exists()

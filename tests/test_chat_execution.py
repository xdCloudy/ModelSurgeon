"""End-to-end contract tests for the chat-to-optimize execution adapter."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

from modelsurgeon.adapters import ModelFormat
from modelsurgeon.config import (
    ConstraintConfig,
    ModelConfig,
    ObjectiveConfig,
    ObjectiveTermConfig,
    OptimizeMetric,
    Settings,
    TaskQualityConfig,
)
from modelsurgeon.config import (
    ObjectiveDirection as ConfigObjectiveDirection,
)
from modelsurgeon.config import (
    ObjectiveNormalization as ConfigObjectiveNormalization,
)
from modelsurgeon.conversation import (
    CampaignLifecycle,
    CampaignOutcome,
    CampaignStateStore,
    ChatOptimizeAdapter,
    ToolCancellationToken,
    ToolOutcome,
)
from modelsurgeon.conversation.execution import _settings_for_contract
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
    ContractMetric,
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


class _FailingRuntime(_Runtime):
    def run_stage(self, context: StageContext) -> StageResult:
        if context.stage is OptimizeStage.PROFILE:
            raise RuntimeError("runtime fixture failed")
        return super().run_stage(context)


def _adapter(
    tmp_path: Path,
    runtime: _Runtime,
    *,
    resume: bool = False,
    settings: Settings | None = None,
) -> ChatOptimizeAdapter:
    return ChatOptimizeAdapter(
        settings or Settings(model=ModelConfig(path="models/tiny", revision="revision-1")),
        state_path=tmp_path / "chat-run.json",
        runtime=runtime,
        approvals=_APPROVALS,
        resume=resume,
    )


def _preview() -> SpecPreview:
    return build_spec_preview(_intent("retain quality and reduce latency"))


def test_chat_contract_applies_explicit_task_quality_extension(tmp_path: Path) -> None:
    dataset = tmp_path / "coding.jsonl"
    dataset.write_text(
        '{"id":"one","prompt":"write a function","reference":"return 1"}\n',
        encoding="utf-8",
    )
    contract = ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                ConstraintDirection.MINIMUM,
                0.98,
                MetricUnit.RATIO,
            ),
        ),
        objectives=(
            SoftObjective(
                ContractMetric.LATENCY,
                ContractObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                normalization=ContractObjectiveNormalization.IDENTITY,
            ),
        ),
    )

    settings = _settings_for_contract(
        Settings(model=ModelConfig(path="models/tiny", revision="revision-1")),
        contract,
        {
            **contract.to_record(),
            "task_quality": {
                "method": "code_exact_match",
                "dataset": str(dataset),
                "dataset_revision": None,
                "split": "test",
                "max_new_tokens": 128,
                "max_samples": None,
            },
        },
    )

    assert settings.task_quality.method == "code_exact_match"
    assert settings.task_quality.dataset == dataset


def test_chat_contract_preserves_target_task_quality_execution_parameters(tmp_path: Path) -> None:
    dataset = tmp_path / "coding.jsonl"
    dataset.write_text(
        '{"id":"one","prompt":"write a function","reference":"return 1"}\n',
        encoding="utf-8",
    )
    configured = Settings(
        model=ModelConfig(path="models/tiny", revision="revision-1"),
        task_quality=TaskQualityConfig(
            method="code_exact_match",
            dataset=dataset,
            dataset_revision="benchmark-v1",
            split="validation",
            max_new_tokens=32,
            max_samples=4,
        ),
    )
    contract = ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                ConstraintDirection.MINIMUM,
                0.99,
                MetricUnit.RATIO,
            ),
        ),
        objectives=(
            SoftObjective(
                ContractMetric.LATENCY,
                ContractObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                normalization=ContractObjectiveNormalization.IDENTITY,
            ),
        ),
    )

    settings = _settings_for_contract(
        configured,
        contract,
        {
            **contract.to_record(),
            "task_quality": {
                "method": "code_exact_match",
                "dataset": str(dataset),
                "dataset_revision": None,
                "split": "test",
                "max_new_tokens": 128,
                "max_samples": None,
            },
        },
    )

    assert settings.task_quality.dataset_revision == "benchmark-v1"
    assert settings.task_quality.split == "validation"
    assert settings.task_quality.max_new_tokens == 32
    assert settings.task_quality.max_samples == 4


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


def test_chat_execution_uses_first_party_runtime_by_default(
    tmp_path: Path, monkeypatch: object
) -> None:
    runtime = _Runtime()
    captured: list[object] = []

    def factory(plan: object) -> _Runtime:
        captured.append(plan)
        return runtime

    monkeypatch.setattr(
        "modelsurgeon.conversation.execution.build_first_party_optimize_runtime",
        factory,
    )
    adapter = ChatOptimizeAdapter(
        Settings(model=ModelConfig(path="models/tiny", revision="revision-1")),
        state_path=tmp_path / "chat-run.json",
        approvals=_APPROVALS,
    )
    preview = _preview()
    adapter.preview("chat-session", "chat-request", preview)
    executed = adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    assert executed.outcome is ToolOutcome.SUPPORTED
    assert len(captured) == 1
    assert runtime.calls[0] is OptimizeStage.PROFILE


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


def test_accepted_execution_projects_canonical_campaign_evidence(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, _Runtime())
    preview = _preview()
    adapter.preview("chat-session", "chat-request", preview)

    result = adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    assert result.outcome is ToolOutcome.SUPPORTED
    assert result.campaign_id is not None
    with CampaignStateStore(tmp_path / "chat-run.campaign.sqlite3") as store:
        state = store.load(result.campaign_id)
        evidence = store.evidence(result.campaign_id)
    assert state.lifecycle is CampaignLifecycle.COMPLETED
    assert state.outcome is CampaignOutcome.SUPPORTED
    assert state.approval.status.value == "approved"
    assert tuple(item.evidence_id for item in evidence) == result.evidence_refs
    assert evidence[-1].artifact_digest is not None
    assert evidence[-1].provenance["decision"] == "accepted"
    assert result.tool_result.provenance.evidence_id == evidence[-1].evidence_id


def test_rejected_candidate_retains_negative_evidence_without_artifact(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, _Runtime(feasible=False))
    preview = _preview()
    adapter.preview("chat-session", "chat-request", preview)

    result = adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    assert result.outcome is ToolOutcome.FAILED
    assert result.artifact_id is None
    with CampaignStateStore(tmp_path / "chat-run.campaign.sqlite3") as store:
        state = store.load_for_session("chat-session")
        evidence = store.evidence(state.campaign_id)
    assert state.lifecycle is CampaignLifecycle.COMPLETED
    assert state.outcome is CampaignOutcome.FAILED
    assert evidence[-1].provenance["decision"] == "rejected"
    assert all(item.artifact_digest is None for item in evidence)


def test_unsupported_plan_is_retained_and_never_enters_the_runtime(
    tmp_path: Path,
) -> None:
    settings = Settings(
        model=ModelConfig(path="models/tiny.gguf", revision="revision-1", format=ModelFormat.GGUF)
    )
    runtime = _Runtime()
    adapter = _adapter(tmp_path, runtime, settings=settings)
    preview = _preview()
    adapter.preview("chat-session", "chat-request", preview)

    result = adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    assert result.outcome is ToolOutcome.UNSUPPORTED
    assert runtime.calls == []
    with CampaignStateStore(tmp_path / "chat-run.campaign.sqlite3") as store:
        state = store.load_for_session("chat-session")
        evidence = store.evidence(state.campaign_id)
    assert state.lifecycle is CampaignLifecycle.COMPLETED
    assert state.outcome is CampaignOutcome.UNSUPPORTED
    assert evidence[-1].inconclusive is False


def test_runtime_failure_is_retained_as_failed_campaign_evidence(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, _FailingRuntime())
    preview = _preview()
    adapter.preview("chat-session", "chat-request", preview)

    result = adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    assert result.outcome is ToolOutcome.FAILED
    with CampaignStateStore(tmp_path / "chat-run.campaign.sqlite3") as store:
        state = store.load_for_session("chat-session")
        evidence = store.evidence(state.campaign_id)
    assert state.lifecycle is CampaignLifecycle.FAILED
    assert state.outcome is CampaignOutcome.FAILED
    assert evidence[-1].inconclusive is True
    assert evidence[-1].artifact_digest is None


def test_interrupted_campaign_is_paused_and_resume_reuses_canonical_identity(
    tmp_path: Path,
) -> None:
    first = _adapter(tmp_path, _Runtime(interrupt_once=True))
    preview = _preview()
    first.preview("chat-session", "chat-request", preview)
    paused = first.execute("chat-session", "chat-request", preview, "approval-chat")

    resumed_adapter = _adapter(tmp_path, _Runtime(), resume=True)
    resumed_adapter.preview("chat-session", "chat-request", preview)
    resumed = resumed_adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    assert paused.outcome is ToolOutcome.UNKNOWN
    assert resumed.outcome is ToolOutcome.SUPPORTED
    assert paused.campaign_id == resumed.campaign_id
    with CampaignStateStore(tmp_path / "chat-run.campaign.sqlite3") as store:
        state = store.load(resumed.campaign_id or "missing")
    assert state.lifecycle is CampaignLifecycle.COMPLETED
    assert state.outcome is CampaignOutcome.SUPPORTED


def test_explicit_reconnect_restart_pause_and_cancel_are_durable(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, _Runtime(interrupt_once=True))
    preview = _preview()
    adapter.preview("chat-session", "chat-request", preview)
    paused = adapter.execute("chat-session", "chat-request", preview, "approval-chat")
    assert paused.campaign_id is not None

    recovered = adapter.reconnect(paused.campaign_id, "chat-session")
    assert recovered.lifecycle is CampaignLifecycle.PAUSED
    restarted = adapter.restart(paused.campaign_id, "chat-session", operation_id="restart-test")
    assert restarted.lifecycle is CampaignLifecycle.RUNNING
    paused_again = adapter.pause(
        paused.campaign_id,
        "chat-session",
        operation_id="pause-test",
        detail="test pause",
    )
    assert paused_again.lifecycle is CampaignLifecycle.PAUSED
    cancelled = adapter.cancel(
        paused.campaign_id,
        "chat-session",
        operation_id="cancel-test",
        detail="test cancellation",
    )
    assert cancelled.lifecycle is CampaignLifecycle.CANCELLED
    assert adapter.reconnect(paused.campaign_id, "chat-session").lifecycle is (
        CampaignLifecycle.CANCELLED
    )


def test_chat_campaign_projection_matches_direct_run_identity(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, _Runtime())
    preview = _preview()
    adapter.preview("chat-session", "chat-request", preview)
    result = adapter.execute("chat-session", "chat-request", preview, "approval-chat")

    run = json.loads((tmp_path / "chat-run.json").read_text(encoding="utf-8"))
    with CampaignStateStore(tmp_path / "chat-run.campaign.sqlite3") as store:
        state = store.load(result.campaign_id or "missing")
    assert state.run_id == run["run_id"]
    plan_context = state.policy_state["plan"]
    assert isinstance(plan_context, dict)
    assert plan_context["plan_id"] == run["plan_id"]
    assert state.source_model_digest == run["source_artifact_digest"]

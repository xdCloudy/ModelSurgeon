"""Scoped approval, reuse, expiry, and audit evidence regressions for #481."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from modelsurgeon.config import ModelConfig, Settings
from modelsurgeon.conversation import CampaignStateStore, ChatOptimizeAdapter, ToolOutcome
from modelsurgeon.experiments import (
    ApprovalDecision,
    ApprovalDecisionKind,
    ApprovalRequest,
    ApprovalReuse,
    OptimizationPackageError,
    build_approval_audit_record,
    validate_approval,
)
from modelsurgeon.optimization import build_optimize_plan
from modelsurgeon.optimization_orchestrator import (
    OptimizeInterrupted,
    OptimizeOrchestrator,
    OptimizeStage,
    StageContext,
    StageResult,
    WorkflowOutcome,
)
from modelsurgeon.search.spec_preview import build_spec_preview
from modelsurgeon.surgery.contracts import TransactionState
from test_chat import _intent

_APPROVALS = ("artifact_write", "plan_review", "resource_budget", "source_model")


class _InterruptingRuntime:
    def __init__(self) -> None:
        self.interrupted = False

    def run_stage(self, context: StageContext) -> StageResult:
        if not self.interrupted and context.stage is OptimizeStage.SURGERY:
            self.interrupted = True
            raise OptimizeInterrupted()
        if context.stage is OptimizeStage.PARETO:
            return StageResult(
                WorkflowOutcome.SUPPORTED,
                "evidence_scoped_approval",
                "measured fixture result",
                measured=True,
                complete=True,
                constraints_passed=True,
                artifact_digest="sha256:" + "b" * 64,
                candidate_id="candidate_scoped",
                candidate_state_id="state_scoped",
                evaluation_id="evaluation_scoped",
                transaction_state=TransactionState.COMMITTED,
                artifact_immutable=True,
            )
        return StageResult(
            WorkflowOutcome.SUPPORTED,
            f"evidence_{context.stage.value}",
            "bounded fixture stage completed",
            complete=True,
        )


def _plan() -> object:
    return build_optimize_plan(
        Settings(model=ModelConfig(path="models/tiny", revision="revision-1")),
        dry_run=False,
    )


def test_overbroad_scope_and_actor_mismatch_fail_closed() -> None:
    request = ApprovalRequest(
        "execute",
        "plan.fixture",
        "a" * 64,
        ("artifact_write", "execute"),
        "diff.fixture",
        "2026-09-08T10:00:00Z",
        "2026-09-08T11:00:00Z",
        "operator-alice",
        reuse=ApprovalReuse.ONE_TIME,
    )
    decision = ApprovalDecision(
        request.request_id,
        request.code,
        request.plan_id,
        request.plan_digest,
        request.diff_id,
        ApprovalDecisionKind.APPROVED,
        "2026-09-08T10:01:00Z",
        "operator-alice",
        "reviewed",
    )
    with pytest.raises(OptimizationPackageError, match="overbroad"):
        validate_approval(
            request,
            decision,
            current_plan_id=request.plan_id,
            current_plan_digest=request.plan_digest,
            required_scope=("execute",),
        )

    mismatched = ApprovalDecision(
        request.request_id,
        request.code,
        request.plan_id,
        request.plan_digest,
        request.diff_id,
        ApprovalDecisionKind.APPROVED,
        "2026-09-08T10:01:00Z",
        "operator-bob",
        "reviewed",
    )
    with pytest.raises(OptimizationPackageError, match="actor"):
        validate_approval(
            request,
            mismatched,
            current_plan_id=request.plan_id,
            current_plan_digest=request.plan_digest,
        )


def test_one_time_direct_approval_is_consumed_and_not_reused(tmp_path: Path) -> None:
    plan = _plan()
    state_path = tmp_path / "run.json"
    first = OptimizeOrchestrator(plan, state_path).run(
        _InterruptingRuntime(),
        approvals=_APPROVALS,
        approval_reuse={"plan_review": ApprovalReuse.ONE_TIME},
    )
    assert first.status.value == "paused"
    plan_review = next(item for item in first.approvals if item.code == "plan_review")
    assert plan_review.uses == 1
    assert any(item.kind == "consumed" for item in first.approval_audit)

    with pytest.raises(RuntimeError, match="one-time"):
        OptimizeOrchestrator(plan, state_path).run(
            _InterruptingRuntime(),
            resume=True,
            approvals=_APPROVALS,
            approval_reuse={"plan_review": ApprovalReuse.ONE_TIME},
        )


def test_redacted_audit_evidence_is_immutable_and_round_trips() -> None:
    request = ApprovalRequest(
        "execute",
        "plan.fixture",
        "a" * 64,
        ("execute",),
        "diff.fixture",
        "2026-09-08T10:00:00Z",
        "2026-09-08T11:00:00Z",
        "operator-alice",
    )
    decision = ApprovalDecision(
        request.request_id,
        request.code,
        request.plan_id,
        request.plan_digest,
        request.diff_id,
        ApprovalDecisionKind.APPROVED,
        "2026-09-08T10:01:00Z",
        "operator-alice",
        "token=super-secret",
    )
    audit = build_approval_audit_record(
        request,
        decision,
        kind="decided",
        detail="token=super-secret approved",
    )
    record = audit.to_record()
    assert "super-secret" not in str(record)
    assert record["detail"] == "token=<redacted> approved"
    assert build_approval_audit_record(
        request,
        decision,
        kind="decided",
        detail="token=super-secret approved",
    ) == audit


def test_expired_chat_approval_is_refused_before_runtime(tmp_path: Path) -> None:
    class Runtime:
        def run_stage(self, _context: StageContext) -> StageResult:
            raise AssertionError("expired approval must not reach the runtime")

    adapter = ChatOptimizeAdapter(
        Settings(model=ModelConfig(path="models/tiny", revision="revision-1")),
        state_path=tmp_path / "chat-run.json",
        runtime=Runtime(),
        approvals=_APPROVALS,
        approval_expires_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
    )
    preview = build_spec_preview(_intent("retain quality and reduce latency"))
    adapter.preview("chat-session", "chat-request", preview)
    result = adapter.execute("chat-session", "chat-request", preview, "approval-chat")
    assert result.outcome is ToolOutcome.REFUSED
    assert result.tool_result.failure is not None
    assert result.tool_result.failure.code.value == "approval_invalid"
    assert not (tmp_path / "chat-run.campaign.sqlite3").exists()


def test_chat_projects_plan_bound_approval_audit(tmp_path: Path) -> None:
    class Runtime:
        def run_stage(self, context: StageContext) -> StageResult:
            if context.stage is OptimizeStage.PARETO:
                return StageResult(
                    WorkflowOutcome.SUPPORTED,
                    "evidence_chat_scope",
                    "measured fixture result",
                    measured=True,
                    complete=True,
                    constraints_passed=True,
                    artifact_digest="sha256:" + "b" * 64,
                    candidate_id="candidate_chat_scope",
                    candidate_state_id="state_chat_scope",
                    evaluation_id="evaluation_chat_scope",
                    transaction_state=TransactionState.COMMITTED,
                    artifact_immutable=True,
                )
            return StageResult(
                WorkflowOutcome.SUPPORTED,
                f"evidence_{context.stage.value}",
                "bounded fixture stage completed",
                complete=True,
            )

    adapter = ChatOptimizeAdapter(
        Settings(model=ModelConfig(path="models/tiny", revision="revision-1")),
        state_path=tmp_path / "chat-run.json",
        runtime=Runtime(),
        approvals=_APPROVALS,
    )
    preview = build_spec_preview(_intent("retain quality and reduce latency"))
    adapter.preview("chat-session", "chat-request", preview)
    result = adapter.execute("chat-session", "chat-request", preview, "approval-chat")
    assert result.campaign_id is not None
    with CampaignStateStore(tmp_path / "chat-run.campaign.sqlite3") as store:
        state = store.load(result.campaign_id)
        history = store.history(result.campaign_id)
    assert state.approval.plan_id == result.plan_id
    plan_context = state.policy_state["plan"]
    assert isinstance(plan_context, dict)
    assert state.approval.plan_digest == plan_context["plan_digest"]
    assert "execute_approved_plan" in state.approval.scope
    assert state.approval.audit_evidence
    assert any(item.kind == "approval_audit_projected" for item in history)

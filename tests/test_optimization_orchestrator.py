"""Scenario coverage for the autonomous optimize orchestration boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.config import ModelConfig, Settings
from modelsurgeon.optimization import build_optimize_plan
from modelsurgeon.optimization_orchestrator import (
    OptimizeInterrupted,
    OptimizeOrchestrator,
    OptimizeOrchestratorError,
    OptimizeStage,
    StageContext,
    StageResult,
    WorkflowOutcome,
    WorkflowStatus,
)
from modelsurgeon.surgery.contracts import TransactionState


def _plan():
    return build_optimize_plan(
        Settings(model=ModelConfig(path="models/tiny", revision="revision-1")),
        dry_run=False,
    )


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
                "evaluation_tiny",
                "measured final candidate evidence",
                measured=True,
                complete=True,
                constraints_passed=self.feasible,
                artifact_digest="sha256:" + "b" * 64,
                candidate_id="candidate_tiny",
                candidate_state_id="state_tiny",
                evaluation_id="evaluation_tiny",
                transaction_state=TransactionState.COMMITTED,
                artifact_immutable=True,
                alternatives=("candidate_alternative",),
            )
        return StageResult(
            WorkflowOutcome.SUPPORTED,
            f"evidence_{context.stage.value}",
            "bounded stage completed",
            complete=True,
        )


def _approvals() -> tuple[str, ...]:
    return ("plan_review", "source_model", "resource_budget", "artifact_write")


def test_missing_approval_is_durable_and_does_not_start_work(tmp_path: Path) -> None:
    runtime = _Runtime()
    run = OptimizeOrchestrator(_plan(), tmp_path / "run.json").run(runtime)

    assert run.status is WorkflowStatus.PAUSED
    assert run.outcome is WorkflowOutcome.UNKNOWN
    assert runtime.calls == []
    assert "required approvals" in " ".join(run.reasons)


def test_execution_resumes_without_repeating_completed_stages(tmp_path: Path) -> None:
    state = tmp_path / "run.json"
    runtime = _Runtime()
    orchestrator = OptimizeOrchestrator(_plan(), state)
    first = orchestrator.run(runtime, approvals=_approvals())

    assert first.status is WorkflowStatus.COMPLETED
    assert first.outcome is WorkflowOutcome.SUPPORTED
    assert first.accepted_artifact_digest == "sha256:" + "b" * 64
    assert first.alternatives == ("candidate_alternative",)
    calls = len(runtime.calls)

    resumed = orchestrator.run(_Runtime(), resume=True, approvals=_approvals())
    assert resumed == first
    assert len(runtime.calls) == calls
    assert all(stage.attempts == 1 for stage in resumed.stages)


def test_interruption_leaves_running_stage_replayable(tmp_path: Path) -> None:
    state = tmp_path / "run.json"
    orchestrator = OptimizeOrchestrator(_plan(), state)
    paused = orchestrator.run(_Runtime(interrupt_once=True), approvals=_approvals())

    assert paused.status is WorkflowStatus.PAUSED
    assert paused.cursor == 5
    assert paused.stages[5].status.value == "running"

    resumed = orchestrator.run(_Runtime(), resume=True, approvals=_approvals())
    assert resumed.status is WorkflowStatus.COMPLETED
    assert resumed.outcome is WorkflowOutcome.SUPPORTED
    assert resumed.stages[5].attempts == 2


def test_no_feasible_result_is_a_failure_without_promotion(tmp_path: Path) -> None:
    run = OptimizeOrchestrator(_plan(), tmp_path / "run.json").run(
        _Runtime(feasible=False), approvals=_approvals()
    )

    assert run.status is WorkflowStatus.COMPLETED
    assert run.outcome is WorkflowOutcome.FAILED
    assert run.accepted_artifact_digest is None
    assert "no feasible measured candidate" in " ".join(run.reasons)


def test_resume_rejects_material_plan_change_with_diff_evidence(tmp_path: Path) -> None:
    state = tmp_path / "run.json"
    first = OptimizeOrchestrator(_plan(), state).run(
        _Runtime(interrupt_once=True), approvals=_approvals()
    )
    assert first.plan_digest
    assert first.plan_record["plan_id"] == first.plan_id
    assert first.approvals[0].expires_at
    changed = build_optimize_plan(
        Settings(model=ModelConfig(path="models/changed", revision="revision-1")),
        dry_run=False,
    )

    with pytest.raises(OptimizeOrchestratorError, match="material_diff=True"):
        OptimizeOrchestrator(changed, state).run(
            _Runtime(), resume=True, approvals=_approvals()
        )

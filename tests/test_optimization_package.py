"""Approval, replay, and offline package regression tests."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from modelsurgeon.config import ModelConfig, Settings
from modelsurgeon.experiments import (
    ApprovalDecision,
    ApprovalDecisionKind,
    ApprovalRequest,
    EvidenceBundleExternalReference,
    EvidenceKeyRecord,
    EvidenceKeyStatus,
    OptimizationPackageError,
    PackageVerificationStatus,
    diff_plans,
    plan_digest,
    replay_decision_evidence,
    replay_optimize_run,
    validate_approval,
    verify_reproducibility_package,
    write_reproducibility_package,
)
from modelsurgeon.optimization import build_optimize_plan
from modelsurgeon.optimization_orchestrator import (
    OptimizeOrchestrator,
    OptimizeStage,
    StageContext,
    StageResult,
    WorkflowOutcome,
)
from modelsurgeon.surgery.contracts import TransactionState


def _plan(model: str = "models/tiny"):
    return build_optimize_plan(
        Settings(model=ModelConfig(path=model, revision="revision-1")),
        dry_run=False,
    )


def test_material_plan_diff_requires_a_new_approval_decision() -> None:
    before = _plan()
    after = _plan("models/changed")
    changed = diff_plans(before, after)

    assert changed.material
    assert "resolved_config.model.path" in changed.changed_paths
    assert changed.diff_id.startswith("plan_diff_")

    requested = ApprovalRequest(
        "artifact_write",
        before.plan_id,
        plan_digest(before),
        ("artifact_write", "plan"),
        changed.diff_id,
        "2026-09-07T10:00:00+00:00",
        "2026-09-07T11:00:00+00:00",
        "operator-7",
        (("ticket", "change-428"),),
    )
    decision = ApprovalDecision(
        requested.request_id,
        requested.code,
        requested.plan_id,
        requested.plan_digest,
        requested.diff_id,
        ApprovalDecisionKind.APPROVED,
        "2026-09-07T10:01:00+00:00",
        "operator-7",
        "reviewed the requested artifact publication",
    )

    with pytest.raises(OptimizationPackageError, match="current plan"):
        validate_approval(
            requested,
            decision,
            current_plan_id=after.plan_id,
            current_plan_digest=plan_digest(after),
            at=datetime(2026, 9, 7, 10, 2, tzinfo=UTC),
        )


def test_expired_and_secret_contaminated_approvals_fail_closed() -> None:
    with pytest.raises(OptimizationPackageError, match="secret"):
        ApprovalRequest(
            "plan_review",
            "plan-1",
            "a" * 64,
            ("plan",),
            "diff-1",
            "2026-09-07T10:00:00+00:00",
            "2026-09-07T11:00:00+00:00",
            "operator-7",
            (("api_token", "do-not-record"),),
        )

    request = ApprovalRequest(
        "plan_review",
        "plan-1",
        "a" * 64,
        ("plan",),
        "diff-1",
        "2026-09-07T10:00:00+00:00",
        "2026-09-07T11:00:00+00:00",
        "operator-7",
    )
    decision = ApprovalDecision(
        request.request_id,
        request.code,
        request.plan_id,
        request.plan_digest,
        request.diff_id,
        ApprovalDecisionKind.APPROVED,
        "2026-09-07T11:01:00+00:00",
        "operator-7",
        "late",
    )
    with pytest.raises(OptimizationPackageError, match="expired"):
        validate_approval(
            request,
            decision,
            current_plan_id=request.plan_id,
            current_plan_digest=request.plan_digest,
            at=datetime(2026, 9, 7, 11, 2, tzinfo=UTC),
        )


def test_identical_evidence_replays_the_same_selection() -> None:
    evidence = (
        {
            "candidate_id": "candidate-b",
            "score": 2.0,
            "measured": True,
            "complete": True,
            "constraints_passed": True,
        },
        {
            "candidate_id": "candidate-a",
            "score": 1.0,
            "measured": True,
            "complete": True,
            "constraints_passed": True,
        },
    )
    first = replay_decision_evidence(evidence)
    second = replay_decision_evidence(evidence)

    assert first == second
    assert first.selected_candidate_id == "candidate-a"
    assert first.strategy_decision_id.startswith("strategy_replay_")

    changed = replay_decision_evidence((evidence[0], {**evidence[1], "score": 3.0}))
    assert changed.evidence_digest != first.evidence_digest
    assert changed.selected_candidate_id == "candidate-b"


class _Runtime:
    def run_stage(self, context: StageContext) -> StageResult:
        if context.stage is OptimizeStage.PARETO:
            return StageResult(
                WorkflowOutcome.SUPPORTED,
                "evaluation_tiny",
                "measured final candidate evidence",
                measured=True,
                complete=True,
                constraints_passed=True,
                artifact_digest="sha256:" + "b" * 64,
                candidate_id="candidate_tiny",
                candidate_state_id="state_tiny",
                evaluation_id="evaluation_tiny",
                transaction_state=TransactionState.COMMITTED,
                artifact_immutable=True,
            )
        return StageResult(
            WorkflowOutcome.SUPPORTED,
            f"evidence_{context.stage.value}",
            "bounded stage completed",
            complete=True,
        )


def test_signed_package_verifies_offline_and_tampering_is_detected(tmp_path: Path) -> None:
    plan = _plan()
    run = OptimizeOrchestrator(plan, tmp_path / "run.json").run(
        _Runtime(),
        approvals=("plan_review", "source_model", "resource_budget", "artifact_write"),
    )
    key_material = b"package-test-key"
    key = EvidenceKeyRecord(
        "key-package-test",
        hashlib.sha256(key_material).hexdigest(),
        EvidenceKeyStatus.ACTIVE,
    )
    evidence = [item["result"] for item in run.to_record()["stages"] if item["result"] is not None]
    assert replay_optimize_run(run).selected_candidate_id == "candidate_tiny"
    package = write_reproducibility_package(
        tmp_path / "package",
        plan=plan,
        run=run,
        decision_evidence=evidence,
        source_commit="a" * 40,
        data_revisions=("data-revision-1",),
        hardware_contexts=("cpu-small",),
        signing_key=key,
        signing_key_material=key_material,
    )

    verified = verify_reproducibility_package(
        package.destination,
        key_material={key.key_id: key_material},
    )
    assert verified.status is PackageVerificationStatus.VERIFIED
    assert verified.verified

    (package.destination / "contents" / "run.json").write_text("tampered\n", encoding="utf-8")
    tampered = verify_reproducibility_package(
        package.destination,
        key_material={key.key_id: key_material},
    )
    assert tampered.status is PackageVerificationStatus.FAILED


def test_package_declares_unavailable_external_artifacts(tmp_path: Path) -> None:
    plan = _plan()
    run = OptimizeOrchestrator(plan, tmp_path / "run.json").run(
        _Runtime(),
        approvals=("plan_review", "source_model", "resource_budget", "artifact_write"),
    )
    package = write_reproducibility_package(
        tmp_path / "package",
        plan=plan,
        run=run,
        decision_evidence=[
            {
                "candidate_id": "candidate_tiny",
                "measured": True,
                "complete": True,
                "constraints_passed": True,
            }
        ],
        external_artifacts=(
            EvidenceBundleExternalReference(
                "datasets/heldout.jsonl",
                "c" * 64,
                20,
                "https://example.invalid/heldout.jsonl",
                "missing",
            ),
        ),
    )
    result = verify_reproducibility_package(package.destination)
    assert result.status is PackageVerificationStatus.INCOMPLETE
    assert "external references are unavailable" in " ".join(result.issues)
    assert package.manifest["unavailable_external_artifacts"] == ["datasets/heldout.jsonl"]

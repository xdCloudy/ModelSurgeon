"""Canonical campaign projection for the bounded conversational execution slice.

The optimize orchestrator remains the execution authority and retains its
stage cursor in JSON.  This module projects that trusted result into the
WAL-backed conversational campaign store from #469.  It never accepts
provider text, transcript data, or tool output as evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from modelsurgeon.conversation.campaign_state import (
    ApprovalStatus,
    CampaignApproval,
    CampaignBudget,
    CampaignEvidence,
    CampaignLifecycle,
    CampaignOutcome,
    CampaignSpec,
    CampaignState,
    CampaignStateError,
    CampaignStateStore,
    new_campaign_state,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.optimization_package import (
    ApprovalDecision,
    ApprovalDecisionKind,
    ApprovalRequest,
    ApprovalReuse,
    build_approval_audit_record,
    diff_plans,
    plan_digest,
)
from modelsurgeon.optimization import OptimizePlan
from modelsurgeon.optimization_orchestrator import (
    OptimizeRun,
    StageRecord,
    WorkflowOutcome,
    WorkflowStatus,
    optimize_run_id,
    source_artifact_digest,
)

if TYPE_CHECKING:
    from modelsurgeon.search.spec_preview import SpecPreview

_APPROVAL_TTL = timedelta(hours=1)


def _canonical(value: object) -> str:
    return canonical_identity_json(value)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _identifier_digest(prefix: str, value: object) -> str:
    return prefix + _digest(value)[len("sha256:") :]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _expiry() -> str:
    return (datetime.now(UTC) + _APPROVAL_TTL).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CampaignStateError(f"{label} must be non-empty text")
    return value


def _campaign_outcome(value: WorkflowOutcome) -> CampaignOutcome:
    try:
        return CampaignOutcome(value.value)
    except ValueError as error:  # pragma: no cover - enum values are intentionally aligned
        raise CampaignStateError(f"unsupported workflow outcome {value.value!r}") from error


def _hard_constraints(preview: SpecPreview) -> tuple[Mapping[str, object], ...]:
    return tuple(preview.hard_constraints)


def _plan_context(plan: OptimizePlan) -> dict[str, object]:
    return {
        "record_type": "optimize_plan_provenance",
        "plan_id": plan.plan_id,
        "plan_digest": "sha256:" + plan_digest(plan),
        "source_model_digest": source_artifact_digest(plan),
        "optimize_schema": plan.to_record().get("schema_version"),
    }


def _stage_evidence(run: OptimizeRun, record: StageRecord) -> CampaignEvidence | None:
    if record.result is None:
        return None
    result = record.result
    evidence_id = _identifier_digest(
        "campaign_evidence_",
        {"run_id": run.run_id, "stage": record.stage.value, "result": result.to_record()},
    )
    return CampaignEvidence(
        evidence_id,
        run.source_artifact_digest,
        _campaign_outcome(result.outcome),
        result.detail,
        {
            "record_type": "optimize_stage_evidence",
            "run_id": run.run_id,
            "plan_id": run.plan_id,
            "stage": record.stage.value,
            "stage_evidence_id": result.evidence_id,
            "measured": result.measured,
            "complete": result.complete,
            "constraints_passed": result.constraints_passed,
        },
        result.artifact_digest
        if result.artifact_immutable
        and run.accepted_artifact_digest == result.artifact_digest
        and run.outcome is WorkflowOutcome.SUPPORTED
        else None,
        inconclusive=result.outcome is WorkflowOutcome.UNKNOWN
        or not result.complete
        or not result.measured,
    )


def _run_evidence(run: OptimizeRun) -> CampaignEvidence:
    evidence_id = _identifier_digest(
        "campaign_evidence_",
        {"run_id": run.run_id, "record_type": "optimize_run", "run": run.to_record()},
    )
    detail = "; ".join(run.reasons) or f"optimize run completed with {run.outcome.value} outcome"
    return CampaignEvidence(
        evidence_id,
        run.source_artifact_digest,
        _campaign_outcome(run.outcome),
        detail,
        {
            "record_type": "optimize_campaign_evidence",
            "run_id": run.run_id,
            "plan_id": run.plan_id,
            "plan_digest": run.plan_digest,
            "workflow_status": run.status.value,
            "stage_evidence_ids": [
                item.result.evidence_id
                for item in run.stages
                if item.result is not None
            ],
            "decision": "accepted"
            if run.accepted_artifact_digest is not None
            else "rejected"
            if run.outcome is WorkflowOutcome.FAILED
            else run.outcome.value,
        },
        run.accepted_artifact_digest,
        inconclusive=run.outcome is WorkflowOutcome.UNKNOWN,
    )


class CanonicalCampaignRecorder:
    """Bind one optimize plan/run to the canonical campaign state store."""

    def __init__(
        self,
        path: str | Path,
        *,
        session_id: str,
        plan: OptimizePlan,
        preview: SpecPreview,
        provider_context: Mapping[str, object] | None = None,
        approval_id: str | None = None,
        recorded_by: str = "chat",
        approval_expires_at: str | None = None,
        approval_reuse: Mapping[str, ApprovalReuse | str] | None = None,
    ) -> None:
        if not preview.executable or preview.spec is None or preview.spec_identity is None:
            raise CampaignStateError("canonical campaign recording requires an executable preview")
        if preview.policy_decision is None or not preview.policy_decision.executable:
            raise CampaignStateError(
                "canonical campaign recording requires an executable policy decision"
            )
        assert preview.policy_decision is not None
        self.path = Path(path)
        self.session_id = _text(session_id, "campaign session ID")
        self.plan = plan
        self.preview = preview
        self.policy_decision = preview.policy_decision
        self.provider_context = {} if provider_context is None else dict(provider_context)
        self.approval_id = approval_id
        self.recorded_by = _text(recorded_by, "campaign recorder")
        self.approval_expires_at = approval_expires_at
        self.approval_reuse = None if approval_reuse is None else dict(approval_reuse)
        self.run_id = optimize_run_id(plan)
        self.source_model_digest = source_artifact_digest(plan)
        self.campaign_id = _identifier_digest(
            "campaign_",
            {
                "session_id": self.session_id,
                "run_id": self.run_id,
                "source_model_digest": self.source_model_digest,
                "spec_identity": preview.spec_identity,
                "spec_digest": preview.spec_digest,
            },
        )

    def _initial(self) -> CampaignState:
        assert self.preview.spec is not None
        assert self.preview.spec_identity is not None
        assert self.preview.spec_digest is not None
        approval = CampaignApproval.pending(self.preview.spec_digest)
        plan_digest_value = plan_digest(self.plan)
        plan_diff_id = diff_plans(self.plan, self.plan).diff_id
        required_codes = tuple(item.code for item in self.plan.approvals if item.required)
        approval_scope = tuple(sorted({"execute_approved_plan", *required_codes}))
        reuse_values: list[ApprovalReuse] = []
        for code in (*required_codes, "execute_approved_plan"):
            raw = None if self.approval_reuse is None else self.approval_reuse.get(code)
            if raw is not None:
                try:
                    reuse_values.append(
                        raw if isinstance(raw, ApprovalReuse) else ApprovalReuse(str(raw))
                    )
                except ValueError as error:
                    raise CampaignStateError("approval reuse policy is invalid") from error
        approval_reuse = (
            ApprovalReuse.ONE_TIME
            if ApprovalReuse.ONE_TIME in reuse_values
            else ApprovalReuse.REUSABLE
        )
        expiry = self.approval_expires_at or _expiry()
        audit_evidence: tuple[Mapping[str, object], ...] = ()
        if self.approval_id is not None:
            requested_at = _now()
            request = ApprovalRequest(
                "execute_approved_plan",
                self.plan.plan_id,
                plan_digest_value,
                approval_scope,
                plan_diff_id,
                requested_at,
                expiry,
                self.recorded_by,
                (("campaign_id", self.campaign_id),),
                approval_reuse,
                1 if approval_reuse is ApprovalReuse.ONE_TIME else None,
            )
            decision = ApprovalDecision(
                request.request_id,
                request.code,
                request.plan_id,
                request.plan_digest,
                request.diff_id,
                ApprovalDecisionKind.APPROVED,
                requested_at,
                self.recorded_by,
                "explicit chat approval recorded for the exact plan scope",
                request.operator_context,
            )
            audit_evidence = (
                build_approval_audit_record(
                    request,
                    decision,
                    kind="issued",
                    detail="scoped chat approval issued for the exact current plan",
                ).to_record(),
            )
            approval = CampaignApproval(
                ApprovalStatus.APPROVED,
                self.preview.spec_digest,
                approval_id=self.approval_id,
                recorded_by=self.recorded_by,
                expires_at=expiry,
                provenance={
                    "record_type": "explicit_campaign_approval",
                    "source": "trusted_boundary",
                    "plan_id": self.plan.plan_id,
                },
                plan_id=self.plan.plan_id,
                plan_digest="sha256:" + plan_digest_value,
                diff_id=plan_diff_id,
                scope=approval_scope,
                reuse=approval_reuse,
                audit_evidence=audit_evidence,
            )
        else:
            approval = CampaignApproval(
                ApprovalStatus.PENDING,
                self.preview.spec_digest,
                plan_id=self.plan.plan_id,
                plan_digest="sha256:" + plan_digest_value,
                diff_id=plan_diff_id,
                scope=approval_scope,
                reuse=approval_reuse,
            )
        return new_campaign_state(
            session_id=self.session_id,
            run_id=self.run_id,
            source_model_digest=self.source_model_digest,
            spec=CampaignSpec(
                self.preview.spec_identity,
                self.preview.spec_digest,
                self.preview.spec,
                _hard_constraints(self.preview),
            ),
            policy_state={
                "record_type": "conversational_policy_decision",
                "intent_id": self.preview.intent_id,
                "outcome": self.preview.outcome.value,
                "approval_required": self.preview.approval_required,
                "plan": _plan_context(self.plan),
                "hard_constraints": [dict(item) for item in self.preview.hard_constraints],
                "provenance": dict(self.preview.provenance),
                "policy_precedence": self.policy_decision.to_record(),
            },
            provider_context=self.provider_context,
            budget=CampaignBudget(
                600.0,
                self.plan.budget.max_ram_bytes,
                self.plan.budget.evaluations,
                self.plan.budget.max_artifact_bytes,
            ),
            approval=approval,
            campaign_id=self.campaign_id,
            provenance={
                "record_type": "conversational_campaign_creation",
                "source": "modelsurgeon.conversation",
                "plan_id": self.plan.plan_id,
                "plan_digest": "sha256:" + plan_digest(self.plan),
                "preview_id": self.preview.preview_id,
            },
        )

    def _open(self) -> CampaignStateStore:
        return CampaignStateStore(self.path)

    def ensure(self) -> CampaignState:
        """Create or reconnect the exact campaign, rejecting identity drift."""

        with self._open() as store:
            try:
                current = store.load(self.campaign_id)
            except CampaignStateError as error:
                if not str(error).startswith("unknown campaign"):
                    raise
                current = store.create(self._initial())
            plan_context = current.policy_state.get("plan")
            if not isinstance(plan_context, Mapping):
                raise CampaignStateError("campaign policy provenance is missing its plan")
            if (
                current.session_id != self.session_id
                or current.run_id != self.run_id
                or current.source_model_digest != self.source_model_digest
                or current.spec_digest != self.preview.spec_digest
                or plan_context.get("plan_id") != self.plan.plan_id
                or current.approval.plan_id != self.plan.plan_id
                or current.approval.plan_digest != "sha256:" + plan_digest(self.plan)
                or current.policy_state.get("policy_precedence")
                != self.policy_decision.to_record()
            ):
                raise CampaignStateError("campaign identity or policy provenance has drifted")
            return current

    def start(self) -> CampaignState:
        current = self.ensure()
        if current.approval.approval_id is not None and not current.approval.active:
            if current.approval.status is ApprovalStatus.APPROVED:
                expired = replace(current.approval, status=ApprovalStatus.EXPIRED)
                with self._open() as store:
                    current = store.transition(
                        self.campaign_id,
                        expected_version=current.state_version,
                        kind="approval_expired",
                        lifecycle=CampaignLifecycle.PAUSED,
                        approval=expired,
                        provenance={
                            "record_type": "campaign_approval_expiry",
                            "source": "trusted_campaign_recorder",
                            "plan_id": self.plan.plan_id,
                        },
                    )
            raise CampaignStateError("campaign approval is missing, expired, or consumed")
        if current.lifecycle is CampaignLifecycle.CREATED:
            with self._open() as store:
                return store.transition(
                    self.campaign_id,
                    expected_version=current.state_version,
                    kind="started",
                    lifecycle=CampaignLifecycle.RUNNING,
                    outcome=CampaignOutcome.UNKNOWN,
                    provenance={
                        "record_type": "campaign_execution_start",
                        "source": "trusted_optimize_adapter",
                        "plan_id": self.plan.plan_id,
                    },
                    approval=replace(
                        current.approval,
                        uses=current.approval.uses + 1
                        if current.approval.approval_id is not None
                        else current.approval.uses,
                    ),
                )
        if current.lifecycle is CampaignLifecycle.PAUSED:
            with self._open() as store:
                return store.transition(
                    self.campaign_id,
                    expected_version=current.state_version,
                    kind="resumed",
                    lifecycle=CampaignLifecycle.RUNNING,
                    outcome=CampaignOutcome.UNKNOWN,
                    provenance={
                        "record_type": "campaign_execution_resume",
                        "source": "trusted_optimize_adapter",
                        "plan_id": self.plan.plan_id,
                    },
                )
        return current

    def retain_run(self, run: OptimizeRun) -> tuple[CampaignState, tuple[str, ...]]:
        """Retain all completed stage evidence and the final run decision."""

        state = self.start()
        references: list[str] = []
        with self._open() as store:
            if run.approval_audit:
                projected = tuple(item.to_record() for item in run.approval_audit)
                existing = state.approval.audit_evidence
                merged: tuple[Mapping[str, object], ...]
                if not existing or existing == projected[: len(existing)]:
                    merged = projected
                elif len(existing) == 1 and existing[0] not in projected:
                    merged = existing + projected
                elif len(existing) > 1 and existing[1:] == projected[: len(existing) - 1]:
                    merged = existing + projected[len(existing) - 1 :]
                elif existing[-len(projected) :] == projected:
                    merged = existing
                else:
                    raise CampaignStateError("campaign approval audit history conflicts")
                if merged != existing:
                    state = store.transition(
                        self.campaign_id,
                        expected_version=state.state_version,
                        kind="approval_audit_projected",
                        approval=replace(state.approval, audit_evidence=merged),
                        provenance={
                            "record_type": "campaign_approval_audit_projection",
                            "source": "trusted_optimize_orchestrator",
                            "audit_count": len(merged),
                            "plan_id": self.plan.plan_id,
                        },
                    )
            for record in run.stages:
                evidence = _stage_evidence(run, record)
                if evidence is None:
                    continue
                state = store.append_evidence(
                    self.campaign_id,
                    evidence,
                    expected_version=state.state_version,
                )
                references.append(evidence.evidence_id)
            final = _run_evidence(run)
            state = store.append_evidence(
                self.campaign_id,
                final,
                expected_version=state.state_version,
            )
            references.append(final.evidence_id)
            lifecycle = (
                CampaignLifecycle.PAUSED
                if run.status is WorkflowStatus.PAUSED
                else CampaignLifecycle.FAILED
                if run.status is WorkflowStatus.FAILED
                else CampaignLifecycle.COMPLETED
            )
            run_outcome = _campaign_outcome(run.outcome)
            if state.lifecycle is not lifecycle or state.outcome is not run_outcome:
                state = store.transition(
                    self.campaign_id,
                    expected_version=state.state_version,
                    kind="execution_finished"
                    if lifecycle is CampaignLifecycle.COMPLETED
                    else "execution_paused"
                    if lifecycle is CampaignLifecycle.PAUSED
                    else "execution_failed",
                    lifecycle=lifecycle,
                    outcome=run_outcome,
                    provenance={
                        "record_type": "campaign_execution_result",
                        "source": "trusted_optimize_orchestrator",
                        "run_id": run.run_id,
                        "plan_id": run.plan_id,
                        "evidence_id": final.evidence_id,
                        "artifact_digest": run.accepted_artifact_digest,
                    },
                )
        return state, tuple(references)

    def retain_outcome(
        self,
        detail: str,
        *,
        outcome: CampaignOutcome = CampaignOutcome.FAILED,
        lifecycle: CampaignLifecycle = CampaignLifecycle.FAILED,
        inconclusive: bool = True,
    ) -> tuple[CampaignState, str]:
        """Retain a terminal engine outcome without inventing an artifact."""

        state = self.start()
        evidence = CampaignEvidence(
            _identifier_digest(
                "campaign_evidence_",
                {"campaign_id": self.campaign_id, "failure": detail},
            ),
            self.source_model_digest,
            outcome,
            _text(detail, "campaign failure detail"),
            {
                "record_type": "optimize_campaign_failure",
                "source": "trusted_optimize_adapter",
                "plan_id": self.plan.plan_id,
                "run_id": self.run_id,
            },
            inconclusive=inconclusive,
        )
        with self._open() as store:
            state = store.append_evidence(
                self.campaign_id,
                evidence,
                expected_version=state.state_version,
            )
            if state.lifecycle is not lifecycle or state.outcome is not outcome:
                store.transition(
                    self.campaign_id,
                    expected_version=state.state_version,
                    kind="execution_unsupported"
                    if outcome is CampaignOutcome.UNSUPPORTED
                    else "execution_failed",
                    lifecycle=lifecycle,
                    outcome=outcome,
                    provenance={
                        "record_type": "campaign_execution_failure",
                        "source": "trusted_optimize_adapter",
                        "evidence_id": evidence.evidence_id,
                    },
                )
        return state, evidence.evidence_id

    def retain_failure(self, detail: str) -> tuple[CampaignState, str]:
        """Retain an engine failure without inventing an artifact or measurement."""

        return self.retain_outcome(detail)


__all__ = ["CanonicalCampaignRecorder"]

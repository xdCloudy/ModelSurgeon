"""Deterministic stale-context detection and campaign replanning.

Replanning is a control-plane operation.  It compares a trusted context
snapshot with the current canonical campaign state, emits an immutable diff,
and creates a child campaign only after the new plan has its own approval.  A
child references its parent's evidence by ID; it never copies or relabels
that evidence as current.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from modelsurgeon.conversation.campaign_state import (
    ApprovalStatus,
    CampaignApproval,
    CampaignBudget,
    CampaignLifecycle,
    CampaignSpec,
    CampaignState,
    CampaignStateError,
    CampaignStateStore,
    derive_campaign_id,
    new_campaign_state,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.optimization_package import (
    ApprovalDecision,
    ApprovalRequest,
    OptimizationPackageError,
    PlanDiff,
    diff_plans,
    plan_digest,
    validate_approval,
)

if TYPE_CHECKING:
    from modelsurgeon.search.objective_amendments import ObjectiveAmendment


REPLANNING_SCHEMA_VERSION = 1


class ReplanningError(ValueError):
    """Raised when a replan cannot be trusted or applied safely."""


class StaleContextError(ReplanningError):
    """Raised when a replan was prepared from an obsolete context."""

    def __init__(self, report: StaleContextReport) -> None:
        self.report = report
        super().__init__(
            "stale conversational context: " + ", ".join(report.reasons)
        )


class ReplanApprovalRequired(ReplanningError):
    """Raised when a material replan has no fresh approval request."""

    def __init__(self, diff: ReplanDiff) -> None:
        self.diff = diff
        super().__init__("material replans require a new approval request")


class ReplanReplayError(ReplanningError):
    """Raised when a stale, cancelled, or conflicting replan is replayed."""


class StaleContextReason(StrEnum):
    SPEC = "spec"
    EVIDENCE = "evidence"
    PROVIDER = "provider"
    RESOURCE = "resource"
    PLAN = "plan"
    APPROVAL = "approval"
    TRANSACTION = "transaction"


def _canonical(value: object) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ReplanningError("replanning data must be canonical JSON") from error


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _digest(value: object) -> str:
    return "sha256:" + _sha256(value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplanningError(f"{label} is required")
    return value


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ReplanningError(f"{label} must be a JSON object")
    try:
        decoded = json.loads(_canonical(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ReplanningError(f"{label} must be canonical JSON") from error
    if not isinstance(decoded, dict):
        raise ReplanningError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)


def _timestamp(value: object, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReplanningError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ReplanningError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def _identifier(value: object, label: str) -> str:
    return _text(value, label)


def _record(value: object) -> dict[str, object]:
    if hasattr(value, "to_record"):
        value = value.to_record()
    return _mapping(value, "replanning record")


def _evidence_context(
    evidence: Sequence[object],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], str]:
    records: list[tuple[str, str]] = []
    for item in evidence:
        record = _record(item)
        evidence_id = _identifier(record.get("evidence_id"), "evidence ID")
        digest = record.get("evidence_digest")
        if not isinstance(digest, str):
            digest = _digest(record)
        records.append((evidence_id, digest))
    ordered = tuple(sorted(set(records)))
    ids = tuple(item[0] for item in ordered)
    return ids, ordered, _digest(list(ordered))


@dataclass(frozen=True, slots=True)
class ReplanContext:
    """The trusted context against which a replan request was prepared."""

    campaign_id: str
    spec_identity: str
    spec_digest: str
    evidence_ids: tuple[str, ...]
    evidence_digest: str
    provider_digest: str
    resource_digest: str
    plan_id: str
    plan_digest: str
    approval_id: str | None
    transaction_id: str
    schema_version: int = REPLANNING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign ID")
        _identifier(self.spec_identity, "spec identity")
        _text(self.spec_digest, "spec digest")
        _text(self.evidence_digest, "evidence digest")
        _text(self.provider_digest, "provider digest")
        _text(self.resource_digest, "resource digest")
        _identifier(self.plan_id, "plan ID")
        _text(self.plan_digest, "plan digest")
        _identifier(self.transaction_id, "transaction ID")
        if self.approval_id is not None:
            _identifier(self.approval_id, "approval ID")
        if self.evidence_ids != tuple(sorted(set(self.evidence_ids))):
            raise ReplanningError("context evidence IDs must be sorted and unique")
        if self.schema_version != REPLANNING_SCHEMA_VERSION:
            raise ReplanningError("unsupported replanning context schema")

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "stale_replanning_context",
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "spec_identity": self.spec_identity,
            "spec_digest": self.spec_digest,
            "evidence_ids": list(self.evidence_ids),
            "evidence_digest": self.evidence_digest,
            "provider_digest": self.provider_digest,
            "resource_digest": self.resource_digest,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "approval_id": self.approval_id,
            "transaction_id": self.transaction_id,
        }

    @property
    def digest(self) -> str:
        return _digest(self.to_record())


def context_from_campaign(
    state: CampaignState,
    plan: object,
    evidence: Sequence[object],
    *,
    transaction_id: str,
) -> ReplanContext:
    """Build a context snapshot from trusted state, plan, and evidence."""

    plan_record = _record(plan)
    evidence_ids, _evidence_records, evidence_digest = _evidence_context(evidence)
    return ReplanContext(
        state.campaign_id,
        state.spec_identity,
        state.spec_digest,
        evidence_ids,
        evidence_digest,
        _digest(state.provider_context),
        _digest(state.budget.to_record()),
        _identifier(plan_record.get("plan_id"), "plan ID"),
        plan_digest(plan_record),
        state.approval.approval_id,
        _identifier(transaction_id, "transaction ID"),
    )


@dataclass(frozen=True, slots=True)
class StaleContextReport:
    expected: ReplanContext
    current: ReplanContext
    reasons: tuple[str, ...]
    schema_version: int = REPLANNING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.reasons != tuple(sorted(set(self.reasons))):
            raise ReplanningError("stale-context reasons must be sorted and unique")
        if any(
            item not in {reason.value for reason in StaleContextReason}
            for item in self.reasons
        ):
            raise ReplanningError("stale-context reason is unsupported")
        if self.schema_version != REPLANNING_SCHEMA_VERSION:
            raise ReplanningError("unsupported stale-context schema")

    @property
    def stale(self) -> bool:
        return bool(self.reasons)

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "stale_context_report",
            "schema_version": self.schema_version,
            "expected": self.expected.to_record(),
            "current": self.current.to_record(),
            "reasons": list(self.reasons),
            "stale": self.stale,
        }


def detect_stale_context(
    expected: ReplanContext,
    current: ReplanContext,
) -> StaleContextReport:
    """Compare semantic context, ignoring a harmless state-version/restart."""

    reasons: set[str] = set()
    if (expected.spec_identity, expected.spec_digest) != (
        current.spec_identity,
        current.spec_digest,
    ):
        reasons.add(StaleContextReason.SPEC.value)
    if (expected.evidence_ids, expected.evidence_digest) != (
        current.evidence_ids,
        current.evidence_digest,
    ):
        reasons.add(StaleContextReason.EVIDENCE.value)
    if expected.provider_digest != current.provider_digest:
        reasons.add(StaleContextReason.PROVIDER.value)
    if expected.resource_digest != current.resource_digest:
        reasons.add(StaleContextReason.RESOURCE.value)
    if (expected.plan_id, expected.plan_digest) != (current.plan_id, current.plan_digest):
        reasons.add(StaleContextReason.PLAN.value)
    if expected.approval_id != current.approval_id:
        reasons.add(StaleContextReason.APPROVAL.value)
    if expected.transaction_id != current.transaction_id:
        reasons.add(StaleContextReason.TRANSACTION.value)
    if expected.campaign_id != current.campaign_id:
        reasons.add(StaleContextReason.SPEC.value)
    return StaleContextReport(expected, current, tuple(sorted(reasons)))


@dataclass(frozen=True, slots=True)
class ReplanDiff:
    """Canonical, versioned diff emitted for one replan attempt."""

    before_campaign_id: str
    after_campaign_id: str | None
    before_plan_id: str
    after_plan_id: str
    before_plan_digest: str
    after_plan_digest: str
    before_spec_identity: str
    after_spec_identity: str
    changed_paths: tuple[str, ...]
    stale_reasons: tuple[str, ...]
    plan_diff: PlanDiff
    material: bool
    plan_version: int
    schema_version: int = REPLANNING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.before_campaign_id, "previous campaign ID")
        if self.after_campaign_id is not None:
            _identifier(self.after_campaign_id, "new campaign ID")
        _identifier(self.before_plan_id, "previous plan ID")
        _identifier(self.after_plan_id, "new plan ID")
        _text(self.before_plan_digest, "previous plan digest")
        _text(self.after_plan_digest, "new plan digest")
        _identifier(self.before_spec_identity, "previous spec identity")
        _identifier(self.after_spec_identity, "new spec identity")
        if self.changed_paths != tuple(sorted(set(self.changed_paths))):
            raise ReplanningError("replan diff paths must be sorted and unique")
        if self.stale_reasons != tuple(sorted(set(self.stale_reasons))):
            raise ReplanningError("replan stale reasons must be sorted and unique")
        if self.material != bool(self.changed_paths):
            raise ReplanningError("replan materiality does not match changed paths")
        if self.material and self.after_campaign_id is None:
            raise ReplanningError("material replans require a child campaign ID")
        if not isinstance(self.plan_version, int) or self.plan_version < 1:
            raise ReplanningError("plan version must be positive")
        if self.schema_version != REPLANNING_SCHEMA_VERSION:
            raise ReplanningError("unsupported replan diff schema")

    @property
    def diff_id(self) -> str:
        return "replan_diff_" + _sha256(self.to_record(include_id=False))

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "record_type": "versioned_plan_diff",
            "schema_version": self.schema_version,
            "plan_version": self.plan_version,
            "before_campaign_id": self.before_campaign_id,
            "after_campaign_id": self.after_campaign_id,
            "before_plan_id": self.before_plan_id,
            "after_plan_id": self.after_plan_id,
            "before_plan_digest": self.before_plan_digest,
            "after_plan_digest": self.after_plan_digest,
            "before_spec_identity": self.before_spec_identity,
            "after_spec_identity": self.after_spec_identity,
            "changed_paths": list(self.changed_paths),
            "stale_reasons": list(self.stale_reasons),
            "plan_diff": self.plan_diff.to_record(),
            "material": self.material,
        }
        if include_id:
            record["diff_id"] = self.diff_id
        return record


@dataclass(frozen=True, slots=True)
class ReplanProposal:
    """Pure proposal returned before a material child is persisted."""

    parent: CampaignState
    diff: ReplanDiff
    stale: StaleContextReport
    preserved_evidence_ids: tuple[str, ...]
    approval_request: ApprovalRequest | None
    objective_amendment_id: str | None
    transaction_id: str

    @property
    def material(self) -> bool:
        return self.diff.material

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "campaign_replan_proposal",
            "schema_version": REPLANNING_SCHEMA_VERSION,
            "parent_campaign_id": self.parent.campaign_id,
            "diff": self.diff.to_record(),
            "stale_context": self.stale.to_record(),
            "preserved_evidence_ids": list(self.preserved_evidence_ids),
            "approval_request": None
            if self.approval_request is None
            else self.approval_request.to_record(),
            "objective_amendment_id": self.objective_amendment_id,
            "transaction_id": self.transaction_id,
        }


@dataclass(frozen=True, slots=True)
class ReplanResult:
    """Result of a no-op or an applied material replan."""

    proposal: ReplanProposal
    state: CampaignState
    child_campaign_id: str | None
    approval_decision_id: str | None

    @property
    def material(self) -> bool:
        return self.proposal.material

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "campaign_replan_result",
            "schema_version": REPLANNING_SCHEMA_VERSION,
            "material": self.material,
            "parent_campaign_id": self.proposal.parent.campaign_id,
            "child_campaign_id": self.child_campaign_id,
            "approval_decision_id": self.approval_decision_id,
            "proposal": self.proposal.to_record(),
            "state": self.state.to_record(),
        }


def _changed_paths(
    before: CampaignState,
    current_plan: object,
    candidate_plan: object,
    candidate_spec: CampaignSpec,
    candidate_provider_context: Mapping[str, object],
    candidate_budget: CampaignBudget,
    *,
    stale: StaleContextReport,
    objective_amendment: ObjectiveAmendment | None,
) -> tuple[tuple[str, ...], PlanDiff]:
    plan_diff = diff_plans(current_plan, candidate_plan)
    changed: set[str] = {f"plan.{path}" for path in plan_diff.changed_paths}
    if before.spec.spec_digest != candidate_spec.spec_digest or before.spec.spec_identity != (
        candidate_spec.spec_identity
    ):
        changed.add("spec")
    if _digest(before.provider_context) != _digest(candidate_provider_context):
        changed.add("provider_context")
    if _digest(before.budget.to_record()) != _digest(candidate_budget.to_record()):
        changed.add("resource_budget")
    if objective_amendment is not None and objective_amendment.diff.material:
        changed.add("objective_amendment")
    # New evidence can make the context stale, but it is not itself a material
    # plan change when the exact approved plan remains unchanged.
    return tuple(sorted(changed)), plan_diff


def _derive_child_run_id(
    parent: CampaignState,
    candidate_plan_id: str,
    candidate_plan_digest: str,
    candidate_spec_identity: str,
    changed_paths: tuple[str, ...],
    transaction_id: str,
) -> str:
    return "run_" + _sha256(
        {
            "parent_campaign_id": parent.campaign_id,
            "parent_run_id": parent.run_id,
            "candidate_plan_id": candidate_plan_id,
            "candidate_plan_digest": candidate_plan_digest,
            "candidate_spec_identity": candidate_spec_identity,
            "changed_paths": list(changed_paths),
            "transaction_id": transaction_id,
        }
    )


def build_replan_proposal(
    store: CampaignStateStore,
    *,
    campaign_id: str,
    expected_context: ReplanContext,
    current_plan: object,
    candidate_plan: object,
    candidate_spec: CampaignSpec,
    candidate_provider_context: Mapping[str, object],
    candidate_budget: CampaignBudget,
    transaction_id: str,
    approval_request: ApprovalRequest | None = None,
    amendment: ObjectiveAmendment | None = None,
) -> ReplanProposal:
    """Build a deterministic replan proposal without mutating the store."""

    parent = store.load(campaign_id)
    if parent.lifecycle is CampaignLifecycle.CANCELLED:
        raise ReplanReplayError("cancelled campaigns cannot be replanned")
    evidence = store.evidence(campaign_id)
    current_context = context_from_campaign(
        parent,
        current_plan,
        evidence,
        transaction_id=transaction_id,
    )
    stale = detect_stale_context(expected_context, current_context)
    if stale.stale:
        raise StaleContextError(stale)
    if expected_context.campaign_id != campaign_id:
        raise StaleContextError(
            StaleContextReport(
                expected_context,
                current_context,
                (StaleContextReason.SPEC.value,),
            )
        )
    if amendment is not None:
        status = getattr(amendment, "status", None)
        status_value = getattr(status, "value", status)
        if status_value not in {"approved", "applied"}:
            raise ReplanReplayError(
                "only approved objective amendments can gate a replan"
            )
        if amendment.original_spec_identity != parent.spec_identity:
            raise ReplanReplayError("objective amendment is stale for the current campaign spec")
        if amendment.proposed_spec_identity != candidate_spec.spec_identity:
            raise ReplanReplayError("candidate spec does not match the objective amendment")
    changed_paths, plan_diff = _changed_paths(
        parent,
        current_plan,
        candidate_plan,
        candidate_spec,
        candidate_provider_context,
        candidate_budget,
        stale=stale,
        objective_amendment=amendment,
    )
    # Evidence-only refresh is a valid no-op.  The current state remains the
    # authority and no approval is consumed.
    material = bool(changed_paths)
    plan_record = _record(candidate_plan)
    candidate_plan_id = _identifier(plan_record.get("plan_id"), "candidate plan ID")
    candidate_digest = plan_digest(plan_record)
    if not material and (
        candidate_plan_id != current_context.plan_id
        or candidate_digest != current_context.plan_digest
    ):
        material = True
        changed_paths = tuple(sorted((*changed_paths, "plan.identity")))
    child_campaign_id: str | None = None
    plan_version = 1
    if material:
        run_id = _derive_child_run_id(
            parent,
            candidate_plan_id,
            candidate_digest,
            candidate_spec.spec_identity,
            changed_paths,
            transaction_id,
        )
        child_campaign_id = derive_campaign_id(
            session_id=parent.session_id,
            run_id=run_id,
            source_model_digest=parent.source_model_digest,
            spec_identity=candidate_spec.spec_identity,
            spec_digest=candidate_spec.spec_digest,
        )
        previous_version = parent.policy_state.get("plan_version", 0)
        if isinstance(previous_version, bool) or not isinstance(previous_version, int):
            raise ReplanningError("parent plan version is not a canonical integer")
        plan_version = previous_version + 1
    diff = ReplanDiff(
        parent.campaign_id,
        child_campaign_id,
        current_context.plan_id,
        candidate_plan_id,
        current_context.plan_digest,
        candidate_digest,
        parent.spec_identity,
        candidate_spec.spec_identity,
        changed_paths,
        stale.reasons,
        plan_diff,
        material,
        plan_version,
    )
    if material:
        if approval_request is not None and (
            approval_request.plan_id != candidate_plan_id
            or approval_request.plan_digest != candidate_digest
        ):
            raise ReplanningError("replan approval request is bound to a different plan")
        if approval_request is not None and approval_request.diff_id != diff.diff_id:
            raise ReplanningError("replan approval request is bound to a different diff")
    elif approval_request is not None:
        raise ReplanningError("no-op replans must not request a new approval")
    preserved = tuple(sorted(item.evidence_id for item in evidence))
    amendment_id = None if amendment is None else amendment.amendment_id
    return ReplanProposal(
        parent, diff, stale, preserved, approval_request, amendment_id, transaction_id
    )


def build_replan_approval_request(
    proposal: ReplanProposal,
    *,
    requested_at: str,
    expires_at: str,
    operator_id: str,
    operator_context: Mapping[str, str] | None = None,
) -> ApprovalRequest:
    """Create the fresh approval request bound to one material diff."""

    if not proposal.material:
        raise ReplanningError("no-op replans do not need approval")
    if _timestamp(expires_at, "approval expiry") <= _timestamp(
        requested_at, "approval request time"
    ):
        raise ReplanningError("approval expiry must be after its request")
    return ApprovalRequest(
        "campaign_replan",
        proposal.diff.after_plan_id,
        proposal.diff.after_plan_digest,
        tuple(sorted({"campaign_replan", *proposal.diff.changed_paths})),
        proposal.diff.diff_id,
        requested_at,
        expires_at,
        operator_id,
        tuple(sorted((str(key), str(value)) for key, value in (operator_context or {}).items())),
    )


def _approved_campaign_approval(
    request: ApprovalRequest,
    decision: ApprovalDecision,
    spec_digest: str,
) -> CampaignApproval:
    try:
        validate_approval(
            request,
            decision,
            current_plan_id=request.plan_id,
            current_plan_digest=request.plan_digest,
            at=_timestamp(decision.decided_at, "approval decision time"),
        )
    except OptimizationPackageError as error:
        raise ReplanningError(str(error)) from error
    return CampaignApproval(
        ApprovalStatus.APPROVED,
        spec_digest,
        approval_id=decision.decision_id,
        recorded_by=decision.operator_id,
        expires_at=request.expires_at,
        provenance={
            "record_type": "campaign_replan_approval",
            "request_id": request.request_id,
            "decision_id": decision.decision_id,
            "diff_id": request.diff_id,
        },
    )


def apply_replan(
    store: CampaignStateStore,
    proposal: ReplanProposal,
    *,
    candidate_spec: CampaignSpec,
    candidate_policy_state: Mapping[str, object],
    candidate_provider_context: Mapping[str, object],
    candidate_budget: CampaignBudget,
    approval_decision: ApprovalDecision | None = None,
    run_id: str | None = None,
) -> ReplanResult:
    """Persist a no-op or a fresh approved/pending child campaign."""

    if not proposal.material:
        return ReplanResult(proposal, store.load(proposal.parent.campaign_id), None, None)
    request = proposal.approval_request
    if request is None:
        raise ReplanApprovalRequired(proposal.diff)
    selected_run_id = run_id or _derive_child_run_id(
        proposal.parent,
        proposal.diff.after_plan_id,
        proposal.diff.after_plan_digest,
        candidate_spec.spec_identity,
        proposal.diff.changed_paths,
        proposal.transaction_id,
    )
    child_id = derive_campaign_id(
        session_id=proposal.parent.session_id,
        run_id=selected_run_id,
        source_model_digest=proposal.parent.source_model_digest,
        spec_identity=candidate_spec.spec_identity,
        spec_digest=candidate_spec.spec_digest,
    )
    if proposal.diff.after_campaign_id != child_id:
        raise ReplanningError("candidate run identity does not match the canonical replan ID")
    approval = CampaignApproval.pending(candidate_spec.spec_digest)
    decision_id: str | None = None
    if approval_decision is not None:
        approval = _approved_campaign_approval(
            request, approval_decision, candidate_spec.spec_digest
        )
        decision_id = approval_decision.decision_id
    plan_state = _mapping(candidate_policy_state, "candidate policy state")
    plan_state.update(
        {
            "record_type": "conversational_replan_policy",
            "plan_version": proposal.diff.plan_version,
            "plan_diff_id": proposal.diff.diff_id,
            "parent_campaign_id": proposal.parent.campaign_id,
            "parent_plan_id": proposal.diff.before_plan_id,
            "current_plan_id": proposal.diff.after_plan_id,
            "transaction_id": proposal.transaction_id,
            "preserved_evidence_ids": list(proposal.preserved_evidence_ids),
        }
    )
    child = new_campaign_state(
        session_id=proposal.parent.session_id,
        run_id=selected_run_id,
        source_model_digest=proposal.parent.source_model_digest,
        spec=candidate_spec,
        policy_state=plan_state,
        provider_context=candidate_provider_context,
        budget=candidate_budget,
        approval=approval,
        campaign_id=child_id,
        provenance={
            "record_type": "conversational_campaign_replan",
            "parent_campaign_id": proposal.parent.campaign_id,
            "parent_state_version": proposal.parent.state_version,
            "parent_plan_id": proposal.diff.before_plan_id,
            "child_plan_id": proposal.diff.after_plan_id,
            "plan_version": proposal.diff.plan_version,
            "plan_diff": proposal.diff.to_record(),
            "preserved_evidence_ids": list(proposal.preserved_evidence_ids),
            "transaction_id": proposal.transaction_id,
            "approval_request_id": request.request_id,
            "approval_decision_id": decision_id,
            "objective_amendment_id": proposal.objective_amendment_id,
        },
    )
    try:
        existing = store.load(child_id)
    except CampaignStateError as error:
        if not str(error).startswith("unknown campaign"):
            raise
        parent = store.load(proposal.parent.campaign_id)
        if parent.state_version != proposal.parent.state_version:
            raise ReplanReplayError("replan parent changed before replay") from None
        store.transition(
            parent.campaign_id,
            expected_version=parent.state_version,
            kind="replan_requested",
            provenance={
                "record_type": "campaign_replan_request",
                "child_campaign_id": child_id,
                "plan_diff_id": proposal.diff.diff_id,
                "transaction_id": proposal.transaction_id,
                "preserved_evidence_ids": list(proposal.preserved_evidence_ids),
            },
        )
        existing = store.create(child)
    else:
        if existing.provenance.get("plan_diff") != proposal.diff.to_record():
            raise ReplanReplayError("replan child identity conflicts with immutable history")
    return ReplanResult(proposal, existing, child_id, decision_id)


def replan_campaign(
    store: CampaignStateStore,
    *,
    campaign_id: str,
    expected_context: ReplanContext,
    current_plan: object,
    candidate_plan: object,
    candidate_spec: CampaignSpec,
    candidate_policy_state: Mapping[str, object],
    candidate_provider_context: Mapping[str, object],
    candidate_budget: CampaignBudget,
    transaction_id: str,
    approval_request: ApprovalRequest | None = None,
    approval_decision: ApprovalDecision | None = None,
    amendment: ObjectiveAmendment | None = None,
    run_id: str | None = None,
) -> ReplanResult:
    """Build and apply a deterministic replan in one explicit operation."""

    proposal = build_replan_proposal(
        store,
        campaign_id=campaign_id,
        expected_context=expected_context,
        current_plan=current_plan,
        candidate_plan=candidate_plan,
        candidate_spec=candidate_spec,
        candidate_provider_context=candidate_provider_context,
        candidate_budget=candidate_budget,
        transaction_id=transaction_id,
        approval_request=approval_request,
        amendment=amendment,
    )
    return apply_replan(
        store,
        proposal,
        candidate_spec=candidate_spec,
        candidate_policy_state=candidate_policy_state,
        candidate_provider_context=candidate_provider_context,
        candidate_budget=candidate_budget,
        approval_decision=approval_decision,
        run_id=run_id,
    )


def replay_replan(
    store: CampaignStateStore,
    proposal: ReplanProposal,
    *,
    candidate_spec: CampaignSpec,
    candidate_policy_state: Mapping[str, object],
    candidate_provider_context: Mapping[str, object],
    candidate_budget: CampaignBudget,
    approval_decision: ApprovalDecision | None = None,
    run_id: str | None = None,
) -> ReplanResult:
    """Replay an exact proposal, rejecting stale or cancelled campaigns."""

    current = store.load(proposal.parent.campaign_id)
    if current.lifecycle is CampaignLifecycle.CANCELLED:
        raise ReplanReplayError("cancelled campaigns cannot be replayed")
    selected_run_id = run_id or _derive_child_run_id(
        proposal.parent,
        proposal.diff.after_plan_id,
        proposal.diff.after_plan_digest,
        candidate_spec.spec_identity,
        proposal.diff.changed_paths,
        proposal.transaction_id,
    )
    child_id = derive_campaign_id(
        session_id=proposal.parent.session_id,
        run_id=selected_run_id,
        source_model_digest=proposal.parent.source_model_digest,
        spec_identity=candidate_spec.spec_identity,
        spec_digest=candidate_spec.spec_digest,
    )
    try:
        store.load(child_id)
    except CampaignStateError as error:
        if not str(error).startswith("unknown campaign"):
            raise
        if current.state_version != proposal.parent.state_version:
            raise ReplanReplayError("replan proposal is stale") from None
    return apply_replan(
        store,
        proposal,
        candidate_spec=candidate_spec,
        candidate_policy_state=candidate_policy_state,
        candidate_provider_context=candidate_provider_context,
        candidate_budget=candidate_budget,
        approval_decision=approval_decision,
        run_id=run_id,
    )


__all__ = [
    "REPLANNING_SCHEMA_VERSION",
    "ReplanApprovalRequired",
    "ReplanContext",
    "ReplanDiff",
    "ReplanProposal",
    "ReplanReplayError",
    "ReplanResult",
    "ReplanningError",
    "StaleContextError",
    "StaleContextReason",
    "StaleContextReport",
    "apply_replan",
    "build_replan_approval_request",
    "build_replan_proposal",
    "context_from_campaign",
    "detect_stale_context",
    "replan_campaign",
    "replay_replan",
]

"""Immutable, approval-bound amendments to objective contracts.

An amendment is a control-plane record, not an in-place edit.  It captures the
contract that produced the evidence, the proposed contract, the exact visible
diff, the existing v2 approval request/decision, and the evidence archive that
motivated the proposal.  Applying a material amendment creates a new campaign
identity; the original campaign and evidence remain queryable by their original
identities.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import TYPE_CHECKING, cast

from modelsurgeon.conversation.campaign_state import derive_campaign_id
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.optimization_package import (
    ApprovalDecision,
    ApprovalDecisionKind,
    ApprovalRequest,
    OptimizationPackageError,
    validate_approval,
)
from modelsurgeon.search.objective_contract import HardConstraint, ObjectiveContract

if TYPE_CHECKING:
    from modelsurgeon.explain.infeasibility import FeasibilityExplanation

OBJECTIVE_AMENDMENT_SCHEMA_VERSION = 1


class ObjectiveAmendmentError(ValueError):
    """Raised when an amendment is ambiguous, stale, or not approval-bound."""


class ObjectiveAmendmentStatus(StrEnum):
    PENDING = "pending"
    PROPOSED = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    CANCELED = "cancelled"
    APPLIED = "applied"


AmendmentStatus = ObjectiveAmendmentStatus


def _canonical(value: object) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ObjectiveAmendmentError("amendment value is not canonical JSON") from error


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObjectiveAmendmentError(f"{label} is required")
    return value


def _identifier(value: object, label: str) -> str:
    return _text(value, label)


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _content_digest(value: object) -> str:
    return f"sha256:{_sha256(value)}"


def _record(value: object, label: str) -> dict[str, object]:
    if hasattr(value, "to_record"):
        value = value.to_record()
    if not isinstance(value, Mapping):
        raise ObjectiveAmendmentError(f"{label} must be a JSON object")
    try:
        decoded = json.loads(_canonical(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ObjectiveAmendmentError(f"{label} must contain canonical JSON") from error
    if not isinstance(decoded, dict):
        raise ObjectiveAmendmentError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)


def _timestamp(value: object, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ObjectiveAmendmentError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ObjectiveAmendmentError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], path))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}[{index}]"))
        return result
    return {prefix: value}


def _approval_contract_record(contract: ObjectiveContract) -> dict[str, object]:
    record = contract.to_record()
    record.pop("contract_id", None)
    return record


def _approval_contract_digest(contract: ObjectiveContract) -> str:
    return _sha256(_approval_contract_record(contract))


def _required_scope(diff: ObjectiveAmendmentDiff) -> tuple[str, ...]:
    required = {"objective_amendment", *diff.changed_paths}
    if diff.hard_constraints_changed:
        required.add("hard_constraints")
    return tuple(sorted(required))


@dataclass(frozen=True, slots=True)
class ObjectiveAmendmentDiff:
    """Canonical contract diff bound into the approval request."""

    original_spec_identity: str
    proposed_spec_identity: str
    original_digest: str
    proposed_digest: str
    changed_paths: tuple[str, ...]
    material: bool
    schema_version: int = OBJECTIVE_AMENDMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.original_spec_identity, "original spec identity")
        _identifier(self.proposed_spec_identity, "proposed spec identity")
        for value, label in (
            (self.original_digest, "original objective digest"),
            (self.proposed_digest, "proposed objective digest"),
        ):
            if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
                raise ObjectiveAmendmentError(f"{label} must be a SHA-256 content digest")
        if self.changed_paths != tuple(sorted(set(self.changed_paths))):
            raise ObjectiveAmendmentError("objective diff paths must be sorted and unique")
        if self.material != bool(self.changed_paths):
            raise ObjectiveAmendmentError("objective diff materiality does not match paths")
        if self.schema_version != OBJECTIVE_AMENDMENT_SCHEMA_VERSION:
            raise ObjectiveAmendmentError("unsupported objective amendment diff schema")

    @property
    def diff_id(self) -> str:
        return "objective_diff_" + _sha256(self.to_record(include_id=False))

    @property
    def hard_constraints_changed(self) -> bool:
        return any(
            path.startswith("constraints") or path.startswith("objective.constraints")
            for path in self.changed_paths
        )

    @property
    def non_material(self) -> bool:
        return not self.material

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "record_type": "objective_amendment_diff",
            "schema_version": self.schema_version,
            "original_spec_identity": self.original_spec_identity,
            "proposed_spec_identity": self.proposed_spec_identity,
            "original_digest": self.original_digest,
            "proposed_digest": self.proposed_digest,
            "changed_paths": list(self.changed_paths),
            "material": self.material,
        }
        if include_id:
            record["diff_id"] = self.diff_id
        return record


ObjectiveDiff = ObjectiveAmendmentDiff


def diff_objective_contracts(
    original: ObjectiveContract, proposed: ObjectiveContract
) -> ObjectiveAmendmentDiff:
    """Return a deterministic visible diff between two objective contracts."""

    before = _approval_contract_record(original)
    after = _approval_contract_record(proposed)
    before_flat = _flatten(before, "objective")
    after_flat = _flatten(after, "objective")
    changed = tuple(
        sorted(
            path
            for path in set(before_flat) | set(after_flat)
            if before_flat.get(path) != after_flat.get(path)
        )
    )
    return ObjectiveAmendmentDiff(
        original.contract_id,
        proposed.contract_id,
        _content_digest(before),
        _content_digest(after),
        changed,
        bool(changed),
    )


@dataclass(frozen=True, slots=True)
class AmendmentCancellation:
    """Immutable cancellation event for a proposal that will not be applied."""

    operator_id: str
    cancelled_at: str
    reason: str

    def __post_init__(self) -> None:
        _text(self.operator_id, "cancellation operator")
        _timestamp(self.cancelled_at, "cancellation time")
        _text(self.reason, "cancellation reason")

    @property
    def cancellation_id(self) -> str:
        return "amendment_cancellation_" + _sha256(self.to_record(include_id=False))

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "operator_id": self.operator_id,
            "cancelled_at": self.cancelled_at,
            "reason": self.reason,
        }
        if include_id:
            record["cancellation_id"] = self.cancellation_id
        return record


@dataclass(frozen=True, slots=True)
class ObjectiveAmendmentApplication:
    """The immutable result of applying an approved amendment."""

    amendment_id: str
    diff_id: str
    material: bool
    original_spec_identity: str
    effective_spec_identity: str
    original_campaign_id: str | None
    downstream_campaign_id: str | None
    source_model_digest: str
    evidence_archive_id: str
    preserved_evidence_ids: tuple[str, ...]
    approval_id: str
    applied_at: str

    def __post_init__(self) -> None:
        _identifier(self.amendment_id, "amendment ID")
        _identifier(self.diff_id, "amendment diff ID")
        _identifier(self.original_spec_identity, "original spec identity")
        _identifier(self.effective_spec_identity, "effective spec identity")
        _text(self.source_model_digest, "source model digest")
        _identifier(self.evidence_archive_id, "evidence archive ID")
        _identifier(self.approval_id, "approval decision ID")
        _timestamp(self.applied_at, "amendment application time")
        if self.preserved_evidence_ids != tuple(sorted(set(self.preserved_evidence_ids))):
            raise ObjectiveAmendmentError("preserved evidence IDs must be sorted and unique")
        if self.material and self.downstream_campaign_id is None:
            raise ObjectiveAmendmentError("material amendments require a downstream campaign")
        if not self.material and self.effective_spec_identity != self.original_spec_identity:
            raise ObjectiveAmendmentError("non-material amendments cannot change spec identity")

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "objective_amendment_application",
            "schema_version": OBJECTIVE_AMENDMENT_SCHEMA_VERSION,
            "amendment_id": self.amendment_id,
            "diff_id": self.diff_id,
            "material": self.material,
            "original_spec_identity": self.original_spec_identity,
            "effective_spec_identity": self.effective_spec_identity,
            "original_campaign_id": self.original_campaign_id,
            "downstream_campaign_id": self.downstream_campaign_id,
            "source_model_digest": self.source_model_digest,
            "evidence_archive_id": self.evidence_archive_id,
            "preserved_evidence_ids": list(self.preserved_evidence_ids),
            "approval_id": self.approval_id,
            "applied_at": self.applied_at,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveAmendment:
    """An append-only proposal and its approval/application state."""

    original_objective: ObjectiveContract
    proposed_amendment: ObjectiveContract
    rationale: str
    evidence: FeasibilityExplanation
    approval_request: ApprovalRequest
    diff: ObjectiveAmendmentDiff
    source_model_digest: str
    parent_campaign_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    status: ObjectiveAmendmentStatus = ObjectiveAmendmentStatus.PENDING
    decision: ApprovalDecision | None = None
    cancellation: AmendmentCancellation | None = None
    application: ObjectiveAmendmentApplication | None = None
    schema_version: int = OBJECTIVE_AMENDMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.rationale, "amendment rationale")
        _text(self.source_model_digest, "amendment source model digest")
        if self.schema_version != OBJECTIVE_AMENDMENT_SCHEMA_VERSION:
            raise ObjectiveAmendmentError("unsupported objective amendment schema")
        if self.diff != diff_objective_contracts(self.original_objective, self.proposed_amendment):
            raise ObjectiveAmendmentError("amendment diff does not match its contracts")
        if self.evidence.contract_id != self.original_objective.contract_id:
            raise ObjectiveAmendmentError("amendment evidence belongs to a different objective")
        if self.evidence.provenance.source_model_digest != self.source_model_digest:
            raise ObjectiveAmendmentError("amendment evidence source does not match")
        if self.approval_request.plan_id != self.proposed_amendment.contract_id:
            raise ObjectiveAmendmentError("approval request is not bound to the proposed spec")
        if self.approval_request.plan_digest != _approval_contract_digest(self.proposed_amendment):
            raise ObjectiveAmendmentError("approval request digest does not match the proposal")
        if self.approval_request.diff_id != self.diff.diff_id:
            raise ObjectiveAmendmentError("approval request is not bound to the amendment diff")
        if not set(_required_scope(self.diff)).issubset(self.approval_request.scope):
            raise ObjectiveAmendmentError(
                "approval scope does not cover the visible amendment diff"
            )
        if self.parent_campaign_id is not None:
            _identifier(self.parent_campaign_id, "parent campaign ID")
        if self.session_id is not None:
            _identifier(self.session_id, "amendment session ID")
        if self.run_id is not None:
            _identifier(self.run_id, "amendment run ID")
        if not isinstance(self.status, ObjectiveAmendmentStatus):
            raise ObjectiveAmendmentError("amendment status is invalid")
        if self.status is ObjectiveAmendmentStatus.PENDING and (
            self.decision is not None
            or self.cancellation is not None
            or self.application is not None
        ):
            raise ObjectiveAmendmentError("pending amendments cannot have terminal events")
        if self.status is ObjectiveAmendmentStatus.APPLIED:
            if self.decision is None or self.decision.decision is not ApprovalDecisionKind.APPROVED:
                raise ObjectiveAmendmentError("applied amendments require an approval decision")
            if self.application is None or self.application.amendment_id != self.amendment_id:
                raise ObjectiveAmendmentError("applied amendments require application evidence")
        if self.status is ObjectiveAmendmentStatus.APPROVED:
            if self.decision is None or self.decision.decision is not ApprovalDecisionKind.APPROVED:
                raise ObjectiveAmendmentError("approved amendments require an approval decision")
            if self.cancellation is not None or self.application is not None:
                raise ObjectiveAmendmentError("approved amendments cannot have terminal events")
        if (
            self.status is ObjectiveAmendmentStatus.REJECTED
            and (
                self.decision is None
                or self.decision.decision is not ApprovalDecisionKind.REJECTED
            )
        ):
            raise ObjectiveAmendmentError("rejected amendments require a rejection decision")
        if (
            self.status is ObjectiveAmendmentStatus.EXPIRED
            and (
                self.decision is None
                or self.decision.decision is not ApprovalDecisionKind.EXPIRED
            )
        ):
            raise ObjectiveAmendmentError("expired amendments require an expiry decision")
        if self.status is ObjectiveAmendmentStatus.CANCELLED and self.cancellation is None:
            raise ObjectiveAmendmentError("cancelled amendments require a cancellation event")

    @property
    def amendment_id(self) -> str:
        return "objective_amendment_" + _sha256(self._identity_record())

    @property
    def original_spec_identity(self) -> str:
        return self.original_objective.contract_id

    @property
    def proposed_spec_identity(self) -> str:
        return self.proposed_amendment.contract_id

    @property
    def effective_spec_identity(self) -> str:
        return self.proposed_spec_identity if self.diff.material else self.original_spec_identity

    @property
    def approval_scope(self) -> tuple[str, ...]:
        return self.approval_request.scope

    @property
    def approval_provenance(self) -> dict[str, object]:
        return {
            "request_id": self.approval_request.request_id,
            "operator_id": self.approval_request.operator_id,
            "operator_context": dict(self.approval_request.operator_context),
            "scope": list(self.approval_request.scope),
            "diff_id": self.diff.diff_id,
            "decision_id": None if self.decision is None else self.decision.decision_id,
        }

    @property
    def approval_id(self) -> str | None:
        return None if self.decision is None else self.decision.decision_id

    @property
    def preserved_evidence_ids(self) -> tuple[str, ...]:
        if self.evidence.provenance.evidence_ids:
            return self.evidence.provenance.evidence_ids
        return tuple(sorted(item.evidence_id for item in self.evidence.retained_evidence))

    @property
    def evidence_archive_id(self) -> str:
        return self.evidence.provenance.archive_id

    @property
    def hard_constraints(self) -> tuple[HardConstraint, ...]:
        """The original hard constraints, retained without in-place mutation."""

        return self.original_objective.constraints

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "original_objective": self.original_objective.to_record(),
            "proposed_amendment": self.proposed_amendment.to_record(),
            "rationale": self.rationale,
            "evidence_result_id": self.evidence.result_id,
            "evidence_archive_id": self.evidence_archive_id,
            "preserved_evidence_ids": list(self.preserved_evidence_ids),
            "approval_request": self.approval_request.to_record(),
            "diff": self.diff.to_record(),
            "source_model_digest": self.source_model_digest,
            "parent_campaign_id": self.parent_campaign_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
        }

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "objective_amendment",
            "schema_version": self.schema_version,
            "amendment_id": self.amendment_id,
            "original_objective": self.original_objective.to_record(),
            "proposed_amendment": self.proposed_amendment.to_record(),
            "rationale": self.rationale,
            "evidence": self.evidence.to_record(),
            "approval_request": self.approval_request.to_record(),
            "approval_provenance": self.approval_provenance,
            "diff": self.diff.to_record(),
            "source_model_digest": self.source_model_digest,
            "parent_campaign_id": self.parent_campaign_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "decision": None if self.decision is None else self.decision.to_record(),
            "cancellation": None
            if self.cancellation is None
            else self.cancellation.to_record(),
            "application": None if self.application is None else self.application.to_record(),
            "effective_spec_identity": self.effective_spec_identity,
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


def propose_objective_amendment(
    original_objective: ObjectiveContract,
    proposed_amendment: ObjectiveContract,
    *,
    rationale: str,
    evidence: FeasibilityExplanation,
    operator_id: str,
    requested_at: str,
    expires_at: str,
    scope: Sequence[str] | None = None,
    operator_context: Mapping[str, str] | None = None,
    source_model_digest: str | None = None,
    parent_campaign_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
) -> ObjectiveAmendment:
    """Create a pending amendment with an exact v2 approval request."""

    diff = diff_objective_contracts(original_objective, proposed_amendment)
    requested_scope = set(_required_scope(diff))
    if scope is not None:
        supplied_scope = tuple(scope)
        if supplied_scope != tuple(sorted(set(supplied_scope))):
            raise ObjectiveAmendmentError("approval scope must be sorted and unique")
        if not requested_scope.issubset(supplied_scope):
            raise ObjectiveAmendmentError("approval scope must cover every changed path")
        requested_scope = set(supplied_scope)
    selected_source = source_model_digest or evidence.provenance.source_model_digest
    approval = ApprovalRequest(
        "objective_amendment",
        proposed_amendment.contract_id,
        _approval_contract_digest(proposed_amendment),
        tuple(sorted(requested_scope)),
        diff.diff_id,
        requested_at,
        expires_at,
        operator_id,
        tuple(sorted((str(key), str(value)) for key, value in (operator_context or {}).items())),
    )
    try:
        return ObjectiveAmendment(
            original_objective,
            proposed_amendment,
            rationale,
            evidence,
            approval,
            diff,
            selected_source,
            parent_campaign_id,
            session_id,
            run_id,
        )
    except OptimizationPackageError as error:
        raise ObjectiveAmendmentError(str(error)) from error


def _require_pending(amendment: ObjectiveAmendment, action: str) -> None:
    if amendment.status is not ObjectiveAmendmentStatus.PENDING:
        raise ObjectiveAmendmentError(
            f"cannot {action} amendment in {amendment.status.value} state"
        )


def _decision_binding(amendment: ObjectiveAmendment, decision: ApprovalDecision) -> None:
    if decision.request_id != amendment.approval_request.request_id:
        raise ObjectiveAmendmentError("approval decision does not match the amendment request")
    if decision.code != amendment.approval_request.code:
        raise ObjectiveAmendmentError("approval decision code does not match the amendment")
    if decision.plan_id != amendment.proposed_spec_identity:
        raise ObjectiveAmendmentError("approval decision is bound to a different effective spec")
    if decision.plan_digest != amendment.approval_request.plan_digest:
        raise ObjectiveAmendmentError("approval decision digest is stale")
    if decision.diff_id != amendment.diff.diff_id:
        raise ObjectiveAmendmentError("approval decision is bound to a different diff")


def approve_objective_amendment(
    amendment: ObjectiveAmendment,
    *,
    operator_id: str,
    decided_at: str,
    reason: str = "explicitly approved by operator",
) -> ObjectiveAmendment:
    """Approve the exact pending proposal, reusing v2 approval validation."""

    _require_pending(amendment, "approve")
    decision = ApprovalDecision(
        amendment.approval_request.request_id,
        amendment.approval_request.code,
        amendment.proposed_spec_identity,
        amendment.approval_request.plan_digest,
        amendment.diff.diff_id,
        ApprovalDecisionKind.APPROVED,
        decided_at,
        operator_id,
        reason,
    )
    try:
        validate_approval(
            amendment.approval_request,
            decision,
            current_plan_id=amendment.proposed_spec_identity,
            current_plan_digest=amendment.approval_request.plan_digest,
            at=_timestamp(decided_at, "approval decision time"),
        )
    except OptimizationPackageError as error:
        raise ObjectiveAmendmentError(str(error)) from error
    return replace(amendment, status=ObjectiveAmendmentStatus.APPROVED, decision=decision)


def reject_objective_amendment(
    amendment: ObjectiveAmendment,
    *,
    operator_id: str,
    decided_at: str,
    reason: str,
) -> ObjectiveAmendment:
    """Record an explicit rejection without changing either objective."""

    _require_pending(amendment, "reject")
    decision = ApprovalDecision(
        amendment.approval_request.request_id,
        amendment.approval_request.code,
        amendment.proposed_spec_identity,
        amendment.approval_request.plan_digest,
        amendment.diff.diff_id,
        ApprovalDecisionKind.REJECTED,
        decided_at,
        operator_id,
        reason,
    )
    _decision_binding(amendment, decision)
    if _timestamp(decided_at, "rejection time") >= _timestamp(
        amendment.approval_request.expires_at, "approval expiry"
    ):
        raise ObjectiveAmendmentError("cannot reject an expired amendment")
    return replace(amendment, status=ObjectiveAmendmentStatus.REJECTED, decision=decision)


def expire_objective_amendment(
    amendment: ObjectiveAmendment,
    *,
    expired_at: str | None = None,
    reason: str = "approval window expired",
) -> ObjectiveAmendment:
    """Record expiry once the approval request is no longer usable."""

    if amendment.status not in {
        ObjectiveAmendmentStatus.PENDING,
        ObjectiveAmendmentStatus.APPROVED,
    }:
        raise ObjectiveAmendmentError(
            f"cannot expire amendment in {amendment.status.value} state"
        )
    selected = expired_at or amendment.approval_request.expires_at
    if _timestamp(selected, "amendment expiry time") < _timestamp(
        amendment.approval_request.expires_at, "approval expiry"
    ):
        raise ObjectiveAmendmentError("amendment cannot expire before its approval expiry")
    decision = ApprovalDecision(
        amendment.approval_request.request_id,
        amendment.approval_request.code,
        amendment.proposed_spec_identity,
        amendment.approval_request.plan_digest,
        amendment.diff.diff_id,
        ApprovalDecisionKind.EXPIRED,
        selected,
        amendment.approval_request.operator_id,
        reason,
    )
    _decision_binding(amendment, decision)
    return replace(amendment, status=ObjectiveAmendmentStatus.EXPIRED, decision=decision)


def cancel_objective_amendment(
    amendment: ObjectiveAmendment,
    *,
    operator_id: str,
    cancelled_at: str,
    reason: str,
) -> ObjectiveAmendment:
    """Cancel a pending or approved amendment without mutating the contract."""

    if amendment.status not in {
        ObjectiveAmendmentStatus.PENDING,
        ObjectiveAmendmentStatus.APPROVED,
    }:
        raise ObjectiveAmendmentError(
            f"cannot cancel amendment in {amendment.status.value} state"
        )
    return replace(
        amendment,
        status=ObjectiveAmendmentStatus.CANCELLED,
        cancellation=AmendmentCancellation(operator_id, cancelled_at, reason),
    )


def _application_campaign_id(
    amendment: ObjectiveAmendment,
    *,
    current_campaign_id: str | None,
    session_id: str | None,
    run_id: str | None,
) -> str | None:
    if not amendment.diff.material:
        return current_campaign_id or amendment.parent_campaign_id
    selected_session = session_id or amendment.session_id
    selected_run = run_id or amendment.run_id
    if selected_session is None or selected_run is None:
        raise ObjectiveAmendmentError(
            "material amendments require session and run identities for a new campaign"
        )
    return derive_campaign_id(
        session_id=selected_session,
        run_id=selected_run,
        source_model_digest=amendment.source_model_digest,
        spec_identity=amendment.effective_spec_identity,
        spec_digest=amendment.diff.proposed_digest,
    )


def apply_objective_amendment(
    amendment: ObjectiveAmendment,
    *,
    current_objective: ObjectiveContract,
    current_campaign_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    applied_at: str,
) -> ObjectiveAmendmentApplication:
    """Apply only an approved, non-stale amendment and retain prior evidence."""

    if amendment.status is ObjectiveAmendmentStatus.APPLIED and amendment.application is not None:
        return amendment.application
    if amendment.status is not ObjectiveAmendmentStatus.APPROVED:
        raise ObjectiveAmendmentError(
            f"cannot apply amendment in {amendment.status.value} state"
        )
    current_digest = _approval_contract_digest(current_objective)
    if (
        current_objective.contract_id != amendment.original_spec_identity
        or current_digest != amendment.diff.original_digest[7:]
    ):
        raise ObjectiveAmendmentError("stale amendment: current objective is not the original spec")
    decision = amendment.decision
    if decision is None:
        raise ObjectiveAmendmentError("approved amendment is missing its approval decision")
    try:
        validate_approval(
            amendment.approval_request,
            decision,
            current_plan_id=amendment.proposed_spec_identity,
            current_plan_digest=amendment.approval_request.plan_digest,
            at=_timestamp(applied_at, "amendment application time"),
        )
    except OptimizationPackageError as error:
        raise ObjectiveAmendmentError(str(error)) from error
    selected_campaign = _application_campaign_id(
        amendment,
        current_campaign_id=current_campaign_id,
        session_id=session_id,
        run_id=run_id,
    )
    if amendment.diff.material and selected_campaign == (
        current_campaign_id or amendment.parent_campaign_id
    ):
        raise ObjectiveAmendmentError("material amendment must create a new campaign identity")
    return ObjectiveAmendmentApplication(
        amendment.amendment_id,
        amendment.diff.diff_id,
        amendment.diff.material,
        amendment.original_spec_identity,
        amendment.effective_spec_identity,
        current_campaign_id or amendment.parent_campaign_id,
        selected_campaign,
        amendment.source_model_digest,
        amendment.evidence_archive_id,
        amendment.preserved_evidence_ids,
        decision.decision_id,
        applied_at,
    )


def replay_objective_amendment(
    amendment: ObjectiveAmendment,
    *,
    current_objective: ObjectiveContract,
    current_campaign_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    replayed_at: str,
) -> ObjectiveAmendmentApplication:
    """Replay an application idempotently, while rejecting stale proposals."""

    if amendment.status is ObjectiveAmendmentStatus.APPLIED and amendment.application is not None:
        return amendment.application
    return apply_objective_amendment(
        amendment,
        current_objective=current_objective,
        current_campaign_id=current_campaign_id,
        session_id=session_id,
        run_id=run_id,
        applied_at=replayed_at,
    )


_TERMINAL = {
    ObjectiveAmendmentStatus.REJECTED,
    ObjectiveAmendmentStatus.EXPIRED,
    ObjectiveAmendmentStatus.CANCELLED,
    ObjectiveAmendmentStatus.APPLIED,
}


def _valid_transition(
    before: ObjectiveAmendmentStatus, after: ObjectiveAmendmentStatus
) -> bool:
    return (before, after) in {
        (ObjectiveAmendmentStatus.PENDING, ObjectiveAmendmentStatus.APPROVED),
        (ObjectiveAmendmentStatus.PENDING, ObjectiveAmendmentStatus.REJECTED),
        (ObjectiveAmendmentStatus.PENDING, ObjectiveAmendmentStatus.EXPIRED),
        (ObjectiveAmendmentStatus.PENDING, ObjectiveAmendmentStatus.CANCELLED),
        (ObjectiveAmendmentStatus.APPROVED, ObjectiveAmendmentStatus.EXPIRED),
        (ObjectiveAmendmentStatus.APPROVED, ObjectiveAmendmentStatus.CANCELLED),
        (ObjectiveAmendmentStatus.APPROVED, ObjectiveAmendmentStatus.APPLIED),
    }


@dataclass(frozen=True, slots=True)
class ObjectiveAmendmentHistory:
    """Append-only immutable amendment history."""

    entries: tuple[ObjectiveAmendment, ...] = ()

    def __post_init__(self) -> None:
        for index, item in enumerate(self.entries):
            same_amendment = index and item.amendment_id == self.entries[index - 1].amendment_id
            if same_amendment and not _valid_transition(
                self.entries[index - 1].status, item.status
            ):
                raise ObjectiveAmendmentError(
                    "amendment history contains an invalid transition"
                )
            if any(
                earlier.amendment_id == item.amendment_id
                and earlier._identity_record() != item._identity_record()
                for earlier in self.entries[:index]
            ):
                raise ObjectiveAmendmentError("amendment history contains conflicting replay")

    def append(self, amendment: ObjectiveAmendment) -> ObjectiveAmendmentHistory:
        if self.entries and self.entries[-1].amendment_id == amendment.amendment_id:
            if self.entries[-1].canonical_json() == amendment.canonical_json():
                return self
            if not _valid_transition(self.entries[-1].status, amendment.status):
                raise ObjectiveAmendmentError("amendment replay or transition is stale")
        elif amendment.amendment_id in {item.amendment_id for item in self.entries}:
            raise ObjectiveAmendmentError("amendment history transitions must be contiguous")
        if self.entries and self.entries[-1].status in _TERMINAL:
            pass
        return replace(self, entries=(*self.entries, amendment))

    def inspect(self, amendment_id: str) -> ObjectiveAmendment:
        _identifier(amendment_id, "amendment ID")
        for item in reversed(self.entries):
            if item.amendment_id == amendment_id:
                return item
        raise ObjectiveAmendmentError(f"unknown amendment {amendment_id}")

    def for_campaign(self, campaign_id: str) -> tuple[ObjectiveAmendment, ...]:
        _identifier(campaign_id, "campaign ID")
        return tuple(
            item
            for item in self.entries
            if item.parent_campaign_id == campaign_id
            or (
                item.application is not None
                and item.application.downstream_campaign_id == campaign_id
            )
        )

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "objective_amendment_history",
            "schema_version": OBJECTIVE_AMENDMENT_SCHEMA_VERSION,
            "entries": [item.to_record() for item in self.entries],
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


@dataclass(slots=True)
class ObjectiveAmendmentLedger:
    """Thread-safe append-only facade for direct amendment APIs."""

    _history: ObjectiveAmendmentHistory = field(default_factory=ObjectiveAmendmentHistory)
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    @property
    def history(self) -> ObjectiveAmendmentHistory:
        with self._lock:
            return self._history

    def _append(self, amendment: ObjectiveAmendment) -> ObjectiveAmendment:
        with self._lock:
            self._history = self._history.append(amendment)
        return amendment

    def propose(self, amendment: ObjectiveAmendment) -> ObjectiveAmendment:
        return self._append(amendment)

    def inspect(self, amendment_id: str) -> ObjectiveAmendment:
        return self.history.inspect(amendment_id)

    def approve(self, amendment_id: str, **kwargs: str) -> ObjectiveAmendment:
        return self._append(approve_objective_amendment(self.inspect(amendment_id), **kwargs))

    def reject(self, amendment_id: str, **kwargs: str) -> ObjectiveAmendment:
        return self._append(reject_objective_amendment(self.inspect(amendment_id), **kwargs))

    def expire(self, amendment_id: str, **kwargs: str) -> ObjectiveAmendment:
        return self._append(expire_objective_amendment(self.inspect(amendment_id), **kwargs))

    def cancel(self, amendment_id: str, **kwargs: str) -> ObjectiveAmendment:
        return self._append(cancel_objective_amendment(self.inspect(amendment_id), **kwargs))

    def apply(self, amendment_id: str, **kwargs: object) -> ObjectiveAmendmentApplication:
        with self._lock:
            amendment = self._history.inspect(amendment_id)
            application = apply_objective_amendment(amendment, **kwargs)  # type: ignore[arg-type]
            if amendment.status is not ObjectiveAmendmentStatus.APPLIED:
                self._history = self._history.append(
                    replace(
                        amendment,
                        status=ObjectiveAmendmentStatus.APPLIED,
                        application=application,
                    )
                )
            return application


__all__ = [
    "OBJECTIVE_AMENDMENT_SCHEMA_VERSION",
    "AmendmentCancellation",
    "AmendmentStatus",
    "ObjectiveAmendment",
    "ObjectiveAmendmentApplication",
    "ObjectiveAmendmentDiff",
    "ObjectiveAmendmentError",
    "ObjectiveAmendmentHistory",
    "ObjectiveAmendmentLedger",
    "ObjectiveAmendmentStatus",
    "ObjectiveDiff",
    "apply_objective_amendment",
    "approve_objective_amendment",
    "cancel_objective_amendment",
    "diff_objective_contracts",
    "expire_objective_amendment",
    "propose_objective_amendment",
    "reject_objective_amendment",
    "replay_objective_amendment",
]

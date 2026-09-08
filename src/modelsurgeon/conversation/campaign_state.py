"""Durable canonical state for conversational optimization campaigns.

The text model and its transcript are deliberately absent from this module.
Only trusted, typed records can create or advance a campaign.  A campaign is
recovered from this store by identity and version; no chat replay is needed or
permitted to reconstruct authoritative state.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self, cast

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.optimization_package import ApprovalAuditRecord, ApprovalReuse
from modelsurgeon.policy import PolicyDecision, PolicyDecisionError

CAMPAIGN_STATE_SCHEMA_VERSION = 1
CAMPAIGN_STATE_DB_SCHEMA_VERSION = 1
CAMPAIGN_APPROVAL_SCHEMA_VERSION = 2

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TRANSITION_KIND = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_FORBIDDEN_KEYS = re.compile(
    r"^(?:transcript|messages|history|summary|provider_memory|secret|password|"
    r"credential|authorization|access[_-]?token|refresh[_-]?token|api[_-]?key)$",
    re.IGNORECASE,
)


class CampaignStateError(ValueError):
    """Raised when canonical conversational state cannot be trusted."""


class CampaignLifecycle(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CampaignOutcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ApprovalStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


def _canonical(value: object) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise CampaignStateError("canonical state contains a non-JSON value") from error


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise CampaignStateError(f"{label} must be a canonical identifier")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise CampaignStateError(f"{label} must be a sha256 content digest")
    return value


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CampaignStateError(f"{label} must be a JSON object")

    def validate_keys(item: object) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise CampaignStateError(f"{label} contains a non-text object key")
                if _FORBIDDEN_KEYS.fullmatch(key):
                    raise CampaignStateError(
                        f"{label} cannot contain ephemeral or secret field {key!r}"
                    )
                validate_keys(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                validate_keys(nested)

    validate_keys(value)
    try:
        decoded = json.loads(_canonical(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise CampaignStateError(f"{label} is not canonical JSON") from error
    if not isinstance(decoded, dict):
        raise CampaignStateError(f"{label} must be a JSON object")
    return cast(dict[str, object], decoded)


def _object_from_record(value: object, label: str) -> dict[str, object]:
    return _mapping(value, label)


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CampaignStateError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise CampaignStateError(f"{label} must be positive")
    return result


def _positive_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise CampaignStateError(f"{label} must be positive")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise CampaignStateError(f"{label} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    """The exact current OptimizationSpec and its preserved hard constraints."""

    spec_identity: str
    spec_digest: str
    spec: Mapping[str, object]
    hard_constraints: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        _identifier(self.spec_identity, "spec identity")
        _sha256(self.spec_digest, "spec digest")
        normalized_spec = _mapping(self.spec, "campaign spec")
        if _digest(normalized_spec) != self.spec_digest:
            raise CampaignStateError("campaign spec digest does not match its exact payload")
        normalized_constraints = tuple(
            sorted(
                (_mapping(item, "hard constraint") for item in self.hard_constraints),
                key=_canonical,
            )
        )
        object.__setattr__(self, "spec", normalized_spec)
        object.__setattr__(self, "hard_constraints", normalized_constraints)

    def to_record(self) -> dict[str, object]:
        return {
            "spec_identity": self.spec_identity,
            "spec_digest": self.spec_digest,
            "spec": dict(self.spec),
            "hard_constraints": [dict(item) for item in self.hard_constraints],
        }

    @classmethod
    def from_record(cls, value: object) -> CampaignSpec:
        record = _object_from_record(value, "campaign spec")
        if set(record) != {"spec_identity", "spec_digest", "spec", "hard_constraints"}:
            raise CampaignStateError("campaign spec has missing or unknown fields")
        constraints = record["hard_constraints"]
        if not isinstance(constraints, list):
            raise CampaignStateError("hard constraints must be an array")
        return cls(
            cast(str, record["spec_identity"]),
            cast(str, record["spec_digest"]),
            _mapping(record["spec"], "campaign spec payload"),
            tuple(_mapping(item, "hard constraint") for item in constraints),
        )


@dataclass(frozen=True, slots=True)
class CampaignApproval:
    """Approval state bound to one exact spec and plan scope."""

    status: ApprovalStatus
    spec_digest: str
    approval_id: str | None = None
    recorded_by: str | None = None
    expires_at: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)
    plan_id: str | None = None
    plan_digest: str | None = None
    diff_id: str | None = None
    scope: tuple[str, ...] = ()
    reuse: ApprovalReuse = ApprovalReuse.REUSABLE
    uses: int = 0
    audit_evidence: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, ApprovalStatus):
            raise CampaignStateError("approval status is invalid")
        if self.plan_id is None and self.plan_digest is not None:
            raise CampaignStateError("approval plan digest requires a plan ID")
        if self.plan_id is not None and self.plan_digest is None:
            raise CampaignStateError("approval plan ID requires a plan digest")
        _sha256(self.spec_digest, "approval spec digest")
        if self.approval_id is not None:
            _identifier(self.approval_id, "approval ID")
        if self.recorded_by is not None:
            _identifier(self.recorded_by, "approval recorder")
        if self.expires_at is not None:
            try:
                expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            except (TypeError, ValueError) as error:
                raise CampaignStateError("approval expiry must be an ISO-8601 timestamp") from error
            if expiry.tzinfo is None:
                raise CampaignStateError("approval expiry must include a timezone")
        if self.plan_id is not None:
            _identifier(self.plan_id, "approval plan ID")
        if self.plan_digest is not None:
            _sha256(self.plan_digest, "approval plan digest")
        if self.diff_id is not None:
            _identifier(self.diff_id, "approval diff ID")
        if self.scope != tuple(sorted(set(self.scope))) or any(
            not isinstance(item, str) or not item.strip() for item in self.scope
        ):
            raise CampaignStateError("approval scope must be sorted and unique")
        if not isinstance(self.reuse, ApprovalReuse):
            raise CampaignStateError("approval reuse policy is invalid")
        if isinstance(self.uses, bool) or not isinstance(self.uses, int) or self.uses < 0:
            raise CampaignStateError("approval uses must be a non-negative integer")
        if self.reuse is ApprovalReuse.ONE_TIME and self.uses > 1:
            raise CampaignStateError("one-time approval can only be used once")
        if self.status is ApprovalStatus.APPROVED and (
            self.approval_id is None or self.recorded_by is None or self.expires_at is None
        ):
            raise CampaignStateError("approved state requires approval identity and recorder")
        if self.status in {ApprovalStatus.REJECTED, ApprovalStatus.EXPIRED} and (
            self.approval_id is None or self.recorded_by is None or self.expires_at is None
        ):
            raise CampaignStateError(
                "terminal approval state requires approval identity and recorder"
            )
        object.__setattr__(
            self, "provenance", _mapping(self.provenance or {}, "approval provenance")
        )
        normalized_audit = tuple(
            _mapping(item, "approval audit evidence") for item in self.audit_evidence
        )
        for item in normalized_audit:
            try:
                ApprovalAuditRecord.from_record(item)
            except ValueError as error:
                raise CampaignStateError("approval audit evidence is malformed") from error
        object.__setattr__(self, "audit_evidence", normalized_audit)

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status.value,
            "spec_digest": self.spec_digest,
            "approval_id": self.approval_id,
            "recorded_by": self.recorded_by,
            "expires_at": self.expires_at,
            "provenance": dict(self.provenance),
        }
        if (
            self.plan_id is not None
            or self.plan_digest is not None
            or self.diff_id is not None
            or self.scope
            or self.reuse is not ApprovalReuse.REUSABLE
            or self.uses
            or self.audit_evidence
        ):
            record.update(
                {
                    "approval_schema_version": CAMPAIGN_APPROVAL_SCHEMA_VERSION,
                    "plan_id": self.plan_id,
                    "plan_digest": self.plan_digest,
                    "diff_id": self.diff_id,
                    "scope": list(self.scope),
                    "reuse": self.reuse.value,
                    "uses": self.uses,
                    "audit_evidence": [dict(item) for item in self.audit_evidence],
                }
            )
        return record

    @property
    def active(self) -> bool:
        """Whether this approval is currently usable for consequential work."""

        if self.status is not ApprovalStatus.APPROVED or self.expires_at is None:
            return False
        expiry = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
        return (
            expiry.astimezone(UTC) > datetime.now(UTC)
            and not (self.reuse is ApprovalReuse.ONE_TIME and self.uses >= 1)
        )

    @classmethod
    def pending(cls, spec_digest: str) -> CampaignApproval:
        return cls(ApprovalStatus.PENDING, spec_digest)

    @classmethod
    def from_record(cls, value: object) -> CampaignApproval:
        record = _object_from_record(value, "campaign approval")
        version = record.get("approval_schema_version", 1)
        if version not in {1, CAMPAIGN_APPROVAL_SCHEMA_VERSION}:
            raise CampaignStateError("unsupported campaign approval schema version")
        expected = {
            "approval_schema_version",
            "status",
            "spec_digest",
            "approval_id",
            "recorded_by",
            "expires_at",
            "provenance",
            "plan_id",
            "plan_digest",
            "diff_id",
            "scope",
            "reuse",
            "uses",
            "audit_evidence",
        }
        legacy = {"status", "spec_digest", "approval_id", "recorded_by", "expires_at", "provenance"}
        if set(record) not in (expected, legacy):
            raise CampaignStateError("campaign approval has missing or unknown fields")
        try:
            status = ApprovalStatus(cast(str, record["status"]))
        except ValueError as error:
            raise CampaignStateError("campaign approval has an unknown status") from error
        raw_scope = record.get("scope", [])
        raw_audit = record.get("audit_evidence", [])
        if not isinstance(raw_scope, list) or not isinstance(raw_audit, list):
            raise CampaignStateError("campaign approval scope or audit evidence is malformed")
        try:
            reuse = ApprovalReuse(str(record.get("reuse", ApprovalReuse.REUSABLE.value)))
        except ValueError as error:
            raise CampaignStateError("campaign approval reuse policy is invalid") from error
        return cls(
            status,
            cast(str, record["spec_digest"]),
            None if record["approval_id"] is None else cast(str, record["approval_id"]),
            None if record["recorded_by"] is None else cast(str, record["recorded_by"]),
            None if record["expires_at"] is None else cast(str, record["expires_at"]),
            _mapping(record["provenance"], "approval provenance"),
            None if record.get("plan_id") is None else cast(str, record["plan_id"]),
            None if record.get("plan_digest") is None else cast(str, record["plan_digest"]),
            None if record.get("diff_id") is None else cast(str, record["diff_id"]),
            tuple(str(item) for item in raw_scope),
            reuse,
            cast(int, record.get("uses", 0)),
            tuple(_mapping(item, "approval audit evidence") for item in raw_audit),
        )


@dataclass(frozen=True, slots=True)
class CampaignBudget:
    """Hard execution ceilings retained at the campaign boundary."""

    max_wall_seconds: float
    max_memory_bytes: int
    max_evaluation_count: int
    max_output_bytes: int

    def __post_init__(self) -> None:
        _positive_float(self.max_wall_seconds, "campaign wall budget")
        _positive_int(self.max_memory_bytes, "campaign memory budget")
        _positive_int(self.max_evaluation_count, "campaign evaluation budget")
        _positive_int(self.max_output_bytes, "campaign output budget")

    def to_record(self) -> dict[str, object]:
        return {
            "max_wall_seconds": self.max_wall_seconds,
            "max_memory_bytes": self.max_memory_bytes,
            "max_evaluation_count": self.max_evaluation_count,
            "max_output_bytes": self.max_output_bytes,
        }

    @classmethod
    def from_record(cls, value: object) -> CampaignBudget:
        record = _object_from_record(value, "campaign budget")
        expected = {
            "max_wall_seconds",
            "max_memory_bytes",
            "max_evaluation_count",
            "max_output_bytes",
        }
        if set(record) != expected:
            raise CampaignStateError("campaign budget has missing or unknown fields")
        return cls(
            cast(float, record["max_wall_seconds"]),
            cast(int, record["max_memory_bytes"]),
            cast(int, record["max_evaluation_count"]),
            cast(int, record["max_output_bytes"]),
        )


@dataclass(frozen=True, slots=True)
class EvidenceCursor:
    """Monotonic cursor into retained canonical campaign evidence."""

    sequence: int = 0
    evidence_id: str | None = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.sequence, "evidence cursor")
        if self.evidence_id is not None:
            _identifier(self.evidence_id, "evidence cursor ID")
            if self.sequence == 0:
                raise CampaignStateError("a non-empty evidence cursor must advance its sequence")

    def to_record(self) -> dict[str, object]:
        return {"sequence": self.sequence, "evidence_id": self.evidence_id}

    @classmethod
    def from_record(cls, value: object) -> EvidenceCursor:
        record = _object_from_record(value, "evidence cursor")
        if set(record) != {"sequence", "evidence_id"}:
            raise CampaignStateError("evidence cursor has missing or unknown fields")
        return cls(
            cast(int, record["sequence"]),
            None if record["evidence_id"] is None else cast(str, record["evidence_id"]),
        )


@dataclass(frozen=True, slots=True)
class CampaignEvidence:
    """One immutable canonical or negative evidence record."""

    evidence_id: str
    source_digest: str
    outcome: CampaignOutcome
    detail: str
    provenance: Mapping[str, object]
    artifact_digest: str | None = None
    inconclusive: bool = False

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "evidence ID")
        _sha256(self.source_digest, "evidence source digest")
        if not isinstance(self.outcome, CampaignOutcome):
            raise CampaignStateError("evidence outcome is invalid")
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise CampaignStateError("evidence detail is required")
        if self.artifact_digest is not None:
            _sha256(self.artifact_digest, "evidence artifact digest")
        if not isinstance(self.inconclusive, bool):
            raise CampaignStateError("evidence inconclusive flag is invalid")
        object.__setattr__(self, "provenance", _mapping(self.provenance, "evidence provenance"))

    @property
    def digest(self) -> str:
        return _digest(self.to_record(include_digest=False))

    def to_record(self, *, include_digest: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "record_type": "canonical_campaign_evidence",
            "schema_version": CAMPAIGN_STATE_SCHEMA_VERSION,
            "evidence_id": self.evidence_id,
            "source_digest": self.source_digest,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "provenance": dict(self.provenance),
            "artifact_digest": self.artifact_digest,
            "inconclusive": self.inconclusive,
        }
        if include_digest:
            record["evidence_digest"] = self.digest
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> CampaignEvidence:
        record = _object_from_record(value, "campaign evidence")
        expected = {
            "record_type",
            "schema_version",
            "evidence_id",
            "source_digest",
            "outcome",
            "detail",
            "provenance",
            "artifact_digest",
            "inconclusive",
            "evidence_digest",
        }
        if set(record) != expected:
            raise CampaignStateError("campaign evidence has missing or unknown fields")
        if record["record_type"] != "canonical_campaign_evidence":
            raise CampaignStateError("record is not canonical campaign evidence")
        if record["schema_version"] != CAMPAIGN_STATE_SCHEMA_VERSION:
            raise CampaignStateError("unsupported campaign evidence schema version")
        try:
            outcome = CampaignOutcome(cast(str, record["outcome"]))
        except ValueError as error:
            raise CampaignStateError("campaign evidence has an unknown outcome") from error
        result = cls(
            cast(str, record["evidence_id"]),
            cast(str, record["source_digest"]),
            outcome,
            cast(str, record["detail"]),
            _mapping(record["provenance"], "evidence provenance"),
            None if record["artifact_digest"] is None else cast(str, record["artifact_digest"]),
            cast(bool, record["inconclusive"]),
        )
        if record["evidence_digest"] != result.digest:
            raise CampaignStateError("campaign evidence digest does not match its payload")
        return result


@dataclass(frozen=True, slots=True)
class CampaignState:
    """Complete authoritative state for one reconnectable campaign."""

    campaign_id: str
    session_id: str
    run_id: str
    source_model_digest: str
    spec: CampaignSpec
    policy_state: Mapping[str, object]
    approval: CampaignApproval
    evidence_cursor: EvidenceCursor
    provider_context: Mapping[str, object]
    budget: CampaignBudget
    lifecycle: CampaignLifecycle = CampaignLifecycle.CREATED
    outcome: CampaignOutcome = CampaignOutcome.UNKNOWN
    state_version: int = 0
    last_transition_id: str | None = None
    provenance: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = CAMPAIGN_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "campaign ID")
        _identifier(self.session_id, "session ID")
        _identifier(self.run_id, "run ID")
        _sha256(self.source_model_digest, "source model digest")
        if not isinstance(self.lifecycle, CampaignLifecycle):
            raise CampaignStateError("campaign lifecycle is invalid")
        if not isinstance(self.outcome, CampaignOutcome):
            raise CampaignStateError("campaign outcome is invalid")
        _nonnegative_int(self.state_version, "campaign state version")
        if self.last_transition_id is not None:
            _identifier(self.last_transition_id, "last transition ID")
        if self.approval.spec_digest != self.spec.spec_digest:
            raise CampaignStateError("approval is bound to a different spec digest")
        object.__setattr__(
            self, "policy_state", _mapping(self.policy_state, "campaign policy state")
        )
        raw_policy = self.policy_state.get("policy_precedence")
        if raw_policy is not None:
            try:
                policy = PolicyDecision.from_record(raw_policy)
            except PolicyDecisionError as error:
                raise CampaignStateError("campaign policy precedence is malformed") from error
            if not policy.executable:
                raise CampaignStateError(
                    "campaign state cannot be created from a non-executable policy decision"
                )
        object.__setattr__(
            self, "provider_context", _mapping(self.provider_context, "provider context")
        )
        object.__setattr__(self, "provenance", _mapping(self.provenance or {}, "state provenance"))
        if self.schema_version != CAMPAIGN_STATE_SCHEMA_VERSION:
            raise CampaignStateError("unsupported campaign state schema version")

    @property
    def spec_identity(self) -> str:
        return self.spec.spec_identity

    @property
    def spec_digest(self) -> str:
        return self.spec.spec_digest

    @property
    def hard_constraints(self) -> tuple[Mapping[str, object], ...]:
        return self.spec.hard_constraints

    @property
    def policy_decision(self) -> PolicyDecision | None:
        """Return the persisted centralized policy decision, when present."""

        raw = self.policy_state.get("policy_precedence")
        if raw is None:
            return None
        try:
            return PolicyDecision.from_record(raw)
        except PolicyDecisionError as error:
            raise CampaignStateError("campaign policy precedence is malformed") from error

    def to_record(self, *, include_transition: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "record_type": "conversational_campaign_state",
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "source_model_digest": self.source_model_digest,
            "spec": self.spec.to_record(),
            "policy_state": dict(self.policy_state),
            "approval": self.approval.to_record(),
            "evidence_cursor": self.evidence_cursor.to_record(),
            "provider_context": dict(self.provider_context),
            "budget": self.budget.to_record(),
            "lifecycle": self.lifecycle.value,
            "outcome": self.outcome.value,
            "state_version": self.state_version,
            "last_transition_id": self.last_transition_id if include_transition else None,
            "provenance": dict(self.provenance),
        }
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record())

    @property
    def digest(self) -> str:
        return _digest(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> CampaignState:
        record = _object_from_record(value, "campaign state")
        expected = {
            "record_type",
            "schema_version",
            "campaign_id",
            "session_id",
            "run_id",
            "source_model_digest",
            "spec",
            "policy_state",
            "approval",
            "evidence_cursor",
            "provider_context",
            "budget",
            "lifecycle",
            "outcome",
            "state_version",
            "last_transition_id",
            "provenance",
        }
        if set(record) != expected:
            raise CampaignStateError("campaign state has missing or unknown fields")
        if record["record_type"] != "conversational_campaign_state":
            raise CampaignStateError("record is not canonical conversational campaign state")
        try:
            lifecycle = CampaignLifecycle(cast(str, record["lifecycle"]))
            outcome = CampaignOutcome(cast(str, record["outcome"]))
        except ValueError as error:
            raise CampaignStateError(
                "campaign state has an unknown lifecycle or outcome"
            ) from error
        return cls(
            cast(str, record["campaign_id"]),
            cast(str, record["session_id"]),
            cast(str, record["run_id"]),
            cast(str, record["source_model_digest"]),
            CampaignSpec.from_record(record["spec"]),
            _mapping(record["policy_state"], "campaign policy state"),
            CampaignApproval.from_record(record["approval"]),
            EvidenceCursor.from_record(record["evidence_cursor"]),
            _mapping(record["provider_context"], "provider context"),
            CampaignBudget.from_record(record["budget"]),
            lifecycle,
            outcome,
            cast(int, record["state_version"]),
            None
            if record["last_transition_id"] is None
            else cast(str, record["last_transition_id"]),
            _mapping(record["provenance"], "state provenance"),
            cast(int, record["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class CampaignStateTransition:
    """Append-only transition metadata retained for deterministic replay/audit."""

    transition_id: str
    campaign_id: str
    from_version: int
    to_version: int
    kind: str
    state_digest: str
    provenance_digest: str
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        _identifier(self.transition_id, "transition ID")
        _identifier(self.campaign_id, "transition campaign ID")
        if (
            self.from_version < -1
            or self.to_version < 0
            or self.to_version != self.from_version + 1
        ):
            raise CampaignStateError("transition versions are not contiguous")
        if _TRANSITION_KIND.fullmatch(self.kind) is None:
            raise CampaignStateError("transition kind is not canonical")
        _sha256(self.state_digest, "transition state digest")
        _sha256(self.provenance_digest, "transition provenance digest")
        object.__setattr__(self, "provenance", _mapping(self.provenance, "transition provenance"))

    def to_record(self) -> dict[str, object]:
        return {
            "transition_id": self.transition_id,
            "campaign_id": self.campaign_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "kind": self.kind,
            "state_digest": self.state_digest,
            "provenance_digest": self.provenance_digest,
            "provenance": dict(self.provenance),
        }


def derive_campaign_id(
    *,
    session_id: str,
    run_id: str,
    source_model_digest: str,
    spec_identity: str,
    spec_digest: str,
) -> str:
    """Derive the stable campaign identity from immutable linkage inputs."""

    payload = {
        "session_id": _identifier(session_id, "session ID"),
        "run_id": _identifier(run_id, "run ID"),
        "source_model_digest": _sha256(source_model_digest, "source model digest"),
        "spec_identity": _identifier(spec_identity, "spec identity"),
        "spec_digest": _sha256(spec_digest, "spec digest"),
    }
    return "campaign_" + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def new_campaign_state(
    *,
    session_id: str,
    run_id: str,
    source_model_digest: str,
    spec: CampaignSpec,
    policy_state: Mapping[str, object],
    provider_context: Mapping[str, object],
    budget: CampaignBudget,
    approval: CampaignApproval | None = None,
    campaign_id: str | None = None,
    provenance: Mapping[str, object] | None = None,
) -> CampaignState:
    """Build initial state without consulting or retaining conversational text."""

    selected_id = campaign_id or derive_campaign_id(
        session_id=session_id,
        run_id=run_id,
        source_model_digest=source_model_digest,
        spec_identity=spec.spec_identity,
        spec_digest=spec.spec_digest,
    )
    return CampaignState(
        selected_id,
        session_id,
        run_id,
        source_model_digest,
        spec,
        policy_state,
        approval or CampaignApproval.pending(spec.spec_digest),
        EvidenceCursor(),
        provider_context,
        budget,
        provenance={} if provenance is None else provenance,
    )


_MIGRATION_SQL = (
    """
    CREATE TABLE campaign_state_records (
        campaign_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        state_version INTEGER NOT NULL CHECK(state_version >= 0),
        state_json TEXT NOT NULL,
        state_digest TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE campaign_state_transitions (
        transition_id TEXT PRIMARY KEY,
        campaign_id TEXT NOT NULL REFERENCES campaign_state_records(campaign_id),
        from_version INTEGER NOT NULL CHECK(from_version >= -1),
        to_version INTEGER NOT NULL CHECK(to_version = from_version + 1),
        kind TEXT NOT NULL,
        state_json TEXT NOT NULL,
        state_digest TEXT NOT NULL,
        provenance_json TEXT NOT NULL,
        provenance_digest TEXT NOT NULL,
        UNIQUE(campaign_id, to_version)
    )
    """,
    """
    CREATE TABLE campaign_state_evidence (
        campaign_id TEXT NOT NULL REFERENCES campaign_state_records(campaign_id),
        evidence_id TEXT NOT NULL,
        sequence INTEGER NOT NULL CHECK(sequence > 0),
        evidence_json TEXT NOT NULL,
        evidence_digest TEXT NOT NULL,
        PRIMARY KEY(campaign_id, evidence_id),
        UNIQUE(campaign_id, sequence)
    )
    """,
    "CREATE INDEX campaign_state_records_session_idx ON campaign_state_records(session_id)",
)
_MIGRATION_CHECKSUM = hashlib.sha256(
    _canonical({"version": CAMPAIGN_STATE_DB_SCHEMA_VERSION, "statements": _MIGRATION_SQL}).encode(
        "utf-8"
    )
).hexdigest()


def _configure(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")


class CampaignStateStore:
    """WAL-backed store for canonical state, transitions, and retained evidence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().absolute().resolve(strict=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._connection = sqlite3.connect(self.path, timeout=5.0)
            _configure(self._connection)
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")
            self._migrate()
        except (OSError, sqlite3.Error, CampaignStateError) as error:
            if hasattr(self, "_connection"):
                self._connection.close()
            raise CampaignStateError(
                "campaign state database is missing, corrupt, or unsupported"
            ) from error
        self._lock = threading.RLock()
        self._closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()

    def _migrate(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS campaign_state_schema_migrations (
                version INTEGER PRIMARY KEY,
                checksum TEXT NOT NULL
            )
            """
        )
        rows = self._connection.execute(
            "SELECT version, checksum FROM campaign_state_schema_migrations ORDER BY version"
        ).fetchall()
        if tuple(int(row[0]) for row in rows) != tuple(range(1, len(rows) + 1)):
            raise CampaignStateError("campaign state migration versions contain a gap")
        if len(rows) > CAMPAIGN_STATE_DB_SCHEMA_VERSION:
            raise CampaignStateError("campaign state database is newer than this build")
        if rows and (int(rows[0][0]) != 1 or str(rows[0][1]) != _MIGRATION_CHECKSUM):
            raise CampaignStateError("campaign state migration checksum is invalid")
        if rows:
            self._connection.commit()
            return
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            for statement in _MIGRATION_SQL:
                self._connection.execute(statement)
            self._connection.execute(
                "INSERT INTO campaign_state_schema_migrations(version, checksum) VALUES (?, ?)",
                (CAMPAIGN_STATE_DB_SCHEMA_VERSION, _MIGRATION_CHECKSUM),
            )
            self._connection.commit()
        except sqlite3.Error as error:
            self._connection.rollback()
            raise CampaignStateError("campaign state database migration failed") from error

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._connection.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise CampaignStateError("campaign state store is closed")

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        self._require_open()
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        """Open a query-only connection that sees committed WAL snapshots."""

        self._require_open()
        try:
            connection = sqlite3.connect(self.path, timeout=5.0)
            _configure(connection)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
        except sqlite3.Error as error:
            raise CampaignStateError("campaign state reader could not be opened") from error
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _state_from_row(row: sqlite3.Row | tuple[object, ...]) -> CampaignState:
        try:
            raw = json.loads(str(row[3]))
            state = CampaignState.from_record(raw)
            if state.campaign_id != str(row[0]) or state.session_id != str(row[1]):
                raise CampaignStateError("stored campaign linkage is inconsistent")
            stored_version = cast(int, row[2])
            if state.state_version != stored_version or state.digest != str(row[4]):
                raise CampaignStateError("stored campaign state digest or version is inconsistent")
            return state
        except (CampaignStateError, TypeError, ValueError, json.JSONDecodeError) as error:
            if isinstance(error, CampaignStateError):
                raise
            raise CampaignStateError("stored campaign state is malformed") from error

    def load(self, campaign_id: str) -> CampaignState:
        _identifier(campaign_id, "campaign ID")
        with self.reader() as connection:
            row = connection.execute(
                "SELECT campaign_id, session_id, state_version, state_json, state_digest "
                "FROM campaign_state_records WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
        if row is None:
            raise CampaignStateError(f"unknown campaign {campaign_id}")
        return self._state_from_row(row)

    def load_for_session(self, session_id: str, *, campaign_id: str | None = None) -> CampaignState:
        _identifier(session_id, "session ID")
        if campaign_id is not None:
            state = self.load(campaign_id)
            if state.session_id != session_id:
                raise CampaignStateError("campaign does not belong to the requested session")
            return state
        with self.reader() as connection:
            rows = connection.execute(
                "SELECT campaign_id, session_id, state_version, state_json, state_digest "
                "FROM campaign_state_records WHERE session_id = ? ORDER BY campaign_id",
                (session_id,),
            ).fetchall()
        if len(rows) != 1:
            raise CampaignStateError("session does not identify exactly one campaign")
        return self._state_from_row(rows[0])

    def campaigns_for_session(self, session_id: str) -> tuple[CampaignState, ...]:
        """Return every immutable campaign version for a session in ID order."""

        _identifier(session_id, "session ID")
        with self.reader() as connection:
            rows = connection.execute(
                "SELECT campaign_id, session_id, state_version, state_json, state_digest "
                "FROM campaign_state_records WHERE session_id = ? ORDER BY campaign_id",
                (session_id,),
            ).fetchall()
        return tuple(self._state_from_row(row) for row in rows)

    def children(self, campaign_id: str) -> tuple[CampaignState, ...]:
        """Return immutable child campaigns whose provenance names this parent."""

        _identifier(campaign_id, "campaign ID")
        children = tuple(
            state
            for state in self.campaigns_for_session(self.load(campaign_id).session_id)
            if state.provenance.get("parent_campaign_id") == campaign_id
        )
        return tuple(sorted(children, key=lambda item: item.campaign_id))

    def lineage(self, campaign_id: str) -> tuple[CampaignState, ...]:
        """Return a child-to-root immutable campaign lineage."""

        current = self.load(campaign_id)
        result = [current]
        seen = {current.campaign_id}
        while True:
            parent_id = current.provenance.get("parent_campaign_id")
            if not isinstance(parent_id, str):
                break
            if parent_id in seen:
                raise CampaignStateError("campaign lineage contains a cycle")
            current = self.load(parent_id)
            result.append(current)
            seen.add(current.campaign_id)
        return tuple(result)

    def reconnect(self, campaign_id: str, session_id: str) -> CampaignState:
        """Recover by trusted IDs; transcript text is intentionally not an input."""

        return self.load_for_session(session_id, campaign_id=campaign_id)

    @staticmethod
    def _allowed_lifecycle(current: CampaignLifecycle, requested: CampaignLifecycle) -> bool:
        if current is requested:
            return True
        allowed: dict[CampaignLifecycle, frozenset[CampaignLifecycle]] = {
            CampaignLifecycle.CREATED: frozenset(
                {CampaignLifecycle.RUNNING, CampaignLifecycle.PAUSED, CampaignLifecycle.CANCELLED}
            ),
            CampaignLifecycle.RUNNING: frozenset(
                {
                    CampaignLifecycle.PAUSED,
                    CampaignLifecycle.COMPLETED,
                    CampaignLifecycle.FAILED,
                    CampaignLifecycle.CANCELLED,
                }
            ),
            CampaignLifecycle.PAUSED: frozenset(
                {CampaignLifecycle.RUNNING, CampaignLifecycle.CANCELLED}
            ),
            CampaignLifecycle.COMPLETED: frozenset(),
            CampaignLifecycle.FAILED: frozenset(),
            CampaignLifecycle.CANCELLED: frozenset(),
        }
        return requested in allowed[current]

    def _command(
        self,
        campaign_id: str,
        session_id: str,
        *,
        kind: str,
        lifecycle: CampaignLifecycle,
        operation_id: str,
        detail: str,
    ) -> CampaignState:
        _identifier(session_id, "session ID")
        _identifier(operation_id, "operation ID")
        current = self.reconnect(campaign_id, session_id)
        if current.lifecycle is lifecycle and lifecycle is not CampaignLifecycle.RUNNING:
            return current
        if current.lifecycle is not lifecycle and not self._allowed_lifecycle(
            current.lifecycle, lifecycle
        ):
            raise CampaignStateError(
                f"cannot {kind} campaign in {current.lifecycle.value} lifecycle"
            )
        if lifecycle is CampaignLifecycle.RUNNING and not current.approval.active:
            if current.approval.status is ApprovalStatus.APPROVED:
                expired = CampaignApproval(
                    ApprovalStatus.EXPIRED,
                    current.approval.spec_digest,
                    current.approval.approval_id,
                    current.approval.recorded_by,
                    current.approval.expires_at,
                    {
                        "record_type": "approval_expiry",
                        "source": "campaign_state_store",
                    },
                )
                current = self.transition(
                    campaign_id,
                    expected_version=current.state_version,
                    kind="approval_expired",
                    provenance={
                        "record_type": "campaign_transition",
                        "source": "campaign_state_store",
                        "operation_id": operation_id,
                        "detail": "approval expired before recovery",
                    },
                    lifecycle=CampaignLifecycle.PAUSED,
                    approval=expired,
                )
            raise CampaignStateError("campaign approval is missing or expired")
        if current.lifecycle is lifecycle:
            return current
        return self.transition(
            campaign_id,
            expected_version=current.state_version,
            kind=kind,
            provenance={
                "record_type": "campaign_transition",
                "source": "campaign_state_store",
                "operation_id": operation_id,
                "detail": detail,
            },
            lifecycle=lifecycle,
        )

    def pause(
        self,
        campaign_id: str,
        session_id: str,
        *,
        operation_id: str = "pause",
        detail: str = "paused by operator",
    ) -> CampaignState:
        """Durably request a pause without discarding completed work."""

        return self._command(
            campaign_id,
            session_id,
            kind="paused",
            lifecycle=CampaignLifecycle.PAUSED,
            operation_id=operation_id,
            detail=detail,
        )

    def resume(
        self,
        campaign_id: str,
        session_id: str,
        *,
        operation_id: str = "resume",
        detail: str = "resumed from the last committed stage",
    ) -> CampaignState:
        """Resume only a non-terminal, approval-valid campaign."""

        return self._command(
            campaign_id,
            session_id,
            kind="resumed",
            lifecycle=CampaignLifecycle.RUNNING,
            operation_id=operation_id,
            detail=detail,
        )

    def cancel(
        self,
        campaign_id: str,
        session_id: str,
        *,
        operation_id: str = "cancel",
        detail: str = "cancelled by operator",
    ) -> CampaignState:
        """Durably cancel a campaign; cancellation is terminal and replayable."""

        return self._command(
            campaign_id,
            session_id,
            kind="cancelled",
            lifecycle=CampaignLifecycle.CANCELLED,
            operation_id=operation_id,
            detail=detail,
        )

    def restart(
        self,
        campaign_id: str,
        session_id: str,
        *,
        operation_id: str = "restart",
    ) -> CampaignState:
        """Reconnect after process loss and mark unfinished work runnable."""

        return self.resume(
            campaign_id,
            session_id,
            operation_id=operation_id,
            detail="restarted from canonical campaign state",
        )

    @staticmethod
    def _transition_id(
        campaign_id: str,
        from_version: int,
        to_version: int,
        kind: str,
        state: CampaignState,
        provenance: Mapping[str, object],
    ) -> str:
        payload = {
            "campaign_id": campaign_id,
            "from_version": from_version,
            "to_version": to_version,
            "kind": kind,
            "state": state.to_record(include_transition=False),
            "provenance": _mapping(provenance, "transition provenance"),
        }
        return (
            "campaign_transition_" + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        )

    @staticmethod
    def _transition(
        current: CampaignState,
        *,
        kind: str,
        provenance: Mapping[str, object],
        lifecycle: CampaignLifecycle | None = None,
        outcome: CampaignOutcome | None = None,
        spec: CampaignSpec | None = None,
        policy_state: Mapping[str, object] | None = None,
        approval: CampaignApproval | None = None,
        evidence_cursor: EvidenceCursor | None = None,
        provider_context: Mapping[str, object] | None = None,
        budget: CampaignBudget | None = None,
    ) -> tuple[CampaignState, CampaignStateTransition]:
        if _TRANSITION_KIND.fullmatch(kind) is None:
            raise CampaignStateError("transition kind is not canonical")
        selected_spec = current.spec if spec is None else spec
        selected_approval = current.approval if approval is None else approval
        if selected_spec.spec_digest != current.spec.spec_digest and approval is None:
            selected_approval = CampaignApproval.pending(selected_spec.spec_digest)
        if selected_approval.spec_digest != selected_spec.spec_digest:
            raise CampaignStateError("approval must bind the transition's current spec")
        selected_lifecycle = current.lifecycle if lifecycle is None else lifecycle
        if not CampaignStateStore._allowed_lifecycle(current.lifecycle, selected_lifecycle):
            raise CampaignStateError(
                "invalid lifecycle transition "
                f"{current.lifecycle.value}->{selected_lifecycle.value}"
            )
        selected_provenance = _mapping(provenance, "transition provenance")
        candidate = replace(
            current,
            spec=selected_spec,
            policy_state=current.policy_state if policy_state is None else policy_state,
            approval=selected_approval,
            evidence_cursor=current.evidence_cursor if evidence_cursor is None else evidence_cursor,
            provider_context=current.provider_context
            if provider_context is None
            else provider_context,
            budget=current.budget if budget is None else budget,
            lifecycle=selected_lifecycle,
            outcome=current.outcome if outcome is None else outcome,
            state_version=current.state_version + 1,
            last_transition_id=None,
            provenance=selected_provenance,
        )
        transition_id = CampaignStateStore._transition_id(
            current.campaign_id,
            current.state_version,
            candidate.state_version,
            kind,
            candidate,
            selected_provenance,
        )
        next_state = replace(candidate, last_transition_id=transition_id)
        transition = CampaignStateTransition(
            transition_id,
            current.campaign_id,
            current.state_version,
            next_state.state_version,
            kind,
            next_state.digest,
            _digest(selected_provenance),
            selected_provenance,
        )
        return next_state, transition

    @staticmethod
    def _write_transition(
        connection: sqlite3.Connection,
        state: CampaignState,
        transition: CampaignStateTransition,
    ) -> None:
        state_json = state.canonical_json()
        connection.execute(
            "INSERT INTO campaign_state_transitions("
            "transition_id, campaign_id, from_version, to_version, kind, state_json, "
            "state_digest, provenance_json, provenance_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                transition.transition_id,
                state.campaign_id,
                transition.from_version,
                transition.to_version,
                transition.kind,
                state_json,
                state.digest,
                _canonical(transition.provenance),
                transition.provenance_digest,
            ),
        )
        connection.execute(
            "UPDATE campaign_state_records SET session_id = ?, state_version = ?, state_json = ?, "
            "state_digest = ? WHERE campaign_id = ?",
            (state.session_id, state.state_version, state_json, state.digest, state.campaign_id),
        )

    def create(self, state: CampaignState) -> CampaignState:
        """Persist one initial state and its provenance-linked creation record."""

        if state.state_version != 0 or state.last_transition_id is not None:
            raise CampaignStateError("new campaign state must start at version zero")
        initial = replace(state, state_version=0, last_transition_id=None)
        transition_id = self._transition_id(
            state.campaign_id, -1, 0, "created", initial, state.provenance
        )
        initial = replace(initial, last_transition_id=transition_id)
        transition = CampaignStateTransition(
            transition_id,
            state.campaign_id,
            -1,
            0,
            "created",
            initial.digest,
            _digest(state.provenance),
            _mapping(state.provenance, "state provenance"),
        )
        with self._write() as connection:
            try:
                connection.execute(
                    "INSERT INTO campaign_state_records(campaign_id, session_id, state_version, "
                    "state_json, state_digest) VALUES (?, ?, ?, ?, ?)",
                    (
                        initial.campaign_id,
                        initial.session_id,
                        initial.state_version,
                        initial.canonical_json(),
                        initial.digest,
                    ),
                )
                self._write_transition(connection, initial, transition)
            except sqlite3.IntegrityError as error:
                raise CampaignStateError(
                    "campaign already exists or has conflicting identity"
                ) from error
        return initial

    def transition(
        self,
        campaign_id: str,
        *,
        expected_version: int,
        kind: str,
        provenance: Mapping[str, object],
        lifecycle: CampaignLifecycle | None = None,
        outcome: CampaignOutcome | None = None,
        spec: CampaignSpec | None = None,
        policy_state: Mapping[str, object] | None = None,
        approval: CampaignApproval | None = None,
        provider_context: Mapping[str, object] | None = None,
        budget: CampaignBudget | None = None,
    ) -> CampaignState:
        """Apply one version-checked trusted transition atomically."""

        _identifier(campaign_id, "campaign ID")
        _nonnegative_int(expected_version, "expected campaign state version")
        with self._write() as connection:
            row = connection.execute(
                "SELECT campaign_id, session_id, state_version, state_json, state_digest "
                "FROM campaign_state_records WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise CampaignStateError(f"unknown campaign {campaign_id}")
            current = self._state_from_row(row)
            if current.state_version != expected_version:
                requested_provenance = _mapping(provenance, "transition provenance")
                operation_id = requested_provenance.get("operation_id")
                if isinstance(operation_id, str):
                    prior = connection.execute(
                        "SELECT kind, provenance_json FROM campaign_state_transitions "
                        "WHERE campaign_id = ? ORDER BY to_version",
                        (campaign_id,),
                    ).fetchall()
                    requested_json = _canonical(requested_provenance)
                    if any(
                        str(item[0]) == kind and str(item[1]) == requested_json for item in prior
                    ):
                        return current
                raise CampaignStateError(
                    "stale campaign state version "
                    f"{expected_version}; current is {current.state_version}"
                )
            next_state, transition = self._transition(
                current,
                kind=kind,
                provenance=provenance,
                lifecycle=lifecycle,
                outcome=outcome,
                spec=spec,
                policy_state=policy_state,
                approval=approval,
                provider_context=provider_context,
                budget=budget,
            )
            try:
                self._write_transition(connection, next_state, transition)
            except sqlite3.IntegrityError as error:
                raise CampaignStateError(
                    "campaign transition conflicts with persisted state"
                ) from error
        return next_state

    def append_evidence(
        self,
        campaign_id: str,
        evidence: CampaignEvidence,
        *,
        expected_version: int,
    ) -> CampaignState:
        """Retain supported, unsupported, failed, unknown, and inconclusive evidence."""

        _identifier(campaign_id, "campaign ID")
        _nonnegative_int(expected_version, "expected campaign state version")
        with self._write() as connection:
            row = connection.execute(
                "SELECT campaign_id, session_id, state_version, state_json, state_digest "
                "FROM campaign_state_records WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                raise CampaignStateError(f"unknown campaign {campaign_id}")
            current = self._state_from_row(row)
            existing = connection.execute(
                "SELECT evidence_json, evidence_digest FROM campaign_state_evidence "
                "WHERE campaign_id = ? AND evidence_id = ?",
                (campaign_id, evidence.evidence_id),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing[1]) != evidence.digest
                    or str(existing[0]) != evidence.canonical_json()
                ):
                    raise CampaignStateError(
                        "evidence ID conflicts with immutable retained evidence"
                    )
                return current
            if current.state_version != expected_version:
                raise CampaignStateError(
                    "stale campaign state version "
                    f"{expected_version}; current is {current.state_version}"
                )
            sequence = current.evidence_cursor.sequence + 1
            next_state, transition = self._transition(
                current,
                kind="evidence_appended",
                provenance={
                    "record_type": "campaign_evidence_append",
                    "evidence_id": evidence.evidence_id,
                    "evidence_digest": evidence.digest,
                },
                evidence_cursor=EvidenceCursor(sequence, evidence.evidence_id),
            )
            try:
                connection.execute(
                    "INSERT INTO campaign_state_evidence(campaign_id, evidence_id, sequence, "
                    "evidence_json, evidence_digest) VALUES (?, ?, ?, ?, ?)",
                    (
                        campaign_id,
                        evidence.evidence_id,
                        sequence,
                        evidence.canonical_json(),
                        evidence.digest,
                    ),
                )
                self._write_transition(connection, next_state, transition)
            except sqlite3.IntegrityError as error:
                raise CampaignStateError(
                    "evidence or campaign transition conflicts with persisted state"
                ) from error
        return next_state

    def history(self, campaign_id: str) -> tuple[CampaignStateTransition, ...]:
        _identifier(campaign_id, "campaign ID")
        with self.reader() as connection:
            rows = connection.execute(
                "SELECT transition_id, campaign_id, from_version, to_version, kind, state_digest, "
                "provenance_digest, provenance_json FROM campaign_state_transitions "
                "WHERE campaign_id = ? ORDER BY to_version",
                (campaign_id,),
            ).fetchall()
        try:
            return tuple(
                CampaignStateTransition(
                    str(row[0]),
                    str(row[1]),
                    int(row[2]),
                    int(row[3]),
                    str(row[4]),
                    str(row[5]),
                    str(row[6]),
                    _mapping(json.loads(str(row[7])), "transition provenance"),
                )
                for row in rows
            )
        except (CampaignStateError, TypeError, ValueError, json.JSONDecodeError) as error:
            if isinstance(error, CampaignStateError):
                raise
            raise CampaignStateError("stored campaign transition is malformed") from error

    def evidence(self, campaign_id: str) -> tuple[CampaignEvidence, ...]:
        _identifier(campaign_id, "campaign ID")
        with self.reader() as connection:
            rows = connection.execute(
                "SELECT evidence_json FROM campaign_state_evidence WHERE campaign_id = ? "
                "ORDER BY sequence",
                (campaign_id,),
            ).fetchall()
        try:
            return tuple(CampaignEvidence.from_record(json.loads(str(row[0]))) for row in rows)
        except (CampaignStateError, TypeError, ValueError, json.JSONDecodeError) as error:
            if isinstance(error, CampaignStateError):
                raise
            raise CampaignStateError("stored campaign evidence is malformed") from error


__all__ = [
    "CAMPAIGN_STATE_DB_SCHEMA_VERSION",
    "CAMPAIGN_STATE_SCHEMA_VERSION",
    "ApprovalStatus",
    "CampaignApproval",
    "CampaignBudget",
    "CampaignEvidence",
    "CampaignLifecycle",
    "CampaignOutcome",
    "CampaignSpec",
    "CampaignState",
    "CampaignStateError",
    "CampaignStateStore",
    "CampaignStateTransition",
    "EvidenceCursor",
    "derive_campaign_id",
    "new_campaign_state",
]

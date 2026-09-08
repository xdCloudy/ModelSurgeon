"""Centralized policy precedence for conversational control-plane decisions.

The conversational surface can propose interpretations, tool calls, and
explanations, but it cannot create a second execution authority.  This module
is deliberately independent of the compiler, dispatcher, campaign, and
explanation packages so all four boundaries use the same ordered decision
record.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

POLICY_DECISION_SCHEMA_VERSION: Final[int] = 1


class PolicyDecisionError(ValueError):
    """Raised when a policy candidate or decision record is malformed."""


class PolicySource(StrEnum):
    """Sources that may appear at the conversational policy boundary.

    The first five sources are trusted engine inputs.  User objectives,
    prompts, and provider text are retained as alternatives only; they never
    acquire execution authority by being present in a decision.
    """

    HARD_CONSTRAINTS = "hard_constraints"
    VALIDATED_SPEC = "validated_spec"
    APPROVAL_POLICY = "approval_policy"
    TOOL_CAPABILITY = "tool_capability"
    EVIDENCE_STATUS = "evidence_status"
    USER_OBJECTIVE = "user_objective"
    PROMPT = "prompt"
    PROVIDER = "provider"

    # Singular/descriptive aliases keep call sites readable without creating
    # additional serialized values.
    HARD_CONSTRAINT = "hard_constraints"
    TOOL = "tool_capability"
    PROVIDER_TEXT = "provider"
    USER_PROMPT = "prompt"


class PolicyOutcome(StrEnum):
    """Disposition of one policy source or of the resolved policy."""

    ALLOW = "allow"
    DENY = "deny"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    CONTRADICTORY = "contradictory"


_TRUSTED_SOURCES: Final[frozenset[PolicySource]] = frozenset(
    {
        PolicySource.HARD_CONSTRAINTS,
        PolicySource.VALIDATED_SPEC,
        PolicySource.APPROVAL_POLICY,
        PolicySource.TOOL_CAPABILITY,
        PolicySource.EVIDENCE_STATUS,
    }
)
_PRECEDENCE: Final[tuple[PolicySource, ...]] = (
    PolicySource.HARD_CONSTRAINTS,
    PolicySource.VALIDATED_SPEC,
    PolicySource.APPROVAL_POLICY,
    PolicySource.TOOL_CAPABILITY,
    PolicySource.EVIDENCE_STATUS,
    PolicySource.USER_OBJECTIVE,
    PolicySource.PROMPT,
    PolicySource.PROVIDER,
)
_RANK: Final[dict[PolicySource, int]] = {
    source: index for index, source in enumerate(_PRECEDENCE)
}


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PolicyDecisionError("policy record must be canonical JSON") from error


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyDecisionError(f"{label} must be non-empty text")
    return value


def _sorted_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if result != tuple(sorted(set(result))) or any(not value.strip() for value in result):
        raise PolicyDecisionError(f"{label} must be sorted and unique")
    return result


@dataclass(frozen=True, slots=True)
class PolicyCandidate:
    """One engine or untrusted proposal considered by the resolver."""

    source: PolicySource
    outcome: PolicyOutcome
    detail: str
    candidate_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source, PolicySource):
            raise PolicyDecisionError("policy candidate source is invalid")
        if not isinstance(self.outcome, PolicyOutcome):
            raise PolicyDecisionError("policy candidate outcome is invalid")
        _text(self.detail, "policy candidate detail")
        candidate_id = self.candidate_id or f"{self.source.value}:{self.outcome.value}"
        _text(candidate_id, "policy candidate ID")
        object.__setattr__(self, "candidate_id", candidate_id)

    @property
    def trusted(self) -> bool:
        return self.source in _TRUSTED_SOURCES

    def to_record(self) -> dict[str, str | bool]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source.value,
            "outcome": self.outcome.value,
            "detail": self.detail,
            "trusted": self.trusted,
        }

    @classmethod
    def from_record(cls, value: object) -> PolicyCandidate:
        if not isinstance(value, dict):
            raise PolicyDecisionError("policy candidate must be an object")
        expected = {"candidate_id", "source", "outcome", "detail", "trusted"}
        if set(value) != expected:
            raise PolicyDecisionError("policy candidate has missing or unknown fields")
        try:
            source = PolicySource(_text(value["source"], "policy candidate source"))
            outcome = PolicyOutcome(_text(value["outcome"], "policy candidate outcome"))
        except ValueError as error:
            raise PolicyDecisionError("policy candidate has an unknown enum value") from error
        candidate = cls(
            source,
            outcome,
            _text(value["detail"], "policy candidate detail"),
            _text(value["candidate_id"], "policy candidate ID"),
        )
        if value["trusted"] is not candidate.trusted:
            raise PolicyDecisionError("policy candidate trust marker is invalid")
        return candidate


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Deterministic precedence result shared by all control-plane layers."""

    operation: str
    outcome: PolicyOutcome
    winning_source: PolicySource | None
    winning_detail: str
    considered: tuple[PolicyCandidate, ...]
    rejected_alternatives: tuple[PolicyCandidate, ...]
    diagnostics: tuple[str, ...] = ()
    schema_version: int = POLICY_DECISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.operation, "policy operation")
        if not isinstance(self.outcome, PolicyOutcome):
            raise PolicyDecisionError("policy decision outcome is invalid")
        if self.winning_source is not None and not isinstance(self.winning_source, PolicySource):
            raise PolicyDecisionError("policy winning source is invalid")
        _text(self.winning_detail, "policy winning detail")
        if self.schema_version != POLICY_DECISION_SCHEMA_VERSION:
            raise PolicyDecisionError("unsupported policy decision schema version")
        if tuple(sorted(self.considered, key=_candidate_key)) != self.considered:
            raise PolicyDecisionError("policy candidates must be in precedence order")
        if tuple(sorted(self.rejected_alternatives, key=_candidate_key)) != (
            self.rejected_alternatives
        ):
            raise PolicyDecisionError("rejected policy alternatives must be canonical")
        _sorted_unique(self.diagnostics, "policy diagnostics")
        winning = self.winning_candidate
        if self.outcome is PolicyOutcome.ALLOW and winning is None:
            raise PolicyDecisionError("an allowed policy decision requires a winning source")
        if winning is not None and winning.source is not self.winning_source:
            raise PolicyDecisionError("winning policy source does not match its candidate")
        if winning is not None and winning.outcome is not self.outcome:
            raise PolicyDecisionError("winning candidate does not match the decision outcome")

    @property
    def executable(self) -> bool:
        """Whether this decision may authorize the bounded operation."""

        return self.outcome is PolicyOutcome.ALLOW

    @property
    def winning_candidate(self) -> PolicyCandidate | None:
        if self.winning_source is None:
            return None
        return next(
            (
                item
                for item in self.considered
                if item.source is self.winning_source and item.outcome is self.outcome
            ),
            None,
        )

    @property
    def decision_id(self) -> str:
        return "policy_decision_" + hashlib.sha256(
            self.canonical_json().encode("utf-8")
        ).hexdigest()

    def to_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "record_type": "central_policy_decision",
            "schema_version": self.schema_version,
            "operation": self.operation,
            "outcome": self.outcome.value,
            "winning_source": None
            if self.winning_source is None
            else self.winning_source.value,
            "winning_detail": self.winning_detail,
            "considered": [item.to_record() for item in self.considered],
            "rejected_alternatives": [
                item.to_record() for item in self.rejected_alternatives
            ],
            "diagnostics": list(self.diagnostics),
        }
        if include_identity:
            record["decision_id"] = self.decision_id
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record(include_identity=False))

    @classmethod
    def from_record(cls, value: object) -> PolicyDecision:
        if not isinstance(value, dict):
            raise PolicyDecisionError("policy decision must be an object")
        expected = {
            "record_type",
            "schema_version",
            "operation",
            "outcome",
            "winning_source",
            "winning_detail",
            "considered",
            "rejected_alternatives",
            "diagnostics",
            "decision_id",
        }
        if set(value) != expected or value["record_type"] != "central_policy_decision":
            raise PolicyDecisionError("policy decision has missing or unknown fields")
        if value["schema_version"] != POLICY_DECISION_SCHEMA_VERSION:
            raise PolicyDecisionError("unsupported policy decision schema version")
        try:
            outcome = PolicyOutcome(_text(value["outcome"], "policy outcome"))
            winning_source = (
                None
                if value["winning_source"] is None
                else PolicySource(_text(value["winning_source"], "winning policy source"))
            )
        except ValueError as error:
            raise PolicyDecisionError("policy decision has an unknown enum value") from error
        considered_raw = value["considered"]
        rejected_raw = value["rejected_alternatives"]
        diagnostics_raw = value["diagnostics"]
        if not isinstance(considered_raw, list) or not isinstance(rejected_raw, list):
            raise PolicyDecisionError("policy decision candidates must be arrays")
        if not isinstance(diagnostics_raw, list) or any(
            not isinstance(item, str) for item in diagnostics_raw
        ):
            raise PolicyDecisionError("policy diagnostics must be a string array")
        result = cls(
            _text(value["operation"], "policy operation"),
            outcome,
            winning_source,
            _text(value["winning_detail"], "policy winning detail"),
            tuple(PolicyCandidate.from_record(item) for item in considered_raw),
            tuple(PolicyCandidate.from_record(item) for item in rejected_raw),
            tuple(cast(list[str], diagnostics_raw)),
        )
        if _text(value["decision_id"], "policy decision ID") != result.decision_id:
            raise PolicyDecisionError("policy decision ID does not match its contents")
        return result


def _candidate_key(item: PolicyCandidate) -> tuple[int, str, str, str]:
    return (_RANK[item.source], item.source.value, item.outcome.value, item.candidate_id)


def policy_precedence() -> tuple[PolicySource, ...]:
    """Return the immutable source precedence from strongest to weakest."""

    return _PRECEDENCE


def resolve_policy(
    operation: str,
    candidates: Sequence[PolicyCandidate],
) -> PolicyDecision:
    """Resolve candidates with monotonic, fail-closed precedence.

    Trusted ``deny``, ``unsupported``, ``unknown``, and ``contradictory``
    states are never relaxed by a lower source.  Same-source disagreement is
    contradictory.  Prompt/provider candidates are retained for audit but are
    never eligible to authorize an operation.
    """

    _text(operation, "policy operation")
    if not candidates:
        return PolicyDecision(
            operation,
            PolicyOutcome.UNKNOWN,
            None,
            "no trusted policy source was supplied",
            (),
            (),
            ("no-trusted-policy-source",),
        )
    normalized = tuple(candidates)
    if any(not isinstance(item, PolicyCandidate) for item in normalized):
        raise PolicyDecisionError("policy candidates must be typed PolicyCandidate values")
    by_source: dict[PolicySource, list[PolicyCandidate]] = {}
    for item in normalized:
        by_source.setdefault(item.source, []).append(item)
    considered: list[PolicyCandidate] = []
    rejected: list[PolicyCandidate] = []
    diagnostics: set[str] = set()
    winner: PolicyCandidate | None = None
    allow_winner: PolicyCandidate | None = None

    for source in _PRECEDENCE:
        source_items = sorted(by_source.get(source, ()), key=_candidate_key)
        if not source_items:
            continue
        considered.extend(source_items)
        outcomes = {item.outcome for item in source_items}
        trusted = source in _TRUSTED_SOURCES
        if not trusted:
            rejected.extend(source_items)
            diagnostics.add(f"untrusted-source:{source.value}")
            continue
        if len(outcomes) > 1:
            if winner is None:
                winner = PolicyCandidate(
                    source,
                    PolicyOutcome.CONTRADICTORY,
                    "same policy source supplied conflicting states",
                    f"{source.value}:contradictory",
                )
                considered.append(winner)
                diagnostics.add(f"contradictory-source:{source.value}")
            continue
        item = source_items[0]
        if item.outcome is not PolicyOutcome.ALLOW:
            if winner is None:
                winner = item
        elif allow_winner is None:
            allow_winner = item

    if winner is None:
        winner = allow_winner

    if winner is None:
        diagnostics.add("no-trusted-policy-source")
        outcome = PolicyOutcome.UNKNOWN
        detail = "untrusted prompt/provider alternatives cannot authorize execution"
    else:
        outcome = winner.outcome
        detail = winner.detail

    winner_source = None if winner is None else winner.source
    winner_id = None if winner is None else winner.candidate_id
    rejected.extend(
        item
        for item in considered
        if item.candidate_id != winner_id
        and item not in rejected
        and (winner is None or _RANK[item.source] >= _RANK[winner.source])
    )
    # A lower trusted allow is a rejected alternative when a stronger veto won;
    # a weaker veto is also retained, while duplicate winning entries are not.
    rejected = sorted(
        {
            item.candidate_id: item
            for item in rejected
            if item.candidate_id != winner_id
        }.values(),
        key=_candidate_key,
    )
    ordered = tuple(sorted(considered, key=_candidate_key))
    return PolicyDecision(
        operation,
        outcome,
        winner_source,
        detail,
        ordered,
        tuple(rejected),
        tuple(sorted(diagnostics)),
    )


def policy_candidate(
    source: PolicySource,
    outcome: PolicyOutcome,
    detail: str,
    *,
    candidate_id: str = "",
) -> PolicyCandidate:
    """Small constructor used by boundary adapters and tests."""

    return PolicyCandidate(source, outcome, detail, candidate_id)


# Compatibility aliases make the record's role explicit to callers that use
# the phrase "policy record" rather than "policy decision".
PolicyDecisionRecord = PolicyDecision
PolicyPrecedence = PolicySource
evaluate_policy = resolve_policy


__all__ = [
    "POLICY_DECISION_SCHEMA_VERSION",
    "PolicyCandidate",
    "PolicyDecision",
    "PolicyDecisionError",
    "PolicyDecisionRecord",
    "PolicyOutcome",
    "PolicyPrecedence",
    "PolicySource",
    "evaluate_policy",
    "policy_candidate",
    "policy_precedence",
    "resolve_policy",
]

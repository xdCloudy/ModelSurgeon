"""Bounded, non-authoritative conversation summaries.

The summary is a transport view, not a second campaign store.  Canonical
campaign state and evidence are copied into a separately labelled section and
validated again on rehydration.  Transcript entries remain untrusted and may
be omitted only with an explicit, deterministic loss marker.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

from modelsurgeon.experiments.identity import canonical_identity_json

from .campaign_state import (
    CampaignEvidence,
    CampaignState,
    CampaignStateError,
    CampaignStateStore,
)

CONVERSATION_SUMMARY_SCHEMA_VERSION: Literal[1] = 1

_SUPPORTED_ROLES = frozenset({"assistant", "system", "tool", "user"})


class ConversationSummaryError(ValueError):
    """Raised when a summary cannot be safely created or rehydrated."""


class ConversationSummaryOutcome(StrEnum):
    SUPPORTED = "supported"
    SUPPORTED_WITH_LOSS = "supported_with_loss"
    UNSUPPORTED = "unsupported"


def _canonical(value: object) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ConversationSummaryError("summary value is not canonical JSON") from error


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversationSummaryError(f"{label} must be a positive integer")
    return value


def _token_units(text: str) -> int:
    """Return a stable provider-independent budget unit.

    This is intentionally not presented as a model tokenizer.  Four UTF-8
    bytes per unit is a conservative, deterministic accounting rule that lets
    callers bound work before selecting a provider-specific tokenizer.
    """

    return max(1, (len(text.encode("utf-8")) + 3) // 4)


@dataclass(frozen=True, slots=True)
class ConversationSummaryBudget:
    """Hard limits for transcript units and the complete summary payload."""

    max_transcript_tokens: int = 512
    max_summary_bytes: int = 32 * 1024

    def __post_init__(self) -> None:
        _positive_int(self.max_transcript_tokens, "maximum transcript token units")
        _positive_int(self.max_summary_bytes, "maximum summary bytes")

    def to_record(self) -> dict[str, object]:
        return {
            "max_transcript_tokens": self.max_transcript_tokens,
            "max_summary_bytes": self.max_summary_bytes,
        }


_DEFAULT_SUMMARY_BUDGET = ConversationSummaryBudget()


@dataclass(frozen=True, slots=True)
class ConversationTranscriptEntry:
    """One untrusted transcript entry retained only in the summary view."""

    role: str
    content: str
    sequence: int = 0

    def __post_init__(self) -> None:
        if self.role not in _SUPPORTED_ROLES:
            raise ConversationSummaryError("unsupported transcript role")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ConversationSummaryError("transcript content must be non-empty text")
        if isinstance(self.sequence, bool) or self.sequence < 0:
            raise ConversationSummaryError("transcript sequence must be non-negative")

    @property
    def token_units(self) -> int:
        return _token_units(self.content)

    @property
    def entry_id(self) -> str:
        return "transcript_" + hashlib.sha256(
            _canonical(
                {"role": self.role, "content": self.content, "sequence": self.sequence}
            ).encode("utf-8")
        ).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "sequence": self.sequence,
            "role": self.role,
            "content": self.content,
            "token_units": self.token_units,
        }


@dataclass(frozen=True, slots=True)
class ConversationSummaryLoss:
    """Explicit accounting for transcript content not retained in the view."""

    omitted_transcript_entries: int = 0
    omitted_transcript_tokens: int = 0
    omitted_transcript_digest: str | None = None
    unsupported_transcript_entries: int = 0
    unsupported_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value, label in (
            (self.omitted_transcript_entries, "omitted transcript entries"),
            (self.omitted_transcript_tokens, "omitted transcript token units"),
            (self.unsupported_transcript_entries, "unsupported transcript entries"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConversationSummaryError(f"{label} must be a non-negative integer")
        if self.omitted_transcript_entries == 0 and self.omitted_transcript_digest is not None:
            raise ConversationSummaryError("an empty omission cannot have a digest")
        if self.omitted_transcript_entries and self.omitted_transcript_digest is None:
            raise ConversationSummaryError("omitted transcript entries require a digest")
        if self.unsupported_roles != tuple(sorted(set(self.unsupported_roles))):
            raise ConversationSummaryError("unsupported transcript roles must be sorted and unique")

    @property
    def has_loss(self) -> bool:
        return bool(self.omitted_transcript_entries or self.unsupported_transcript_entries)

    def to_record(self) -> dict[str, object]:
        return {
            "omitted_transcript_entries": self.omitted_transcript_entries,
            "omitted_transcript_tokens": self.omitted_transcript_tokens,
            "omitted_transcript_digest": self.omitted_transcript_digest,
            "unsupported_transcript_entries": self.unsupported_transcript_entries,
            "unsupported_roles": list(self.unsupported_roles),
        }

    @property
    def marker(self) -> str | None:
        if not self.has_loss:
            return None
        return (
            "[conversation history loss: "
            f"omitted_entries={self.omitted_transcript_entries}, "
            f"omitted_token_units={self.omitted_transcript_tokens}, "
            f"unsupported_entries={self.unsupported_transcript_entries}]"
        )


@dataclass(frozen=True, slots=True)
class ConversationSummary:
    """A deterministic summary with canonical and untrusted zones separated."""

    summary_id: str
    campaign_id: str
    session_id: str
    state_version: int
    state_digest: str
    canonical_state: Mapping[str, object]
    canonical_evidence: tuple[Mapping[str, object], ...]
    untrusted_transcript: tuple[ConversationTranscriptEntry, ...]
    loss: ConversationSummaryLoss
    budget: ConversationSummaryBudget
    outcome: ConversationSummaryOutcome
    unsupported_reason: str | None = None
    schema_version: Literal[1] = CONVERSATION_SUMMARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONVERSATION_SUMMARY_SCHEMA_VERSION:
            raise ConversationSummaryError("unsupported conversation summary schema version")
        try:
            state = CampaignState.from_record(self.canonical_state)
        except CampaignStateError as error:
            raise ConversationSummaryError("summary contains invalid canonical state") from error
        if (
            state.campaign_id != self.campaign_id
            or state.session_id != self.session_id
            or state.state_version != self.state_version
            or state.digest != self.state_digest
            or _canonical(state.to_record()) != _canonical(self.canonical_state)
        ):
            raise ConversationSummaryError("summary canonical state linkage or digest is invalid")
        parsed_evidence: list[Mapping[str, object]] = []
        for record in self.canonical_evidence:
            try:
                evidence = CampaignEvidence.from_record(record)
            except CampaignStateError as error:
                raise ConversationSummaryError(
                    "summary contains invalid canonical evidence"
                ) from error
            parsed_evidence.append(evidence.to_record())
        if _canonical(parsed_evidence) != _canonical(list(self.canonical_evidence)):
            raise ConversationSummaryError("summary canonical evidence is not normalized")
        if len({str(item["evidence_id"]) for item in parsed_evidence}) != len(parsed_evidence):
            raise ConversationSummaryError("summary canonical evidence IDs must be unique")
        if self.outcome is ConversationSummaryOutcome.UNSUPPORTED and not self.unsupported_reason:
            raise ConversationSummaryError("unsupported summaries require a reason")
        if self.outcome is not ConversationSummaryOutcome.UNSUPPORTED and self.unsupported_reason:
            raise ConversationSummaryError("supported summaries cannot have an unsupported reason")
        expected_id = "conversation_summary_" + hashlib.sha256(
            _canonical(self._identity_record()).encode("utf-8")
        ).hexdigest()
        if self.summary_id != expected_id:
            raise ConversationSummaryError("summary ID does not match its canonical payload")

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "session_id": self.session_id,
            "state_version": self.state_version,
            "state_digest": self.state_digest,
            "canonical_state": dict(self.canonical_state),
            "canonical_evidence": [dict(item) for item in self.canonical_evidence],
            "untrusted_transcript": [item.to_record() for item in self.untrusted_transcript],
            "loss": self.loss.to_record(),
            "budget": self.budget.to_record(),
            "outcome": self.outcome.value,
            "unsupported_reason": self.unsupported_reason,
        }

    @property
    def digest(self) -> str:
        return _digest(self._record_without_digest())

    def _record_without_digest(self) -> dict[str, object]:
        return {
            "record_type": "bounded_conversation_summary",
            "schema_version": self.schema_version,
            "summary_id": self.summary_id,
            "campaign_id": self.campaign_id,
            "session_id": self.session_id,
            "state_version": self.state_version,
            "state_digest": self.state_digest,
            "canonical_state": dict(self.canonical_state),
            "canonical_evidence": [dict(item) for item in self.canonical_evidence],
            "untrusted_transcript": [item.to_record() for item in self.untrusted_transcript],
            "loss": self.loss.to_record(),
            "budget": self.budget.to_record(),
            "outcome": self.outcome.value,
            "unsupported_reason": self.unsupported_reason,
        }

    def to_record(self) -> dict[str, object]:
        record = self._record_without_digest()
        record["summary_digest"] = self.digest
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> ConversationSummary:
        if not isinstance(value, Mapping):
            raise ConversationSummaryError("conversation summary must be an object")
        record = dict(value)
        expected = {
            "record_type",
            "schema_version",
            "summary_id",
            "campaign_id",
            "session_id",
            "state_version",
            "state_digest",
            "canonical_state",
            "canonical_evidence",
            "untrusted_transcript",
            "loss",
            "budget",
            "outcome",
            "unsupported_reason",
            "summary_digest",
        }
        if set(record) != expected or record["record_type"] != "bounded_conversation_summary":
            raise ConversationSummaryError("conversation summary has missing or unknown fields")
        try:
            outcome = ConversationSummaryOutcome(cast(str, record["outcome"]))
            budget_record = cast(Mapping[str, object], record["budget"])
            budget = ConversationSummaryBudget(
                cast(int, budget_record["max_transcript_tokens"]),
                cast(int, budget_record["max_summary_bytes"]),
            )
            loss_record = cast(Mapping[str, object], record["loss"])
            loss = ConversationSummaryLoss(
                cast(int, loss_record["omitted_transcript_entries"]),
                cast(int, loss_record["omitted_transcript_tokens"]),
                None
                if loss_record["omitted_transcript_digest"] is None
                else cast(str, loss_record["omitted_transcript_digest"]),
                cast(int, loss_record["unsupported_transcript_entries"]),
                tuple(cast(list[str], loss_record["unsupported_roles"])),
            )
            transcript_records = cast(list[Mapping[str, object]], record["untrusted_transcript"])
            transcript = tuple(
                ConversationTranscriptEntry(
                    cast(str, item["role"]),
                    cast(str, item["content"]),
                    cast(int, item["sequence"]),
                )
                for item in transcript_records
            )
            result = cls(
                cast(str, record["summary_id"]),
                cast(str, record["campaign_id"]),
                cast(str, record["session_id"]),
                cast(int, record["state_version"]),
                cast(str, record["state_digest"]),
                cast(Mapping[str, object], record["canonical_state"]),
                tuple(cast(Mapping[str, object], item) for item in record["canonical_evidence"]),
                transcript,
                loss,
                budget,
                outcome,
                None
                if record["unsupported_reason"] is None
                else cast(str, record["unsupported_reason"]),
                cast(Literal[1], record["schema_version"]),
            )
        except (KeyError, TypeError, ValueError, ConversationSummaryError) as error:
            if isinstance(error, ConversationSummaryError):
                raise
            raise ConversationSummaryError("conversation summary is malformed") from error
        if record["summary_digest"] != result.digest:
            raise ConversationSummaryError("conversation summary digest does not match its payload")
        return result


@dataclass(frozen=True, slots=True)
class ConversationSummaryResult:
    """The explicit result of a bounded summary attempt."""

    outcome: ConversationSummaryOutcome
    summary: ConversationSummary | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.outcome is ConversationSummaryOutcome.UNSUPPORTED:
            if self.summary is not None or not self.reason:
                raise ConversationSummaryError("unsupported summary result requires only a reason")
        elif self.summary is None or self.reason is not None:
            raise ConversationSummaryError("supported summary result requires a summary only")


def _entry_from_raw(value: object, sequence: int) -> ConversationTranscriptEntry | None:
    if isinstance(value, ConversationTranscriptEntry):
        return ConversationTranscriptEntry(value.role, value.content, sequence)
    if not isinstance(value, Mapping) or set(value) - {"role", "content"}:
        return None
    role = value.get("role")
    content = value.get("content")
    if not isinstance(role, str) or role not in _SUPPORTED_ROLES:
        return None
    if not isinstance(content, str) or not content.strip():
        return None
    return ConversationTranscriptEntry(role, content, sequence)


def _make_summary(
    state: CampaignState,
    evidence: tuple[CampaignEvidence, ...],
    transcript: tuple[ConversationTranscriptEntry, ...],
    loss: ConversationSummaryLoss,
    budget: ConversationSummaryBudget,
) -> ConversationSummary:
    outcome = (
        ConversationSummaryOutcome.SUPPORTED_WITH_LOSS
        if loss.has_loss
        else ConversationSummaryOutcome.SUPPORTED
    )
    identity = {
        "schema_version": CONVERSATION_SUMMARY_SCHEMA_VERSION,
        "campaign_id": state.campaign_id,
        "session_id": state.session_id,
        "state_version": state.state_version,
        "state_digest": state.digest,
        "canonical_state": state.to_record(),
        "canonical_evidence": [item.to_record() for item in evidence],
        "untrusted_transcript": [item.to_record() for item in transcript],
        "loss": loss.to_record(),
        "budget": budget.to_record(),
        "outcome": outcome.value,
        "unsupported_reason": None,
    }
    summary_id = "conversation_summary_" + hashlib.sha256(
        _canonical(identity).encode("utf-8")
    ).hexdigest()
    return ConversationSummary(
        summary_id,
        state.campaign_id,
        state.session_id,
        state.state_version,
        state.digest,
        state.to_record(),
        tuple(item.to_record() for item in evidence),
        transcript,
        loss,
        budget,
        outcome,
    )


def summarize_conversation(
    state: CampaignState,
    evidence: Sequence[CampaignEvidence],
    transcript: Sequence[object],
    *,
    budget: ConversationSummaryBudget = _DEFAULT_SUMMARY_BUDGET,
) -> ConversationSummaryResult:
    """Build a bounded summary without promoting transcript text to authority.

    The newest complete supported transcript entries are retained.  Older or
    unsupported entries are never silently dropped: the result carries a
    count, digest/role marker, and deterministic loss outcome.  Canonical
    state/evidence are never truncated; if their required payload exceeds the
    hard byte ceiling the operation returns ``unsupported``.
    """

    canonical_evidence = tuple(evidence)
    normalized: list[ConversationTranscriptEntry] = []
    unsupported_count = 0
    unsupported_roles: set[str] = set()
    for sequence, raw in enumerate(transcript):
        entry = _entry_from_raw(raw, sequence)
        if entry is None:
            unsupported_count += 1
            if isinstance(raw, Mapping) and isinstance(raw.get("role"), str):
                unsupported_roles.add(cast(str, raw["role"]))
        else:
            normalized.append(entry)

    retained: list[ConversationTranscriptEntry] = []
    omitted: list[ConversationTranscriptEntry] = []
    remaining = budget.max_transcript_tokens
    for entry in reversed(normalized):
        if entry.token_units <= remaining:
            retained.append(entry)
            remaining -= entry.token_units
        else:
            omitted.append(entry)
    retained.reverse()
    omitted.reverse()
    loss = ConversationSummaryLoss(
        len(omitted),
        sum(item.token_units for item in omitted),
        None if not omitted else _digest([item.to_record() for item in omitted]),
        unsupported_count,
        tuple(sorted(unsupported_roles)),
    )

    while True:
        summary = _make_summary(state, canonical_evidence, tuple(retained), loss, budget)
        encoded_bytes = len(summary.canonical_json().encode("utf-8"))
        if encoded_bytes <= budget.max_summary_bytes:
            return ConversationSummaryResult(summary.outcome, summary)
        if not retained:
            return ConversationSummaryResult(
                ConversationSummaryOutcome.UNSUPPORTED,
                reason=(
                    "canonical state/evidence and required loss markers exceed "
                    "summary byte budget"
                ),
            )
        removed = retained.pop(0)
        omitted.append(removed)
        loss = ConversationSummaryLoss(
            len(omitted),
            sum(item.token_units for item in omitted),
            _digest([item.to_record() for item in omitted]),
            unsupported_count,
            tuple(sorted(unsupported_roles)),
        )


@dataclass(frozen=True, slots=True)
class ConversationRehydration:
    """State restored from a summary after checking it against trusted storage."""

    state: CampaignState
    evidence: tuple[CampaignEvidence, ...]
    untrusted_transcript: tuple[ConversationTranscriptEntry, ...]
    loss: ConversationSummaryLoss

    def to_provider_context(self) -> dict[str, object]:
        """Return an explicitly zoned context suitable for a provider adapter."""

        return {
            "record_type": "conversation_rehydration_context",
            "schema_version": CONVERSATION_SUMMARY_SCHEMA_VERSION,
            "canonical": {
                "state": self.state.to_record(),
                "evidence": [item.to_record() for item in self.evidence],
            },
            "untrusted": {
                "transcript": [item.to_record() for item in self.untrusted_transcript],
                "loss": self.loss.to_record(),
            },
            "authority": {
                "state_digest": self.state.digest,
                "state_version": self.state.state_version,
                "evidence_cursor": self.state.evidence_cursor.to_record(),
            },
        }


def rehydrate_conversation(
    summary: ConversationSummary,
    *,
    authoritative_state: CampaignState,
    authoritative_evidence: Sequence[CampaignEvidence],
) -> ConversationRehydration:
    """Rehydrate a summary only when trusted state and evidence still match."""

    if summary.outcome is ConversationSummaryOutcome.UNSUPPORTED:
        raise ConversationSummaryError("unsupported conversation summary cannot be rehydrated")
    try:
        embedded_state = CampaignState.from_record(summary.canonical_state)
    except CampaignStateError as error:
        raise ConversationSummaryError("summary canonical state is invalid") from error
    if embedded_state != authoritative_state or summary.state_digest != authoritative_state.digest:
        raise ConversationSummaryError("summary is stale or canonical state was altered")
    expected_evidence = tuple(authoritative_evidence)
    embedded_evidence = tuple(
        CampaignEvidence.from_record(item) for item in summary.canonical_evidence
    )
    if embedded_evidence != expected_evidence:
        raise ConversationSummaryError(
            "summary evidence is stale or canonical evidence was altered"
        )
    return ConversationRehydration(
        authoritative_state,
        expected_evidence,
        summary.untrusted_transcript,
        summary.loss,
    )


def reconnect_conversation(
    store: CampaignStateStore,
    summary: ConversationSummary,
) -> ConversationRehydration:
    """Reconnect against the WAL-backed campaign store, never by replaying chat."""

    state = store.reconnect(summary.campaign_id, summary.session_id)
    return rehydrate_conversation(
        summary,
        authoritative_state=state,
        authoritative_evidence=store.evidence(summary.campaign_id),
    )


__all__ = [
    "CONVERSATION_SUMMARY_SCHEMA_VERSION",
    "ConversationRehydration",
    "ConversationSummary",
    "ConversationSummaryBudget",
    "ConversationSummaryError",
    "ConversationSummaryLoss",
    "ConversationSummaryOutcome",
    "ConversationSummaryResult",
    "ConversationTranscriptEntry",
    "reconnect_conversation",
    "rehydrate_conversation",
    "summarize_conversation",
]

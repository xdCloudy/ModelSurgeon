"""Typed integration for the bounded conversational user journey.

The journey facade composes the existing interpretation, preview, execution,
campaign, evidence, and provider boundaries.  It does not persist transcript
text and it never promotes provider output, predictions, or unapproved plans
to canonical state.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from modelsurgeon.config import ProviderConfig, Settings
from modelsurgeon.experiments.identity import canonical_identity_json

from .campaign_state import CampaignState, CampaignStateError, CampaignStateStore
from .evidence_query import (
    EvidenceQuery,
    EvidenceQueryEngine,
    EvidenceQueryOutcome,
    EvidenceQueryStaleError,
    EvidenceQueryStatus,
    EvidenceSnapshot,
)
from .execution import ChatExecutionError, ChatExecutionRecord
from .session import ChatSession, ChatSessionError, ChatTurnResult

if TYPE_CHECKING:
    from modelsurgeon.providers import ProviderDiagnostic
    from modelsurgeon.search.spec_preview import SpecDiff, SpecPreview

JOURNEY_SCHEMA_VERSION = 1


class JourneyError(ValueError):
    """Raised when a typed journey operation cannot safely proceed."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class JourneyPhase(StrEnum):
    INTERPRETATION = "interpretation"
    CLARIFICATION = "clarification"
    PREVIEW = "preview"
    EXECUTION = "execution"
    LIFECYCLE = "lifecycle"
    EXPLANATION = "explanation"
    DIAGNOSTICS = "diagnostics"


class JourneyOutcome(StrEnum):
    SUPPORTED = "supported"
    CLARIFICATION_REQUIRED = "clarification_required"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"
    REFUSED = "refused"


def _json_record(value: Mapping[str, object], label: str) -> dict[str, object]:
    try:
        decoded = json.loads(canonical_identity_json(dict(value)))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise JourneyError("record_invalid", f"{label} is not canonical JSON") from error
    if not isinstance(decoded, dict):  # pragma: no cover - guarded by the input type
        raise JourneyError("record_invalid", f"{label} must be an object")
    return cast(dict[str, object], decoded)


def _outcome(value: object) -> JourneyOutcome:
    raw = getattr(value, "value", value)
    if raw == "executable":
        return JourneyOutcome.SUPPORTED
    if raw in {"clarification_required", "ambiguous"}:
        return JourneyOutcome.CLARIFICATION_REQUIRED
    if raw in {"refused"}:
        return JourneyOutcome.REFUSED
    try:
        return JourneyOutcome(str(raw))
    except ValueError:
        return JourneyOutcome.FAILED


@dataclass(frozen=True, slots=True)
class JourneyResponse:
    """One canonical response envelope shared by Python and CLI callers."""

    phase: JourneyPhase
    outcome: JourneyOutcome
    session_id: str
    request_id: str | None
    campaign_id: str | None
    state_version: int | None
    state_digest: str | None
    record: Mapping[str, object]
    evidence_refs: tuple[str, ...] = ()
    schema_version: int = JOURNEY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != JOURNEY_SCHEMA_VERSION:
            raise JourneyError("schema_version", "unsupported journey response schema version")
        if not self.session_id.startswith("chat_session_"):
            raise JourneyError("session_id", "journey session identity is invalid")
        if self.request_id is not None and not self.request_id.startswith("chat_request_"):
            raise JourneyError("request_id", "journey request identity is invalid")
        if self.state_version is not None and (
            isinstance(self.state_version, bool) or self.state_version < 0
        ):
            raise JourneyError("state_version", "journey state version is invalid")
        if self.evidence_refs != tuple(sorted(set(self.evidence_refs))):
            raise JourneyError("evidence_refs", "journey evidence references must be sorted")
        object.__setattr__(self, "record", _json_record(self.record, "journey record"))

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "conversational_journey_response",
            "schema_version": self.schema_version,
            "phase": self.phase.value,
            "outcome": self.outcome.value,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "campaign_id": self.campaign_id,
            "state_version": self.state_version,
            "state_digest": self.state_digest,
            "record": dict(self.record),
            "evidence_refs": list(self.evidence_refs),
        }

    def canonical_json(self) -> str:
        return canonical_identity_json(self.to_record())


@dataclass(frozen=True, slots=True)
class JourneyContext:
    """The last trusted campaign snapshot held by a journey instance."""

    campaign_id: str
    session_id: str
    state_version: int
    state_digest: str

    @classmethod
    def from_state(cls, state: CampaignState) -> JourneyContext:
        return cls(state.campaign_id, state.session_id, state.state_version, state.digest)

    def to_record(self) -> dict[str, object]:
        return {
            "campaign_id": self.campaign_id,
            "session_id": self.session_id,
            "state_version": self.state_version,
            "state_digest": self.state_digest,
        }


class ConversationalJourney:
    """Compose the full bounded journey without adding an execution authority.

    The instance retains only typed turn/preview handles and a digest of the
    last trusted campaign snapshot.  A reconnect refreshes that snapshot;
    explanation queries otherwise fail closed when canonical state changes.
    """

    def __init__(
        self,
        session: ChatSession,
        *,
        campaign_state_path: str | Path | None = None,
    ) -> None:
        self.session = session
        self.adapter = session.execution_adapter
        selected_path = campaign_state_path
        if selected_path is None and self.adapter is not None:
            selected_path = self.adapter.campaign_state_path
        self.campaign_state_path = None if selected_path is None else Path(selected_path)
        self._turn: ChatTurnResult | None = None
        self._context: dict[str, JourneyContext] = {}

    @property
    def current_turn(self) -> ChatTurnResult | None:
        return self._turn

    @contextmanager
    def _store(self) -> Iterator[CampaignStateStore]:
        if self.campaign_state_path is None:
            raise JourneyError("campaign_state_unavailable", "campaign state path is required")
        with CampaignStateStore(self.campaign_state_path) as store:
            yield store

    def _response(
        self,
        phase: JourneyPhase,
        outcome: JourneyOutcome,
        record: Mapping[str, object],
        *,
        request_id: str | None = None,
        campaign_id: str | None = None,
        state: CampaignState | None = None,
        evidence_refs: Sequence[str] = (),
    ) -> JourneyResponse:
        if state is not None:
            self._context[state.campaign_id] = JourneyContext.from_state(state)
        return JourneyResponse(
            phase,
            outcome,
            self.session.bootstrap.session_id,
            request_id,
            None if state is None else state.campaign_id if campaign_id is None else campaign_id,
            None if state is None else state.state_version,
            None if state is None else state.digest,
            record,
            tuple(sorted(set(evidence_refs))),
        )

    def interpret(self, request: str) -> JourneyResponse:
        """Interpret one request; no plan or artifact is created."""

        try:
            turn = self.session.interpret(request)
        except ChatSessionError as error:
            raise JourneyError(error.code, str(error)) from error
        self._turn = turn
        phase = (
            JourneyPhase.CLARIFICATION
            if turn.clarification is not None and not turn.clarification.executable
            else JourneyPhase.INTERPRETATION
        )
        return self._response(
            phase,
            _outcome(turn.outcome),
            turn.to_record(),
            request_id=turn.request_id,
        )

    def answer_clarification(self, answer: object) -> JourneyResponse:
        """Apply one typed answer and recompute the canonical preview."""

        if self._turn is None:
            raise JourneyError("clarification_unavailable", "there is no active journey turn")
        from .clarification import ClarificationAnswer

        if not isinstance(answer, ClarificationAnswer):
            raise JourneyError("clarification_type", "clarification answer must be typed")
        try:
            self._turn = self.session.answer_clarification(self._turn, answer)
        except ChatSessionError as error:
            raise JourneyError(error.code, str(error)) from error
        turn = self._turn
        return self._response(
            JourneyPhase.PREVIEW if turn.spec_preview is not None else JourneyPhase.CLARIFICATION,
            _outcome(turn.outcome),
            turn.to_record(),
            request_id=turn.request_id,
        )

    def preview(self) -> JourneyResponse:
        """Resolve a read-only engine plan for the exact current preview."""

        if self._turn is None:
            raise JourneyError("preview_unavailable", "interpret a request before previewing")
        try:
            self._turn = self.session.preview_plan(self._turn)
        except (ChatSessionError, ChatExecutionError) as error:
            raise JourneyError(getattr(error, "code", "preview_failed"), str(error)) from error
        execution = self._turn.execution
        assert execution is not None
        return self._response(
            JourneyPhase.PREVIEW,
            _outcome(execution.outcome),
            self._turn.to_record(),
            request_id=self._turn.request_id,
            evidence_refs=execution.evidence_refs,
        )

    def execute(self, approval_id: str | None) -> JourneyResponse:
        """Execute only the exact preview, with the adapter's scoped approval gate."""

        if self._turn is None:
            raise JourneyError("execution_unavailable", "preview a request before execution")
        if approval_id is None:
            raise JourneyError("approval_required", "explicit approval is required for execution")
        try:
            self._turn = self.session.execute_plan(self._turn, approval_id)
        except (ChatSessionError, ChatExecutionError) as error:
            raise JourneyError(getattr(error, "code", "execution_failed"), str(error)) from error
        execution = self._turn.execution
        assert execution is not None
        state = self._load_execution_state(execution)
        return self._response(
            JourneyPhase.EXECUTION,
            _outcome(execution.outcome),
            self._turn.to_record(),
            request_id=self._turn.request_id,
            campaign_id=execution.campaign_id,
            state=state,
            evidence_refs=execution.evidence_refs,
        )

    def _load_execution_state(self, execution: ChatExecutionRecord) -> CampaignState | None:
        if execution.campaign_id is None:
            return None
        try:
            with self._store() as store:
                return store.reconnect(execution.campaign_id, self.session.bootstrap.session_id)
        except CampaignStateError as error:
            raise JourneyError("campaign_state_invalid", str(error)) from error

    def reconnect(self, campaign_id: str) -> JourneyResponse:
        """Refresh from canonical state by identity; transcript is not consulted."""

        try:
            with self._store() as store:
                state = store.reconnect(campaign_id, self.session.bootstrap.session_id)
        except CampaignStateError as error:
            raise JourneyError("reconnect_failed", str(error)) from error
        return self._response(
            JourneyPhase.LIFECYCLE,
            _outcome(state.outcome),
            state.to_record(),
            campaign_id=state.campaign_id,
            state=state,
        )

    def _lifecycle_approval(self, campaign_id: str, approval_id: str | None) -> CampaignState:
        try:
            with self._store() as store:
                state = store.reconnect(campaign_id, self.session.bootstrap.session_id)
        except CampaignStateError as error:
            raise JourneyError("campaign_state_invalid", str(error)) from error
        if approval_id is None:
            raise JourneyError(
                "approval_required", "explicit approval is required for lifecycle work"
            )
        if not state.approval.active or state.approval.approval_id != approval_id:
            raise JourneyError(
                "approval_invalid",
                "approval is missing, expired, or bound to another campaign",
            )
        return state

    def pause(self, campaign_id: str) -> JourneyResponse:
        """Pause cooperatively; pausing is the safety-preserving lifecycle action."""

        try:
            if self.adapter is not None:
                state = self.adapter.pause(campaign_id, self.session.bootstrap.session_id)
            else:
                with self._store() as store:
                    state = store.pause(campaign_id, self.session.bootstrap.session_id)
        except (ChatExecutionError, CampaignStateError) as error:
            raise JourneyError(getattr(error, "code", "pause_failed"), str(error)) from error
        return self._response(
            JourneyPhase.LIFECYCLE, _outcome(state.outcome), state.to_record(), state=state
        )

    def resume(self, campaign_id: str, approval_id: str | None) -> JourneyResponse:
        """Resume only with a current approval bound to the canonical campaign."""

        self._lifecycle_approval(campaign_id, approval_id)
        try:
            if self.adapter is not None:
                state = self.adapter.resume(campaign_id, self.session.bootstrap.session_id)
            else:
                with self._store() as store:
                    state = store.resume(campaign_id, self.session.bootstrap.session_id)
        except (ChatExecutionError, CampaignStateError) as error:
            raise JourneyError(getattr(error, "code", "resume_failed"), str(error)) from error
        return self._response(
            JourneyPhase.LIFECYCLE, _outcome(state.outcome), state.to_record(), state=state
        )

    def cancel(self, campaign_id: str, approval_id: str | None) -> JourneyResponse:
        """Cancel only with a current scoped approval; cancellation is terminal."""

        self._lifecycle_approval(campaign_id, approval_id)
        try:
            if self.adapter is not None:
                state = self.adapter.cancel(campaign_id, self.session.bootstrap.session_id)
            else:
                with self._store() as store:
                    state = store.cancel(campaign_id, self.session.bootstrap.session_id)
        except (ChatExecutionError, CampaignStateError) as error:
            raise JourneyError(getattr(error, "code", "cancel_failed"), str(error)) from error
        return self._response(
            JourneyPhase.LIFECYCLE, _outcome(state.outcome), state.to_record(), state=state
        )

    def explain(
        self,
        campaign_id: str,
        *,
        fields: Sequence[str] = (),
        outcomes: Sequence[EvidenceQueryOutcome] = (),
        refresh: bool = False,
    ) -> JourneyResponse:
        """Render a deterministic explanation from canonical evidence only."""

        try:
            with self._store() as store:
                state = store.reconnect(campaign_id, self.session.bootstrap.session_id)
                prior = self._context.get(campaign_id)
                expected = None if refresh or prior is None else prior.state_digest
                snapshot = EvidenceSnapshot.from_store(store, campaign_id)
                request = EvidenceQuery(
                    campaign_id,
                    fields=tuple(fields),
                    outcomes=tuple(sorted(set(outcomes), key=lambda item: item.value)),
                    expected_state_digest=expected,
                )
                response = EvidenceQueryEngine(snapshot).query_current(store, request)
        except EvidenceQueryStaleError as error:
            raise JourneyError("stale_context", str(error)) from error
        except (CampaignStateError, ValueError) as error:
            if isinstance(error, JourneyError):
                raise
            raise JourneyError("explanation_failed", str(error)) from error
        from modelsurgeon.explain import render_claim_evidence

        try:
            explanation = render_claim_evidence(response)
        except ValueError as error:
            raise JourneyError("explanation_failed", str(error)) from error
        self._context[campaign_id] = JourneyContext.from_state(state)
        outcome = (
            JourneyOutcome.SUPPORTED
            if response.status is EvidenceQueryStatus.COMPLETE
            else JourneyOutcome.UNKNOWN
        )
        return self._response(
            JourneyPhase.EXPLANATION,
            outcome,
            explanation.to_record(),
            campaign_id=campaign_id,
            state=state,
            evidence_refs=tuple(item.evidence_id for item in response.records),
        )

    def present_alternatives(self, explanation: object) -> JourneyResponse:
        """Expose a typed infeasibility/Pareto explanation without rewriting it."""

        from modelsurgeon.explain import FeasibilityExplanation, ParetoAlternativesExplanation

        if not isinstance(explanation, (FeasibilityExplanation, ParetoAlternativesExplanation)):
            raise JourneyError(
                "alternatives_invalid",
                "infeasibility alternatives must be a canonical explanation record",
            )
        record = explanation.to_record()
        evidence_refs = tuple(
            sorted(
                item.evidence_id
                for item in getattr(explanation, "retained_evidence", ())
            )
        )
        return self._response(
            JourneyPhase.EXPLANATION,
            JourneyOutcome.UNKNOWN,
            record,
            evidence_refs=evidence_refs,
        )

    def diagnose_provider(
        self,
        settings: Settings | ProviderConfig,
        *,
        offline: bool = False,
    ) -> JourneyResponse:
        """Return provider availability as a typed, redacted journey record."""

        diagnostic = self.provider_diagnostics(settings, offline=offline)
        outcome = {
            "supported": JourneyOutcome.SUPPORTED,
            "configured": JourneyOutcome.SUPPORTED,
            "disabled": JourneyOutcome.UNSUPPORTED,
            "unknown": JourneyOutcome.UNKNOWN,
            "unavailable": JourneyOutcome.UNSUPPORTED,
            "unsupported": JourneyOutcome.UNSUPPORTED,
        }.get(diagnostic.status.value, JourneyOutcome.FAILED)
        return self._response(
            JourneyPhase.DIAGNOSTICS,
            outcome,
            diagnostic.to_record(),
        )

    @staticmethod
    def plan_diff(previous: SpecPreview, current: SpecPreview) -> SpecDiff:
        """Return the visible typed diff; it never grants execution authority."""

        from modelsurgeon.search.spec_preview import diff_spec_previews

        return diff_spec_previews(previous, current)

    @staticmethod
    def provider_diagnostics(
        settings: Settings | ProviderConfig,
        *,
        offline: bool = False,
    ) -> ProviderDiagnostic:
        """Use the same redacted diagnostic record as the direct CLI path."""

        from modelsurgeon.providers import provider_diagnostics

        return provider_diagnostics(settings, offline=offline)


__all__ = [
    "JOURNEY_SCHEMA_VERSION",
    "ConversationalJourney",
    "JourneyContext",
    "JourneyError",
    "JourneyOutcome",
    "JourneyPhase",
    "JourneyResponse",
]

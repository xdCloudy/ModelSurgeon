"""Conformance tests for the replaceable text-model provider boundary."""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest

from modelsurgeon.conversation import (
    CancellationToken,
    ClarificationRequest,
    ExplanationProviderOutput,
    ExplanationRequest,
    IntentField,
    IntentOutcome,
    IntentProvenance,
    IntentRecord,
    InterpretationStep,
    InterpretIntentRequest,
    NullTextModelProvider,
    ProviderBudget,
    ProviderCapability,
    ProviderCapabilityCard,
    ProviderContractError,
    ProviderFailure,
    ProviderFailureCode,
    ProviderKind,
    ProviderLimits,
    ProviderModelIdentity,
    ProviderOperation,
    ProviderOutcome,
    ProviderProvenance,
    ProviderRequest,
    ProviderResult,
    ProviderStreamEvent,
    SourceSpan,
    invoke_provider,
    request_digest,
    result_from_raw_output,
)

REQUEST = "reduce latency while retaining at least 0.98 quality"
IDENTITY = ProviderModelIdentity("fixture-provider", "fixture-model", "revision-1")
CARD = ProviderCapabilityCard(
    IDENTITY,
    ProviderKind.LOCAL,
    "provider-revision-1",
    (
        ProviderCapability.CLARIFY_INTENT,
        ProviderCapability.EXPLAIN_EVIDENCE,
        ProviderCapability.INTERPRET_INTENT,
        ProviderCapability.STREAMING,
        ProviderCapability.STRUCTURED_OUTPUT,
    ),
    ProviderLimits(4096, 1024, 8192, max_wall_seconds=10.0),
    ("conversational_intent:1",),
)


def _intent() -> IntentRecord:
    provenance = IntentProvenance(
        "request-revision-1",
        IDENTITY.provider_id,
        "provider-revision-1",
        "tool-revision-1",
        ("evidence:request",),
    )
    return IntentRecord(
        REQUEST,
        (
            # The span text must remain an exact slice of the request.
            SourceSpan("span-request", 0, len(REQUEST), REQUEST),
        ),
        (IntentField("objective.metric", "latency", None, 0.99, ("span-request",)),),
        (),
        (
            InterpretationStep(
                "step-normalize",
                "normalize",
                ("span-request",),
                ("objective.metric",),
                0.99,
                ("evidence:request",),
            ),
        ),
        provenance,
        IntentOutcome.EXECUTABLE,
        ("request normalized",),
        {"contract_id": "objective_contract_example", "schema_version": 1},
    )


class FixtureProvider:
    identity = IDENTITY
    capability_card = CARD

    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.cancelled_ids: list[str] = []

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def call(self, request: ProviderRequest, *, cancellation: CancellationToken) -> ProviderResult:
        if self.delay:
            time.sleep(self.delay)
        cancellation.raise_if_cancelled()
        if isinstance(request, InterpretIntentRequest):
            return result_from_raw_output(
                self,
                request,
                {"operation": request.operation.value, "intent": _intent().to_record()},
            )
        raise AssertionError("fixture only implements interpretation")

    def stream(
        self, request: ProviderRequest, *, cancellation: CancellationToken
    ) -> Iterator[ProviderStreamEvent]:
        yield ProviderStreamEvent(request.request_id, 0, "bounded")

    def cancel(self, request_id: str) -> bool:
        self.cancelled_ids.append(request_id)
        return True


def test_capability_card_and_request_identity_are_deterministic() -> None:
    request = InterpretIntentRequest("request-1", REQUEST)
    assert CARD.canonical_json() == CARD.canonical_json()
    assert CARD.card_id.startswith("provider_card_")
    assert request_digest(request).startswith("sha256:")
    assert request.to_record()["operation"] == "interpret_intent"

    with pytest.raises(ProviderContractError, match="sorted"):
        ProviderCapabilityCard(
            IDENTITY,
            ProviderKind.LOCAL,
            "provider-revision-1",
            (ProviderCapability.STRUCTURED_OUTPUT, ProviderCapability.INTERPRET_INTENT),
            ProviderLimits(4096, 1024, 8192),
        )


def test_typed_requests_do_not_expose_execution_authority() -> None:
    request = ClarificationRequest("request-2", _intent())
    record = request.to_record()
    assert set(record) == {
        "schema_version",
        "request_id",
        "operation",
        "intent_id",
        "intent_schema_version",
        "question",
        "budget",
    }
    assert not hasattr(request, "executor")
    assert not hasattr(request, "tool_callback")


def test_structured_output_is_validated_and_provenance_is_retained() -> None:
    request = InterpretIntentRequest("request-3", REQUEST)
    result = result_from_raw_output(
        FixtureProvider(),
        request,
        {"operation": "interpret_intent", "intent": _intent().to_record()},
    )
    assert result.outcome is ProviderOutcome.SUPPORTED
    assert result.output is not None
    assert result.provenance.request_digest == request_digest(request)
    assert result.canonical_json() == result.canonical_json()


def test_malformed_output_and_secret_diagnostics_are_safe_records() -> None:
    request = InterpretIntentRequest("request-4", REQUEST)
    result = result_from_raw_output(
        FixtureProvider(),
        request,
        {"operation": "interpret_intent", "intent": {"execute": "rm -rf"}},
    )
    assert result.outcome is ProviderOutcome.MALFORMED_OUTPUT
    assert result.output is None
    assert result.failure is not None

    failure = ProviderFailure(
        ProviderFailureCode.PROTOCOL,
        ProviderOperation.INTERPRET_INTENT,
        "authorization=Bearer-secret-token",
        request.request_id,
    )
    assert "Bearer-secret-token" not in failure.to_record()["detail"]
    assert failure.to_record()["detail"] == "authorization=<redacted>"


def test_invoke_provider_handles_timeout_cancellation_and_no_provider() -> None:
    request = InterpretIntentRequest("request-5", REQUEST, budget=ProviderBudget(0.01))
    slow = FixtureProvider(delay=0.05)
    timed_out = invoke_provider(slow, request)
    assert timed_out.outcome is ProviderOutcome.TIMEOUT
    assert timed_out.failure is not None
    assert slow.cancelled_ids == [request.request_id]

    token = CancellationToken()
    token.cancel()
    cancelled = invoke_provider(
        FixtureProvider(), InterpretIntentRequest("request-6", REQUEST), cancellation=token
    )
    assert cancelled.outcome is ProviderOutcome.CANCELLED

    no_provider = invoke_provider(
        NullTextModelProvider(), InterpretIntentRequest("request-7", REQUEST)
    )
    assert no_provider.outcome is ProviderOutcome.UNSUPPORTED
    assert no_provider.unsupported_capabilities == (
        ProviderCapability.INTERPRET_INTENT,
        ProviderCapability.STRUCTURED_OUTPUT,
    )


def test_provider_result_rejects_mismatched_failure_identity() -> None:
    request = InterpretIntentRequest("request-8", REQUEST)
    provenance = ProviderProvenance(
        IDENTITY.provider_id, "provider-revision-1", request_digest(request)
    )
    with pytest.raises(ProviderContractError, match="failure request ID"):
        ProviderResult(
            request.request_id,
            request.operation,
            IDENTITY,
            ProviderOutcome.FAILED,
            provenance,
            failure=ProviderFailure(
                ProviderFailureCode.INTERNAL,
                request.operation,
                "failed",
                "different-request",
            ),
        )


def test_stream_event_requires_content_and_final_output_to_be_terminal() -> None:
    with pytest.raises(ProviderContractError, match="text or a final output"):
        ProviderStreamEvent("request-9", 0)
    with pytest.raises(ProviderContractError, match="done"):
        ProviderStreamEvent(
            "request-9", 0, final_output=ExplanationProviderOutput("done")
        )


def test_explanation_request_is_typed_evidence_only() -> None:
    request = ExplanationRequest("request-10", ({"record_type": "evidence", "value": 1},))
    assert request.to_record()["operation"] == "explain_evidence"
    with pytest.raises(ProviderContractError, match="at least one"):
        ExplanationRequest("request-11", ())

"""Replaceable text-model provider contracts for the conversational control plane.

Providers are deliberately limited to interpretation, clarification, and
evidence-grounded explanation.  They never receive an executor, a model
session, a filesystem handle, or a generic tool callback.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from modelsurgeon.conversation.intent import IntentRecord, IntentRecordError
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.provider_kind import ProviderKind

TEXT_PROVIDER_SCHEMA_VERSION: Literal[1] = 1
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SECRET = re.compile(
    r"(?i)(\b(?:api[_-]?key|authorization|password|secret|token)\b\s*[:=]\s*)([^\s,;]+)"
)

type JSONValue = (
    bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None
)


class ProviderContractError(ValueError):
    """Raised when a provider contract value is unsafe or malformed."""


class ProviderCapability(StrEnum):
    INTERPRET_INTENT = "interpret_intent"
    CLARIFY_INTENT = "clarify_intent"
    EXPLAIN_EVIDENCE = "explain_evidence"
    STREAMING = "streaming"
    STRUCTURED_OUTPUT = "structured_output"


class ProviderOperation(StrEnum):
    INTERPRET_INTENT = "interpret_intent"
    CLARIFY_INTENT = "clarify_intent"
    EXPLAIN_EVIDENCE = "explain_evidence"


class ProviderOutcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    MALFORMED_OUTPUT = "malformed_output"


class ProviderFailureCode(StrEnum):
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    MALFORMED_OUTPUT = "malformed_output"
    AUTHENTICATION = "authentication"
    RATE_LIMITED = "rate_limited"
    PROTOCOL = "protocol"
    LIMIT_EXCEEDED = "limit_exceeded"
    INTERNAL = "internal"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderContractError(f"{label} must be non-empty text")
    return value


def _identifier(value: object, label: str) -> str:
    result = _text(value, label)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ProviderContractError(f"{label} is not a canonical identifier")
    return result


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise ProviderContractError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _canonical(value: object) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError) as error:
        raise ProviderContractError("provider records must be JSON serializable") from error


def _sorted_unique(values: tuple[str, ...], label: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ProviderContractError(f"{label} must be sorted and unique")


def _capabilities(values: tuple[ProviderCapability, ...]) -> None:
    if values != tuple(sorted(set(values), key=lambda item: item.value)):
        raise ProviderContractError("provider capabilities must be sorted and unique")


def _redact(value: str) -> str:
    return _SECRET.sub(r"\1<redacted>", value)


@dataclass(frozen=True, slots=True)
class ProviderModelIdentity:
    """Stable identity of the text model selected by a provider."""

    provider_id: str
    model_id: str
    model_revision: str
    model_digest: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.provider_id, "provider ID")
        _identifier(self.model_id, "model ID")
        _text(self.model_revision, "model revision")
        if self.model_digest is not None:
            _digest(self.model_digest, "model digest")

    def to_record(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_digest": self.model_digest,
        }


@dataclass(frozen=True, slots=True)
class ProviderLimits:
    """Hard provider limits advertised before any request is started."""

    max_input_tokens: int
    max_output_tokens: int
    max_context_tokens: int
    max_concurrent_requests: int = 1
    max_wall_seconds: float = 60.0
    max_stream_events: int = 4096

    def __post_init__(self) -> None:
        integer_values = (
            self.max_input_tokens,
            self.max_output_tokens,
            self.max_context_tokens,
            self.max_concurrent_requests,
            self.max_stream_events,
        )
        if any(isinstance(value, bool) or value <= 0 for value in integer_values):
            raise ProviderContractError("provider limits must be positive integers")
        if isinstance(self.max_wall_seconds, bool) or self.max_wall_seconds <= 0:
            raise ProviderContractError("provider maximum wall time must be positive")
        if self.max_input_tokens + self.max_output_tokens > self.max_context_tokens:
            raise ProviderContractError(
                "provider input and output limits cannot exceed the context limit"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_context_tokens": self.max_context_tokens,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_wall_seconds": self.max_wall_seconds,
            "max_stream_events": self.max_stream_events,
        }


@dataclass(frozen=True, slots=True)
class ProviderCapabilityCard:
    """Machine-readable provider capabilities, limits, and model identity."""

    identity: ProviderModelIdentity
    kind: ProviderKind
    provider_revision: str
    capabilities: tuple[ProviderCapability, ...]
    limits: ProviderLimits
    structured_output_schemas: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.provider_revision, "provider revision")
        if self.identity.provider_id == "none" and self.kind is not ProviderKind.NONE:
            raise ProviderContractError("the none provider must declare kind none")
        _capabilities(self.capabilities)
        _sorted_unique(self.structured_output_schemas, "structured output schemas")
        if (
            self.structured_output_schemas
            and ProviderCapability.STRUCTURED_OUTPUT not in self.capabilities
        ):
            raise ProviderContractError(
                "structured output schemas require the structured_output capability"
            )

    @property
    def provider_id(self) -> str:
        return self.identity.provider_id

    @property
    def card_id(self) -> str:
        payload = {
            "schema_version": TEXT_PROVIDER_SCHEMA_VERSION,
            "identity": self.identity.to_record(),
            "kind": self.kind.value,
            "provider_revision": self.provider_revision,
            "capabilities": [item.value for item in self.capabilities],
            "limits": self.limits.to_record(),
            "structured_output_schemas": list(self.structured_output_schemas),
        }
        return f"provider_card_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": TEXT_PROVIDER_SCHEMA_VERSION,
            "card_id": self.card_id,
            "identity": self.identity.to_record(),
            "kind": self.kind.value,
            "provider_revision": self.provider_revision,
            "capabilities": [item.value for item in self.capabilities],
            "limits": self.limits.to_record(),
            "structured_output_schemas": list(self.structured_output_schemas),
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


@dataclass(frozen=True, slots=True)
class ProviderBudget:
    """Per-request budget; it is a hard boundary, not provider advice."""

    max_wall_seconds: float = 60.0
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_response_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        if isinstance(self.max_wall_seconds, bool) or self.max_wall_seconds <= 0:
            raise ProviderContractError("request wall-time budget must be positive")
        for name, value in (
            ("max_input_tokens", self.max_input_tokens),
            ("max_output_tokens", self.max_output_tokens),
        ):
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise ProviderContractError(f"{name} must be positive when set")
        if isinstance(self.max_response_bytes, bool) or self.max_response_bytes <= 0:
            raise ProviderContractError("maximum response bytes must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "max_wall_seconds": self.max_wall_seconds,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_response_bytes": self.max_response_bytes,
        }


@dataclass(frozen=True, slots=True)
class ProviderProvenance:
    """Deterministic request/response lineage retained with every result."""

    provider_id: str
    provider_revision: str
    request_digest: str
    response_digest: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.provider_id, "provenance provider ID")
        _text(self.provider_revision, "provenance provider revision")
        _digest(self.request_digest, "provenance request digest")
        if self.response_digest is not None:
            _digest(self.response_digest, "provenance response digest")
        _sorted_unique(self.evidence_refs, "provenance evidence references")

    def to_record(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "provider_revision": self.provider_revision,
            "request_digest": self.request_digest,
            "response_digest": self.response_digest,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class InterpretIntentRequest:
    request_id: str
    original_request: str
    intent_schema_version: int = 1
    budget: ProviderBudget = ProviderBudget()

    operation: ProviderOperation = field(init=False, default=ProviderOperation.INTERPRET_INTENT)

    def __post_init__(self) -> None:
        _identifier(self.request_id, "provider request ID")
        _text(self.original_request, "original request")
        if isinstance(self.intent_schema_version, bool) or self.intent_schema_version <= 0:
            raise ProviderContractError("intent schema version must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": TEXT_PROVIDER_SCHEMA_VERSION,
            "request_id": self.request_id,
            "operation": self.operation.value,
            "original_request": self.original_request,
            "intent_schema_version": self.intent_schema_version,
            "budget": self.budget.to_record(),
        }


@dataclass(frozen=True, slots=True)
class ClarificationRequest:
    request_id: str
    intent: IntentRecord
    question: str | None = None
    budget: ProviderBudget = ProviderBudget()

    operation: ProviderOperation = field(init=False, default=ProviderOperation.CLARIFY_INTENT)

    def __post_init__(self) -> None:
        _identifier(self.request_id, "provider request ID")
        if self.question is not None:
            _text(self.question, "clarification question")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": TEXT_PROVIDER_SCHEMA_VERSION,
            "request_id": self.request_id,
            "operation": self.operation.value,
            "intent_id": self.intent.intent_id,
            "intent_schema_version": self.intent.schema_version,
            "question": self.question,
            "budget": self.budget.to_record(),
        }


@dataclass(frozen=True, slots=True)
class ExplanationRequest:
    request_id: str
    evidence_records: tuple[Mapping[str, JSONValue], ...]
    question: str | None = None
    budget: ProviderBudget = ProviderBudget()

    operation: ProviderOperation = field(init=False, default=ProviderOperation.EXPLAIN_EVIDENCE)

    def __post_init__(self) -> None:
        _identifier(self.request_id, "provider request ID")
        if not self.evidence_records:
            raise ProviderContractError("explanation requires at least one evidence record")
        for record in self.evidence_records:
            if not isinstance(record, Mapping):
                raise ProviderContractError("evidence records must be JSON objects")
            _canonical(dict(record))
        if self.question is not None:
            _text(self.question, "explanation question")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": TEXT_PROVIDER_SCHEMA_VERSION,
            "request_id": self.request_id,
            "operation": self.operation.value,
            "evidence_records": [dict(record) for record in self.evidence_records],
            "question": self.question,
            "budget": self.budget.to_record(),
        }


type ProviderRequest = InterpretIntentRequest | ClarificationRequest | ExplanationRequest


@dataclass(frozen=True, slots=True)
class IntentProviderOutput:
    intent: IntentRecord

    operation: ProviderOperation = field(init=False, default=ProviderOperation.INTERPRET_INTENT)

    def to_record(self) -> dict[str, object]:
        return {"operation": self.operation.value, "intent": self.intent.to_record()}


@dataclass(frozen=True, slots=True)
class ClarificationProviderOutput:
    questions: tuple[str, ...]

    operation: ProviderOperation = field(init=False, default=ProviderOperation.CLARIFY_INTENT)

    def __post_init__(self) -> None:
        if not self.questions or any(not value.strip() for value in self.questions):
            raise ProviderContractError("clarification output requires non-empty questions")

    def to_record(self) -> dict[str, object]:
        return {"operation": self.operation.value, "questions": list(self.questions)}


@dataclass(frozen=True, slots=True)
class ExplanationProviderOutput:
    text: str
    evidence_refs: tuple[str, ...] = ()

    operation: ProviderOperation = field(init=False, default=ProviderOperation.EXPLAIN_EVIDENCE)

    def __post_init__(self) -> None:
        _text(self.text, "explanation text")
        _sorted_unique(self.evidence_refs, "explanation evidence references")

    def to_record(self) -> dict[str, object]:
        return {
            "operation": self.operation.value,
            "text": self.text,
            "evidence_refs": list(self.evidence_refs),
        }


type ProviderOutput = IntentProviderOutput | ClarificationProviderOutput | ExplanationProviderOutput


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """Safe failure record; diagnostics are redacted at serialization time."""

    code: ProviderFailureCode
    operation: ProviderOperation
    detail: str
    request_id: str
    retryable: bool = False
    redacted_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.detail, "provider failure detail")
        _identifier(self.request_id, "provider failure request ID")
        _sorted_unique(self.redacted_fields, "provider failure redacted fields")

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "operation": self.operation.value,
            "detail": _redact(self.detail),
            "request_id": self.request_id,
            "retryable": self.retryable,
            "redacted_fields": list(self.redacted_fields),
        }


@dataclass(frozen=True, slots=True)
class ProviderResult:
    request_id: str
    operation: ProviderOperation
    provider: ProviderModelIdentity
    outcome: ProviderOutcome
    provenance: ProviderProvenance
    output: ProviderOutput | None = None
    failure: ProviderFailure | None = None
    unsupported_capabilities: tuple[ProviderCapability, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.request_id, "provider result request ID")
        if self.provenance.provider_id != self.provider.provider_id:
            raise ProviderContractError("result provenance provider does not match model identity")
        _capabilities(self.unsupported_capabilities)
        if self.outcome is ProviderOutcome.SUPPORTED:
            if self.output is None or self.failure is not None:
                raise ProviderContractError("supported result requires output and no failure")
            if self.output.operation is not self.operation:
                raise ProviderContractError("provider output operation does not match result")
        elif self.output is not None:
            raise ProviderContractError("non-supported result cannot contain provider output")
        if self.outcome in {
            ProviderOutcome.FAILED,
            ProviderOutcome.TIMEOUT,
            ProviderOutcome.CANCELLED,
            ProviderOutcome.MALFORMED_OUTPUT,
        } and self.failure is None:
            raise ProviderContractError("failed provider result requires a failure record")
        if self.failure is not None and self.failure.request_id != self.request_id:
            raise ProviderContractError("provider failure request ID does not match result")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": TEXT_PROVIDER_SCHEMA_VERSION,
            "request_id": self.request_id,
            "operation": self.operation.value,
            "provider": self.provider.to_record(),
            "outcome": self.outcome.value,
            "provenance": self.provenance.to_record(),
            "output": None if self.output is None else self.output.to_record(),
            "failure": None if self.failure is None else self.failure.to_record(),
            "unsupported_capabilities": [item.value for item in self.unsupported_capabilities],
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


@dataclass(frozen=True, slots=True)
class ProviderStreamEvent:
    """A bounded, text-only stream event; it cannot carry an execution command."""

    request_id: str
    sequence: int
    text_delta: str = ""
    final_output: ProviderOutput | None = None
    done: bool = False

    def __post_init__(self) -> None:
        _identifier(self.request_id, "stream request ID")
        if isinstance(self.sequence, bool) or self.sequence < 0:
            raise ProviderContractError("stream sequence must be non-negative")
        if not self.text_delta and self.final_output is None:
            raise ProviderContractError("stream event must contain text or a final output")
        if self.final_output is not None and not self.done:
            raise ProviderContractError("final output requires a done stream event")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": TEXT_PROVIDER_SCHEMA_VERSION,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "text_delta": self.text_delta,
            "final_output": None if self.final_output is None else self.final_output.to_record(),
            "done": self.done,
        }


class CancellationToken:
    """Cooperative cancellation passed to provider calls and streams."""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ProviderCancelledError("provider request was cancelled")


class ProviderCancelledError(RuntimeError):
    """Raised by a provider implementation that observes cancellation."""


@runtime_checkable
class TextModelProvider(Protocol):
    """Narrow provider boundary with no optimization or tool execution authority."""

    @property
    def identity(self) -> ProviderModelIdentity: ...

    @property
    def capability_card(self) -> ProviderCapabilityCard: ...

    def start(self) -> None: ...

    def close(self) -> None: ...

    def call(
        self,
        request: ProviderRequest,
        *,
        cancellation: CancellationToken,
    ) -> ProviderResult: ...

    def stream(
        self,
        request: ProviderRequest,
        *,
        cancellation: CancellationToken,
    ) -> Iterator[ProviderStreamEvent]: ...

    def cancel(self, request_id: str) -> bool: ...


def request_digest(request: ProviderRequest) -> str:
    """Return the deterministic digest used by provider provenance."""

    return f"sha256:{hashlib.sha256(_canonical(request.to_record()).encode()).hexdigest()}"


def _result_provenance(
    provider: TextModelProvider,
    request: ProviderRequest,
    response: object | None = None,
) -> ProviderProvenance:
    response_digest = None if response is None else _sha256_json(response)
    return ProviderProvenance(
        provider.identity.provider_id,
        provider.capability_card.provider_revision,
        request_digest(request),
        response_digest,
    )


def _sha256_json(value: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical(value).encode()).hexdigest()}"


def _failure_result(
    provider: TextModelProvider,
    request: ProviderRequest,
    outcome: ProviderOutcome,
    code: ProviderFailureCode,
    detail: str,
    *,
    retryable: bool = False,
    unsupported: tuple[ProviderCapability, ...] = (),
) -> ProviderResult:
    failure = ProviderFailure(code, request.operation, detail, request.request_id, retryable)
    return ProviderResult(
        request.request_id,
        request.operation,
        provider.identity,
        outcome,
        _result_provenance(provider, request),
        failure=failure,
        unsupported_capabilities=unsupported,
    )


def _required_capabilities(request: ProviderRequest) -> tuple[ProviderCapability, ...]:
    operation_capability = ProviderCapability(request.operation.value)
    required = [operation_capability]
    if isinstance(request, InterpretIntentRequest):
        required.append(ProviderCapability.STRUCTURED_OUTPUT)
    return tuple(sorted(set(required), key=lambda item: item.value))


def invoke_provider(
    provider: TextModelProvider,
    request: ProviderRequest,
    *,
    cancellation: CancellationToken | None = None,
) -> ProviderResult:
    """Invoke a provider with capability, cancellation, and wall-time gates.

    The worker is daemonized because a timed-out provider must not hold the
    deterministic control-plane caller open.  Implementations should also
    release their own resources when the cancellation token is observed.
    """

    token = cancellation or CancellationToken()
    missing = tuple(
        capability
        for capability in _required_capabilities(request)
        if capability not in provider.capability_card.capabilities
    )
    if missing:
        return _failure_result(
            provider,
            request,
            ProviderOutcome.UNSUPPORTED,
            ProviderFailureCode.PROTOCOL,
            "provider does not advertise the required capability",
            unsupported=missing,
        )
    if token.cancelled:
        return _failure_result(
            provider,
            request,
            ProviderOutcome.CANCELLED,
            ProviderFailureCode.CANCELLED,
            "provider request was cancelled before start",
        )
    if request.budget.max_wall_seconds > provider.capability_card.limits.max_wall_seconds:
        return _failure_result(
            provider,
            request,
            ProviderOutcome.UNSUPPORTED,
            ProviderFailureCode.LIMIT_EXCEEDED,
            "request budget exceeds the provider wall-time limit",
        )

    result: list[ProviderResult] = []
    error: list[BaseException] = []

    def run() -> None:
        try:
            result.append(provider.call(request, cancellation=token))
        except BaseException as exc:  # failures become explicit boundary data
            error.append(exc)

    worker = threading.Thread(
        target=run, name=f"modelsurgeon-provider-{request.request_id}", daemon=True
    )
    worker.start()
    worker.join(request.budget.max_wall_seconds)
    if worker.is_alive():
        token.cancel()
        provider.cancel(request.request_id)
        return _failure_result(
            provider,
            request,
            ProviderOutcome.TIMEOUT,
            ProviderFailureCode.TIMEOUT,
            "provider exceeded the request wall-time budget",
            retryable=True,
        )
    if error:
        exc = error[0]
        if isinstance(exc, ProviderCancelledError):
            return _failure_result(
                provider,
                request,
                ProviderOutcome.CANCELLED,
                ProviderFailureCode.CANCELLED,
                "provider request was cancelled",
            )
        return _failure_result(
            provider,
            request,
            ProviderOutcome.FAILED,
            ProviderFailureCode.INTERNAL,
            f"provider raised {type(exc).__name__}",
        )
    if not result:
        return _failure_result(
            provider,
            request,
            ProviderOutcome.UNKNOWN,
            ProviderFailureCode.PROTOCOL,
            "provider returned no result",
        )
    returned = result[0]
    if returned.request_id != request.request_id or returned.operation is not request.operation:
        return _failure_result(
            provider,
            request,
            ProviderOutcome.MALFORMED_OUTPUT,
            ProviderFailureCode.PROTOCOL,
            "provider result identity does not match the request",
        )
    return returned


def decode_provider_output(request: ProviderRequest, payload: object) -> ProviderOutput:
    """Decode untrusted structured output into one of the typed output records."""

    if not isinstance(payload, Mapping):
        raise ProviderContractError("provider output must be a JSON object")
    operation = payload.get("operation")
    if operation != request.operation.value:
        raise ProviderContractError("provider output operation does not match the request")
    if request.operation is ProviderOperation.INTERPRET_INTENT:
        if set(payload) != {"operation", "intent"}:
            raise ProviderContractError("interpretation output has unknown fields")
        try:
            encoded = _canonical(payload["intent"])
            return IntentProviderOutput(IntentRecord.from_json(encoded))
        except (IntentRecordError, ProviderContractError) as error:
            raise ProviderContractError("provider returned an invalid intent record") from error
    if request.operation is ProviderOperation.CLARIFY_INTENT:
        if set(payload) != {"operation", "questions"}:
            raise ProviderContractError("clarification output has unknown fields")
        questions = payload["questions"]
        if not isinstance(questions, list) or any(
            not isinstance(value, str) for value in questions
        ):
            raise ProviderContractError("clarification questions must be strings")
        return ClarificationProviderOutput(tuple(questions))
    if set(payload) != {"operation", "text", "evidence_refs"}:
        raise ProviderContractError("explanation output has unknown fields")
    text = payload["text"]
    refs = payload["evidence_refs"]
    if not isinstance(text, str) or not isinstance(refs, list) or any(
        not isinstance(value, str) for value in refs
    ):
        raise ProviderContractError("explanation output has invalid fields")
    return ExplanationProviderOutput(text, tuple(refs))


def result_from_raw_output(
    provider: TextModelProvider,
    request: ProviderRequest,
    payload: object,
) -> ProviderResult:
    """Turn raw provider output into a supported result or retained failure."""

    try:
        output = decode_provider_output(request, payload)
    except ProviderContractError as error:
        return _failure_result(
            provider,
            request,
            ProviderOutcome.MALFORMED_OUTPUT,
            ProviderFailureCode.MALFORMED_OUTPUT,
            str(error),
        )
    return ProviderResult(
        request.request_id,
        request.operation,
        provider.identity,
        ProviderOutcome.SUPPORTED,
        _result_provenance(provider, request, payload),
        output=output,
    )


class NullTextModelProvider:
    """Explicit no-LLM provider; direct ModelSurgeon APIs remain available."""

    identity = ProviderModelIdentity("none", "none", "none")
    capability_card = ProviderCapabilityCard(
        identity,
        ProviderKind.NONE,
        "none",
        (),
        ProviderLimits(1, 1, 2),
    )

    def start(self) -> None:
        return None

    def close(self) -> None:
        return None

    def call(
        self,
        request: ProviderRequest,
        *,
        cancellation: CancellationToken,
    ) -> ProviderResult:
        return _failure_result(
            self,
            request,
            ProviderOutcome.UNSUPPORTED,
            ProviderFailureCode.UNAVAILABLE,
            "no text-model provider is configured",
            unsupported=_required_capabilities(request),
        )

    def stream(
        self,
        request: ProviderRequest,
        *,
        cancellation: CancellationToken,
    ) -> Iterator[ProviderStreamEvent]:
        if False:
            yield ProviderStreamEvent(request.request_id, 0, "")
        return

    def cancel(self, request_id: str) -> bool:
        return False


__all__ = [
    "TEXT_PROVIDER_SCHEMA_VERSION",
    "CancellationToken",
    "ClarificationProviderOutput",
    "ClarificationRequest",
    "ExplanationProviderOutput",
    "ExplanationRequest",
    "IntentProviderOutput",
    "InterpretIntentRequest",
    "NullTextModelProvider",
    "ProviderBudget",
    "ProviderCancelledError",
    "ProviderCapability",
    "ProviderCapabilityCard",
    "ProviderContractError",
    "ProviderFailure",
    "ProviderFailureCode",
    "ProviderKind",
    "ProviderLimits",
    "ProviderModelIdentity",
    "ProviderOperation",
    "ProviderOutcome",
    "ProviderOutput",
    "ProviderProvenance",
    "ProviderRequest",
    "ProviderResult",
    "ProviderStreamEvent",
    "TextModelProvider",
    "decode_provider_output",
    "invoke_provider",
    "request_digest",
    "result_from_raw_output",
]

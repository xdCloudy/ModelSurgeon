"""Fail-closed adapters for compatible and hosted text-model endpoints.

The adapters speak a deliberately small, OpenAI-chat-completions-shaped
protocol.  The wire protocol is not the provider contract: responses are
decoded through :mod:`modelsurgeon.conversation.provider` before they become
provider results.  Endpoint credentials are resolved just before transport
and never enter a canonical record.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC
from email.utils import parsedate_to_datetime
from enum import StrEnum
from time import monotonic
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from modelsurgeon.conversation.isolation import redact_secret_text
from modelsurgeon.conversation.provider import (
    CancellationToken,
    ClarificationRequest,
    InterpretIntentRequest,
    ProviderCapability,
    ProviderCapabilityCard,
    ProviderContractError,
    ProviderFailureCode,
    ProviderKind,
    ProviderLimits,
    ProviderModelIdentity,
    ProviderOperation,
    ProviderOutcome,
    ProviderRequest,
    ProviderResult,
    ProviderStreamEvent,
    _failure_result,
    result_from_raw_output,
)
from modelsurgeon.experiments.identity import canonical_identity_json

ENDPOINT_ADAPTER_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_HEADER_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,127}$")
_RETRY_STATUSES = (408, 429, 500, 502, 503, 504)
_PROVIDER_OUTPUT_SCHEMA = "modelsurgeon.provider_output.v1"


class EndpointAdapterError(ProviderContractError):
    """Raised only for invalid local endpoint configuration."""


class CapabilityProbeOutcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    MALFORMED_OUTPUT = "malformed_output"


class AuthScheme(StrEnum):
    BEARER = "bearer"
    RAW = "raw"


@dataclass(frozen=True, slots=True)
class AuthReference:
    """A non-secret reference resolved by the caller's credential store."""

    credential_ref: str
    header_name: str = "authorization"
    scheme: AuthScheme = AuthScheme.BEARER

    def __post_init__(self) -> None:
        if not isinstance(self.credential_ref, str) or not _IDENTIFIER.fullmatch(
            self.credential_ref
        ):
            raise EndpointAdapterError("credential reference must be a canonical identifier")
        if not isinstance(self.header_name, str) or not _HEADER_NAME.fullmatch(
            self.header_name.lower()
        ):
            raise EndpointAdapterError("auth header name is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "credential_ref": self.credential_ref,
            "header_name": self.header_name.lower(),
            "scheme": self.scheme.value,
        }

    def header_value(self, secret: str) -> str:
        if not isinstance(secret, str) or not secret or "\r" in secret or "\n" in secret:
            raise EndpointAdapterError("resolved credential is absent or invalid")
        if len(secret) > 4096:
            raise EndpointAdapterError("resolved credential exceeds the bounded size")
        return secret if self.scheme is AuthScheme.RAW else f"Bearer {secret}"


class SecretResolver(Protocol):
    """Resolve a credential reference without exposing the credential itself."""

    def __call__(self, credential_ref: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """A bounded retry budget for transport and provider overload responses."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 5.0
    max_retry_after_seconds: float = 10.0
    retry_statuses: tuple[int, ...] = _RETRY_STATUSES

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not 1 <= self.max_attempts <= 5:
            raise EndpointAdapterError("retry attempts must be between one and five")
        for name, value in (
            ("base retry delay", self.base_delay_seconds),
            ("maximum retry delay", self.max_delay_seconds),
            ("maximum Retry-After", self.max_retry_after_seconds),
        ):
            if isinstance(value, bool) or value < 0:
                raise EndpointAdapterError(f"{name} must be non-negative")
        if self.base_delay_seconds > self.max_delay_seconds:
            raise EndpointAdapterError("base retry delay cannot exceed the maximum")
        if self.retry_statuses != tuple(sorted(set(self.retry_statuses))):
            raise EndpointAdapterError("retry statuses must be sorted and unique")
        if any(
            isinstance(status, bool) or not 100 <= status <= 599
            for status in self.retry_statuses
        ):
            raise EndpointAdapterError("retry statuses must be HTTP status codes")


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    """Secret-free, bounded configuration for one remote model endpoint."""

    endpoint_id: str
    base_url: str
    provider_id: str
    model_id: str
    model_revision: str
    kind: ProviderKind
    auth: AuthReference | None = None
    probe_path: str = "/models/{model_id}"
    completion_path: str = "/chat/completions"
    limits: ProviderLimits = field(
        default_factory=lambda: ProviderLimits(6144, 2048, 8192, max_wall_seconds=60.0)
    )
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    max_request_bytes: int = 1 << 20

    def __post_init__(self) -> None:
        for value, label in (
            (self.endpoint_id, "endpoint ID"),
            (self.provider_id, "provider ID"),
            (self.model_id, "model ID"),
        ):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise EndpointAdapterError(f"{label} must be a canonical identifier")
        if not isinstance(self.model_revision, str) or not self.model_revision.strip():
            raise EndpointAdapterError("model revision must be non-empty text")
        if self.kind not in {ProviderKind.COMPATIBLE_ENDPOINT, ProviderKind.HOSTED}:
            raise EndpointAdapterError("endpoint adapters require a compatible or hosted kind")
        parts = urlsplit(self.base_url)
        if parts.scheme != "https" or not parts.netloc or parts.username or parts.password:
            raise EndpointAdapterError("endpoint base URL must be an HTTPS URL without credentials")
        if parts.query or parts.fragment:
            raise EndpointAdapterError("endpoint base URL cannot contain query or fragment data")
        if not self.base_url.rstrip("/"):
            raise EndpointAdapterError("endpoint base URL must be non-empty")
        for path_name, path in (
            ("probe path", self.probe_path),
            ("completion path", self.completion_path),
        ):
            if not path.startswith("/") or "?" in path or "#" in path or ".." in path:
                raise EndpointAdapterError(f"{path_name} must be a relative absolute path")
        if isinstance(self.max_request_bytes, bool) or self.max_request_bytes <= 0:
            raise EndpointAdapterError("maximum request bytes must be positive")

    def to_record(self) -> dict[str, object]:
        """Return canonical configuration metadata with no secret material."""

        return {
            "schema_version": ENDPOINT_ADAPTER_SCHEMA_VERSION,
            "endpoint_id": self.endpoint_id,
            "base_url": self.base_url.rstrip("/"),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "kind": self.kind.value,
            "auth": None if self.auth is None else self.auth.to_record(),
            "probe_path": self.probe_path,
            "completion_path": self.completion_path,
            "limits": self.limits.to_record(),
            "retry_policy": {
                "max_attempts": self.retry_policy.max_attempts,
                "base_delay_seconds": self.retry_policy.base_delay_seconds,
                "max_delay_seconds": self.retry_policy.max_delay_seconds,
                "max_retry_after_seconds": self.retry_policy.max_retry_after_seconds,
                "retry_statuses": list(self.retry_policy.retry_statuses),
            },
            "max_request_bytes": self.max_request_bytes,
        }

    def canonical_json(self) -> str:
        return canonical_identity_json(self.to_record())


@dataclass(frozen=True, slots=True)
class EndpointHttpRequest:
    method: str
    url: str
    headers: tuple[tuple[str, str], ...]
    body: bytes
    timeout_seconds: float
    max_response_bytes: int


@dataclass(frozen=True, slots=True)
class EndpointHttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class EndpointTransport(Protocol):
    def request(self, request: EndpointHttpRequest) -> EndpointHttpResponse: ...


class _TransportFailure(RuntimeError):
    pass


class _ResponseTooLarge(_TransportFailure):
    pass


class UrllibEndpointTransport:
    """Small standard-library transport with a hard response-size ceiling."""

    def request(self, request: EndpointHttpRequest) -> EndpointHttpResponse:
        url_request = UrlRequest(
            request.url,
            data=request.body if request.method != "GET" else None,
            headers=dict(request.headers),
            method=request.method,
        )
        try:
            with urlopen(url_request, timeout=request.timeout_seconds) as response:
                body = response.read(request.max_response_bytes + 1)
                if len(body) > request.max_response_bytes:
                    raise _ResponseTooLarge(
                        "endpoint response exceeded the configured limit"
                    )
                return EndpointHttpResponse(
                    response.status,
                    {key.lower(): value for key, value in response.headers.items()},
                    body,
                )
        except HTTPError as error:
            body = error.read(request.max_response_bytes + 1)
            if len(body) > request.max_response_bytes:
                raise _ResponseTooLarge(
                    "endpoint response exceeded the configured limit"
                ) from error
            return EndpointHttpResponse(
                error.code,
                {key.lower(): value for key, value in error.headers.items()},
                body,
            )
        except (OSError, URLError, TimeoutError) as error:
            raise _TransportFailure("endpoint transport failed") from error


@dataclass(frozen=True, slots=True)
class CapabilityProbeResult:
    endpoint_id: str
    outcome: CapabilityProbeOutcome
    card: ProviderCapabilityCard | None = None
    detail: str | None = None
    retryable: bool = False

    def __post_init__(self) -> None:
        if self.card is None and self.outcome is CapabilityProbeOutcome.SUPPORTED:
            raise EndpointAdapterError("supported capability probe requires a card")
        if self.detail is None and self.outcome is not CapabilityProbeOutcome.SUPPORTED:
            raise EndpointAdapterError("failed capability probe requires a detail")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": ENDPOINT_ADAPTER_SCHEMA_VERSION,
            "endpoint_id": self.endpoint_id,
            "outcome": self.outcome.value,
            "card": None if self.card is None else self.card.to_record(),
            "detail": None
            if self.detail is None
            else redact_secret_text(self.detail),
            "retryable": self.retryable,
        }

    def canonical_json(self) -> str:
        return canonical_identity_json(self.to_record())


def _json_object(body: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EndpointAdapterError(f"{label} was not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise EndpointAdapterError(f"{label} must be a JSON object")
    return value


def _json_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EndpointAdapterError("endpoint request was not JSON serializable") from error


def _bounded_limits(advertised: ProviderLimits, configured: ProviderLimits) -> ProviderLimits:
    return ProviderLimits(
        min(advertised.max_input_tokens, configured.max_input_tokens),
        min(advertised.max_output_tokens, configured.max_output_tokens),
        min(advertised.max_context_tokens, configured.max_context_tokens),
        min(advertised.max_concurrent_requests, configured.max_concurrent_requests),
        min(advertised.max_wall_seconds, configured.max_wall_seconds),
        min(advertised.max_stream_events, configured.max_stream_events),
        min(advertised.max_memory_bytes, configured.max_memory_bytes),
    )


def _provider_limits(value: object) -> ProviderLimits:
    if not isinstance(value, Mapping):
        raise EndpointAdapterError("capability limits must be an object")
    required = {
        "max_input_tokens",
        "max_output_tokens",
        "max_context_tokens",
        "max_concurrent_requests",
        "max_wall_seconds",
        "max_stream_events",
        "max_memory_bytes",
    }
    if set(value) != required:
        raise EndpointAdapterError("capability limits have unknown or missing fields")
    try:
        return ProviderLimits(
            value["max_input_tokens"],
            value["max_output_tokens"],
            value["max_context_tokens"],
            value["max_concurrent_requests"],
            value["max_wall_seconds"],
            value["max_stream_events"],
            value["max_memory_bytes"],
        )
    except (TypeError, ValueError, ProviderContractError) as error:
        raise EndpointAdapterError("capability limits are invalid") from error


def _capability_card(config: EndpointConfig, value: object) -> ProviderCapabilityCard:
    if not isinstance(value, Mapping):
        raise EndpointAdapterError("capability probe metadata must be an object")
    required = {
        "protocol_version",
        "provider_id",
        "model_id",
        "model_revision",
        "capabilities",
        "limits",
        "structured_output_schemas",
    }
    optional = {"model_digest"}
    if set(value) - required - optional or not required.issubset(value):
        raise EndpointAdapterError("capability probe metadata has unknown or missing fields")
    if value["protocol_version"] != ENDPOINT_ADAPTER_SCHEMA_VERSION:
        raise EndpointAdapterError("unsupported endpoint capability protocol version")
    if value["provider_id"] != config.provider_id or value["model_id"] != config.model_id:
        raise EndpointAdapterError("capability probe identity contradicts endpoint configuration")
    if value["model_revision"] != config.model_revision:
        raise EndpointAdapterError("capability probe model revision contradicts configuration")
    capabilities_value = value["capabilities"]
    schemas_value = value["structured_output_schemas"]
    if not isinstance(capabilities_value, list) or not isinstance(schemas_value, list):
        raise EndpointAdapterError("capabilities and structured schemas must be arrays")
    try:
        capabilities = tuple(ProviderCapability(item) for item in capabilities_value)
    except (TypeError, ValueError) as error:
        raise EndpointAdapterError("capability probe contains an unknown capability") from error
    if capabilities != tuple(sorted(set(capabilities), key=lambda item: item.value)):
        raise EndpointAdapterError("capability probe capabilities must be sorted and unique")
    if any(not isinstance(item, str) or not item.strip() for item in schemas_value):
        raise EndpointAdapterError("structured output schema IDs must be non-empty text")
    schemas = tuple(schemas_value)
    if schemas != tuple(sorted(set(schemas))):
        raise EndpointAdapterError("structured output schema IDs must be sorted and unique")
    try:
        identity = ProviderModelIdentity(
            config.provider_id,
            config.model_id,
            config.model_revision,
            value.get("model_digest"),
        )
    except ProviderContractError as error:
        raise EndpointAdapterError("capability probe model digest is invalid") from error
    return ProviderCapabilityCard(
        identity,
        config.kind,
        f"endpoint-capability-v{ENDPOINT_ADAPTER_SCHEMA_VERSION}",
        capabilities,
        _bounded_limits(_provider_limits(value["limits"]), config.limits),
        schemas,
    )


def _retry_after(headers: Mapping[str, str], *, now: float) -> float | None:
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        try:
            date = parsedate_to_datetime(raw)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            seconds = date.timestamp() - now
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds)


def _schema_for(operation: ProviderOperation) -> dict[str, object]:
    if operation is ProviderOperation.INTERPRET_INTENT:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["operation", "intent"],
            "properties": {"operation": {"const": operation.value}, "intent": {"type": "object"}},
        }
    if operation is ProviderOperation.CLARIFY_INTENT:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["operation", "questions"],
            "properties": {
                "operation": {"const": operation.value},
                "questions": {"type": "array", "items": {"type": "string"}},
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["operation", "text", "evidence_refs"],
        "properties": {
            "operation": {"const": operation.value},
            "text": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
        },
    }


def _prompt(request: ProviderRequest) -> tuple[str, str]:
    system = (
        "You are a bounded ModelSurgeon text-model adapter. Return exactly one JSON object "
        "matching the requested operation. Never request or perform model execution, tools, "
        "filesystem access, or optimization mutations."
    )
    if isinstance(request, InterpretIntentRequest):
        return system, request.original_request
    value: dict[str, object]
    if isinstance(request, ClarificationRequest):
        value = {"intent": request.intent.to_record(), "question": request.question}
    else:
        value = {
            "evidence_records": [dict(item) for item in request.evidence_records],
            "question": request.question,
        }
    return system, canonical_identity_json(value)


def _response_payload(response: Mapping[str, object], model_id: str) -> object:
    allowed = {"id", "object", "created", "model", "choices", "usage", "system_fingerprint"}
    if set(response) - allowed or "choices" not in response:
        raise EndpointAdapterError("endpoint response has unknown or missing fields")
    if response.get("model") != model_id:
        raise EndpointAdapterError("endpoint response model contradicts configuration")
    choices = response["choices"]
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise EndpointAdapterError("endpoint response must contain exactly one choice")
    choice = choices[0]
    if set(choice) - {"index", "message", "finish_reason"} or "message" not in choice:
        raise EndpointAdapterError("endpoint choice has unknown or missing fields")
    if choice.get("finish_reason") != "stop":
        raise EndpointAdapterError("endpoint response was truncated or refused")
    message = choice["message"]
    if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
        raise EndpointAdapterError("endpoint message must contain only role and content")
    if message["role"] != "assistant" or not isinstance(message["content"], str):
        raise EndpointAdapterError("endpoint message is not an assistant text response")
    try:
        payload = json.loads(message["content"])
    except json.JSONDecodeError as error:
        raise EndpointAdapterError("endpoint message content was not JSON") from error
    return payload


class _RemoteEndpointProvider:
    """Common implementation for the two explicitly supported endpoint kinds."""

    def __init__(
        self,
        config: EndpointConfig,
        *,
        auth_resolver: SecretResolver | None = None,
        transport: EndpointTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.config = config
        self._auth_resolver = auth_resolver
        self._transport = transport or UrllibEndpointTransport()
        self._sleeper = sleeper
        self._clock = clock
        self._probe: CapabilityProbeResult | None = None
        self._active: dict[str, CancellationToken] = {}
        self._active_lock = threading.Lock()
        self._identity = ProviderModelIdentity(
            config.provider_id, config.model_id, config.model_revision
        )
        self._empty_card = ProviderCapabilityCard(
            self._identity,
            config.kind,
            "unprobed",
            (),
            config.limits,
        )
        self._card = self._empty_card

    @property
    def identity(self) -> ProviderModelIdentity:
        return self._identity

    @property
    def capability_card(self) -> ProviderCapabilityCard:
        return self._card

    @property
    def last_probe(self) -> CapabilityProbeResult | None:
        return self._probe

    def start(self) -> None:
        result = self.probe()
        self._probe = result
        self._card = result.card or self._empty_card

    def close(self) -> None:
        with self._active_lock:
            for token in self._active.values():
                token.cancel()
            self._active.clear()

    def _resolve_headers(self, request_id: str) -> tuple[tuple[str, str], ...]:
        headers = [("accept", "application/json"), ("x-modelsurgeon-request-id", request_id)]
        if self.config.auth is not None:
            if self._auth_resolver is None:
                raise EndpointAdapterError("configured endpoint credential is unavailable")
            try:
                secret = self._auth_resolver(self.config.auth.credential_ref)
            except Exception as error:
                raise EndpointAdapterError(
                    "configured endpoint credential is unavailable"
                ) from error
            if secret is None:
                raise EndpointAdapterError("configured endpoint credential is unavailable")
            headers.append(
                (
                    self.config.auth.header_name.lower(),
                    self.config.auth.header_value(secret),
                )
            )
        return tuple(headers)

    def _url(self, path: str) -> str:
        encoded_model_id = quote(self.config.model_id, safe="")
        return f"{self.config.base_url.rstrip('/')}{path.format(model_id=encoded_model_id)}"

    def _request_with_retries(
        self,
        request: EndpointHttpRequest,
        *,
        deadline: float,
    ) -> tuple[EndpointHttpResponse | None, ProviderFailureCode | None, str, bool]:
        policy = self.config.retry_policy
        for attempt in range(policy.max_attempts):
            remaining = deadline - self._clock()
            if remaining <= 0:
                return (
                    None,
                    ProviderFailureCode.TIMEOUT,
                    "endpoint request exceeded its time budget",
                    True,
                )
            bounded_request = EndpointHttpRequest(
                request.method,
                request.url,
                request.headers,
                request.body,
                remaining,
                request.max_response_bytes,
            )
            try:
                response = self._transport.request(bounded_request)
            except _ResponseTooLarge:
                return (
                    None,
                    ProviderFailureCode.LIMIT_EXCEEDED,
                    "endpoint response exceeded its limit",
                    False,
                )
            except Exception:
                if attempt + 1 >= policy.max_attempts:
                    return (
                        None,
                        ProviderFailureCode.UNAVAILABLE,
                        "endpoint transport failed",
                        True,
                    )
                delay = min(policy.max_delay_seconds, policy.base_delay_seconds * (2**attempt))
                if delay and delay >= deadline - self._clock():
                    return (
                        None,
                        ProviderFailureCode.TIMEOUT,
                        "endpoint retry budget expired",
                        True,
                    )
                self._sleeper(delay)
                continue
            if 200 <= response.status < 300:
                return response, None, "", False
            if response.status in {401, 403}:
                return (
                    None,
                    ProviderFailureCode.AUTHENTICATION,
                    "endpoint authentication failed",
                    False,
                )
            if response.status == 429:
                if attempt + 1 >= policy.max_attempts:
                    return (
                        None,
                        ProviderFailureCode.RATE_LIMITED,
                        "endpoint rate limit persisted",
                        True,
                    )
                delay = _retry_after(response.headers, now=self._clock())
                if delay is None:
                    delay = min(policy.max_delay_seconds, policy.base_delay_seconds * (2**attempt))
                delay = min(delay, policy.max_retry_after_seconds, policy.max_delay_seconds)
            elif response.status in policy.retry_statuses:
                if attempt + 1 >= policy.max_attempts:
                    return (
                        None,
                        ProviderFailureCode.UNAVAILABLE,
                        "endpoint service unavailable",
                        True,
                    )
                delay = min(policy.max_delay_seconds, policy.base_delay_seconds * (2**attempt))
            else:
                return (
                    None,
                    ProviderFailureCode.PROTOCOL,
                    "endpoint returned an unsupported status",
                    False,
                )
            if delay >= deadline - self._clock():
                return (
                    None,
                    ProviderFailureCode.TIMEOUT,
                    "endpoint retry budget expired",
                    True,
                )
            self._sleeper(delay)
        return (
            None,
            ProviderFailureCode.UNAVAILABLE,
            "endpoint request failed",
            True,
        )

    def probe(self) -> CapabilityProbeResult:
        try:
            headers = self._resolve_headers("probe")
        except ProviderContractError as error:
            return CapabilityProbeResult(
                self.config.endpoint_id,
                CapabilityProbeOutcome.FAILED,
                detail=str(error),
            )
        request = EndpointHttpRequest(
            "GET",
            self._url(self.config.probe_path),
            headers,
            b"",
            self.config.limits.max_wall_seconds,
            self.config.max_request_bytes,
        )
        response, code, detail, retryable = self._request_with_retries(
            request, deadline=self._clock() + self.config.limits.max_wall_seconds
        )
        if response is None:
            outcome = (
                CapabilityProbeOutcome.UNSUPPORTED
                if code is ProviderFailureCode.PROTOCOL
                else CapabilityProbeOutcome.FAILED
            )
            return CapabilityProbeResult(
                self.config.endpoint_id, outcome, detail=detail, retryable=retryable
            )
        try:
            payload = _json_object(response.body, label="capability response")
            if set(payload) != {"object", "id", "modelsurgeon"} or payload["object"] != "model":
                raise EndpointAdapterError("capability response has an unsupported shape")
            if payload["id"] != self.config.model_id:
                raise EndpointAdapterError("capability response model contradicts configuration")
            card = _capability_card(self.config, payload["modelsurgeon"])
        except EndpointAdapterError as error:
            return CapabilityProbeResult(
                self.config.endpoint_id,
                CapabilityProbeOutcome.MALFORMED_OUTPUT,
                detail=str(error),
            )
        return CapabilityProbeResult(
            self.config.endpoint_id, CapabilityProbeOutcome.SUPPORTED, card=card
        )

    def _unsupported(
        self,
        request: ProviderRequest,
        detail: str,
        capabilities: tuple[ProviderCapability, ...] = (),
    ) -> ProviderResult:
        return _failure_result(
            self,
            request,
            ProviderOutcome.UNSUPPORTED,
            ProviderFailureCode.PROTOCOL,
            detail,
            unsupported=capabilities,
        )

    def call(self, request: ProviderRequest, *, cancellation: CancellationToken) -> ProviderResult:
        operation_capability = ProviderCapability(request.operation.value)
        if operation_capability not in self._card.capabilities:
            return self._unsupported(
                request,
                "endpoint capability was not probed or advertised",
                (operation_capability,),
            )
        if ProviderCapability.STRUCTURED_OUTPUT not in self._card.capabilities:
            return self._unsupported(
                request,
                "endpoint does not advertise structured output required by this adapter",
                (ProviderCapability.STRUCTURED_OUTPUT,),
            )
        try:
            headers = self._resolve_headers(request.request_id)
            system, user = _prompt(request)
            response_format: dict[str, object]
            if _PROVIDER_OUTPUT_SCHEMA in self._card.structured_output_schemas:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": _PROVIDER_OUTPUT_SCHEMA,
                        "strict": True,
                        "schema": _schema_for(request.operation),
                    },
                }
            else:
                response_format = {"type": "json_object"}
            body_value: dict[str, object] = {
                "model": self.config.model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "stream": False,
                "response_format": response_format,
            }
            body = _json_bytes(body_value)
            if len(body) > self.config.max_request_bytes:
                return _failure_result(
                    self,
                    request,
                    ProviderOutcome.UNSUPPORTED,
                    ProviderFailureCode.LIMIT_EXCEEDED,
                    "endpoint request exceeded its configured limit",
                )
            if request.budget.max_response_bytes > self.config.max_request_bytes:
                return _failure_result(
                    self,
                    request,
                    ProviderOutcome.UNSUPPORTED,
                    ProviderFailureCode.LIMIT_EXCEEDED,
                    "request response budget exceeds endpoint limit",
                )
        except EndpointAdapterError as error:
            return _failure_result(
                self,
                request,
                ProviderOutcome.FAILED,
                ProviderFailureCode.AUTHENTICATION
                if "credential" in str(error)
                else ProviderFailureCode.PROTOCOL,
                str(error),
            )
        with self._active_lock:
            self._active[request.request_id] = cancellation
        try:
            response, code, detail, retryable = self._request_with_retries(
                EndpointHttpRequest(
                    "POST",
                    self._url(self.config.completion_path),
                    (*headers, ("content-type", "application/json")),
                    body,
                    request.budget.max_wall_seconds,
                    min(request.budget.max_response_bytes, self.config.max_request_bytes),
                ),
                deadline=self._clock() + request.budget.max_wall_seconds,
            )
            if cancellation.cancelled:
                return _failure_result(
                    self,
                    request,
                    ProviderOutcome.CANCELLED,
                    ProviderFailureCode.CANCELLED,
                    "provider request was cancelled",
                )
            if response is None:
                outcome = (
                    ProviderOutcome.TIMEOUT
                    if code is ProviderFailureCode.TIMEOUT
                    else ProviderOutcome.FAILED
                )
                return _failure_result(
                    self,
                    request,
                    outcome,
                    code or ProviderFailureCode.INTERNAL,
                    detail,
                    retryable=retryable,
                )
            try:
                payload = _response_payload(
                    _json_object(response.body, label="endpoint response"),
                    self.config.model_id,
                )
            except EndpointAdapterError as error:
                return _failure_result(
                    self,
                    request,
                    ProviderOutcome.MALFORMED_OUTPUT,
                    ProviderFailureCode.MALFORMED_OUTPUT,
                    str(error),
                )
            return result_from_raw_output(self, request, payload)
        finally:
            with self._active_lock:
                self._active.pop(request.request_id, None)

    def stream(
        self, request: ProviderRequest, *, cancellation: CancellationToken
    ) -> Iterator[ProviderStreamEvent]:
        result = self.call(request, cancellation=cancellation)
        if result.output is not None:
            yield ProviderStreamEvent(request.request_id, 0, final_output=result.output, done=True)

    def cancel(self, request_id: str) -> bool:
        with self._active_lock:
            token = self._active.get(request_id)
        if token is None:
            return False
        token.cancel()
        return True


class CompatibleEndpointProvider(_RemoteEndpointProvider):
    """Adapter for a compatible endpoint implementing the bounded wire protocol."""


class HostedProviderAdapter(_RemoteEndpointProvider):
    """Adapter for a hosted provider that explicitly exposes the same protocol."""


CompatibleEndpointAdapter = CompatibleEndpointProvider
HostedProvider = HostedProviderAdapter


__all__ = [
    "ENDPOINT_ADAPTER_SCHEMA_VERSION",
    "AuthReference",
    "AuthScheme",
    "CapabilityProbeOutcome",
    "CapabilityProbeResult",
    "CompatibleEndpointAdapter",
    "CompatibleEndpointProvider",
    "EndpointAdapterError",
    "EndpointConfig",
    "EndpointHttpRequest",
    "EndpointHttpResponse",
    "EndpointTransport",
    "HostedProvider",
    "HostedProviderAdapter",
    "RetryPolicy",
    "SecretResolver",
    "UrllibEndpointTransport",
]

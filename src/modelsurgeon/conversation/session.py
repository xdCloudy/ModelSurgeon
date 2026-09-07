"""Bounded chat-session bootstrap around the conversational control plane.

This module deliberately stops at interpretation and objective validation. It
does not construct a tool dispatcher, optimizer runtime, mutation plan, or
artifact writer. A chat provider is therefore never an execution authority.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from modelsurgeon.adapters.family import ArchitectureEvidence, detect_model_family
from modelsurgeon.adapters.gguf import GGUFParseError, GGUFValueType, open_gguf
from modelsurgeon.provider_kind import ProviderKind

from .local_gguf import (
    LocalGGUFProvider,
    LocalGGUFProviderConfig,
    LocalGGUFProviderError,
    LocalGGUFUnsupportedError,
)
from .provider import (
    CancellationToken,
    IntentProviderOutput,
    InterpretIntentRequest,
    NullTextModelProvider,
    ProviderBudget,
    ProviderOutcome,
    ProviderResult,
    TextModelProvider,
    invoke_provider,
)

if TYPE_CHECKING:
    from modelsurgeon.search.intent_policy import IntentPolicyDecision

CHAT_SESSION_SCHEMA_VERSION: Literal[1] = 1
CHAT_TURN_SCHEMA_VERSION: Literal[1] = 1
DEFAULT_CHAT_MAX_TURNS = 8
_MAX_CHAT_TURNS = 64
_CHUNK_SIZE = 1024 * 1024


class ChatSessionError(ValueError):
    """Raised when a chat session cannot be safely bootstrapped."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    encoded = _canonical(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as error:
        raise ChatSessionError("model_unreadable", f"cannot read chat model: {path}") from error
    return f"sha256:{digest.hexdigest()}"


def _validate_local_model(path: Path) -> tuple[Path, str, str]:
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise ChatSessionError("model_missing", f"chat model file does not exist: {path}")
    if resolved.suffix.lower() != ".gguf":
        raise ChatSessionError(
            "unsupported_model_format",
            "the local chat provider supports only .gguf model files",
        )
    try:
        with open_gguf(resolved) as mapped:
            entry = mapped.container.metadata_entry("general.architecture")
            if entry is None or entry.value_type is not GGUFValueType.STRING:
                raise ChatSessionError(
                    "unsupported_model_metadata",
                    "GGUF chat models require string general.architecture metadata",
                )
            if not isinstance(entry.value, str) or not entry.value.strip():
                raise ChatSessionError(
                    "unsupported_model_metadata",
                    "GGUF chat model architecture metadata is empty",
                )
            try:
                family = detect_model_family(ArchitectureEvidence(gguf_architecture=entry.value))
            except ValueError as error:
                raise ChatSessionError("unsupported_architecture", str(error)) from error
    except GGUFParseError as error:
        raise ChatSessionError(
            "invalid_model", "GGUF chat model container or metadata validation failed"
        ) from error
    return resolved, _file_digest(resolved), family.family.value


def _installed_runtime_revision() -> str:
    try:
        return importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError as error:
        raise ChatSessionError(
            "provider_unavailable",
            "llama-cpp-python is not installed; install the local provider extra",
        ) from error


@dataclass(frozen=True, slots=True)
class ChatSessionBootstrap:
    """Immutable identity and limits established before interaction begins."""

    session_id: str
    provider: Mapping[str, object]
    capability_card: Mapping[str, object]
    model_path: str | None
    model_revision: str | None
    runtime_revision: str | None
    architecture: str | None
    max_turns: int
    schema_version: Literal[1] = CHAT_SESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHAT_SESSION_SCHEMA_VERSION:
            raise ChatSessionError("schema_version", "unsupported chat session schema version")
        if not self.session_id.startswith("chat_session_"):
            raise ChatSessionError("session_id", "invalid chat session identity")
        if not 0 < self.max_turns <= _MAX_CHAT_TURNS:
            raise ChatSessionError("max_turns", "chat turn budget is outside the supported range")

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "chat_session_bootstrap",
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "provider": dict(self.provider),
            "capability_card": dict(self.capability_card),
            "model_path": self.model_path,
            "model_revision": self.model_revision,
            "runtime_revision": self.runtime_revision,
            "architecture": self.architecture,
            "max_turns": self.max_turns,
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


@dataclass(frozen=True, slots=True)
class ChatTurnResult:
    """Retained result for one bounded interpretation attempt."""

    session_id: str
    turn: int
    request_id: str
    request: str
    provider_result: ProviderResult
    policy_decision: IntentPolicyDecision | None
    schema_version: Literal[1] = CHAT_TURN_SCHEMA_VERSION

    @property
    def outcome(self) -> str:
        if self.policy_decision is not None:
            return self.policy_decision.outcome.value
        return self.provider_result.outcome.value

    def to_record(self) -> dict[str, object]:
        intent: dict[str, object] | None = None
        if (
            self.provider_result.outcome is ProviderOutcome.SUPPORTED
            and isinstance(self.provider_result.output, IntentProviderOutput)
        ):
            intent = self.provider_result.output.intent.to_record()
        return {
            "record_type": "chat_turn",
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "turn": self.turn,
            "request_id": self.request_id,
            "request": self.request,
            "outcome": self.outcome,
            "provider_result": self.provider_result.to_record(),
            "intent": intent,
            "policy_decision": (
                None if self.policy_decision is None else self.policy_decision.to_record()
            ),
            "execution": "not_requested",
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


ProviderFactory = Callable[[LocalGGUFProviderConfig], TextModelProvider]


def _build_provider(
    model: Path,
    *,
    provider_kind: ProviderKind,
    model_revision: str | None,
    runtime_revision: str | None,
    max_input_tokens: int,
    max_output_tokens: int,
    max_wall_seconds: float,
    provider_factory: ProviderFactory | None,
) -> tuple[TextModelProvider, Path | None, str | None, str | None, str | None]:
    if provider_kind is ProviderKind.NONE:
        return NullTextModelProvider(), None, None, None, None
    if provider_kind is not ProviderKind.LOCAL:
        raise ChatSessionError(
            "provider_unsupported",
            f"provider kind {provider_kind.value!r} has no chat adapter in this release",
        )
    resolved, computed_revision, architecture = _validate_local_model(model)
    if model_revision is not None and model_revision != computed_revision:
        raise ChatSessionError(
            "model_revision_mismatch",
            "--model-revision does not match the local model SHA-256",
        )
    selected_runtime_revision = runtime_revision or _installed_runtime_revision()
    config = LocalGGUFProviderConfig(
        resolved,
        computed_revision,
        selected_runtime_revision,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_context_tokens=max_input_tokens + max_output_tokens,
        max_wall_seconds=max_wall_seconds,
    )
    provider = (
        LocalGGUFProvider(config) if provider_factory is None else provider_factory(config)
    )
    return provider, resolved, computed_revision, architecture, selected_runtime_revision


def bootstrap_chat_session(
    model: Path,
    *,
    provider_kind: ProviderKind = ProviderKind.LOCAL,
    model_revision: str | None = None,
    runtime_revision: str | None = None,
    max_turns: int = DEFAULT_CHAT_MAX_TURNS,
    max_input_tokens: int = 2048,
    max_output_tokens: int = 1024,
    max_wall_seconds: float = 60.0,
    provider_factory: ProviderFactory | None = None,
) -> ChatSession:
    """Validate, start, and return a bounded chat session.

    Provider startup is the only model execution performed here. It loads the
    selected text-model provider; it never loads or mutates a target model and
    it never invokes an optimization API.
    """

    if isinstance(max_turns, bool) or not 0 < max_turns <= _MAX_CHAT_TURNS:
        raise ChatSessionError("max_turns", f"max turns must be between 1 and {_MAX_CHAT_TURNS}")
    if isinstance(max_input_tokens, bool) or max_input_tokens <= 0:
        raise ChatSessionError("budget", "max input tokens must be positive")
    if isinstance(max_output_tokens, bool) or max_output_tokens <= 0:
        raise ChatSessionError("budget", "max output tokens must be positive")
    if isinstance(max_wall_seconds, bool) or max_wall_seconds <= 0:
        raise ChatSessionError("budget", "max wall seconds must be positive")
    provider, resolved, revision, architecture, selected_runtime_revision = _build_provider(
        model,
        provider_kind=provider_kind,
        model_revision=model_revision,
        runtime_revision=runtime_revision,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_wall_seconds=max_wall_seconds,
        provider_factory=provider_factory,
    )
    try:
        provider.start()
    except ChatSessionError:
        provider.close()
        raise
    except LocalGGUFUnsupportedError as error:
        provider.close()
        raise ChatSessionError("provider_unsupported", str(error)) from error
    except LocalGGUFProviderError as error:
        provider.close()
        raise ChatSessionError("provider_start_failed", str(error)) from error
    except Exception as error:
        provider.close()
        raise ChatSessionError(
            "provider_start_failed", "selected chat provider could not be started"
        ) from error

    identity = provider.identity.to_record()
    card = provider.capability_card.to_record()
    identity_record = {
        "provider": identity,
        "capability_card": card,
        "model_path": None if resolved is None else str(resolved),
        "model_revision": revision,
        "runtime_revision": selected_runtime_revision,
        "architecture": architecture,
        "max_turns": max_turns,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_tokens,
        "max_wall_seconds": max_wall_seconds,
    }
    session_id = "chat_session_" + _digest(identity_record)[len("sha256:") :]
    bootstrap = ChatSessionBootstrap(
        session_id,
        identity,
        card,
        None if resolved is None else str(resolved),
        revision,
        selected_runtime_revision,
        architecture,
        max_turns,
    )
    return ChatSession(
        bootstrap,
        provider,
        ProviderBudget(
            max_wall_seconds=max_wall_seconds,
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        ),
    )


class ChatSession:
    """A lifecycle-owned, interpretation-only interaction loop."""

    def __init__(
        self,
        bootstrap: ChatSessionBootstrap,
        provider: TextModelProvider,
        budget: ProviderBudget,
    ) -> None:
        self.bootstrap = bootstrap
        self.provider = provider
        self.budget = budget
        self._turn = 0
        self._closed = False
        self._cancellation = CancellationToken()

    @property
    def turns_used(self) -> int:
        return self._turn

    def interpret(
        self,
        request: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ChatTurnResult:
        if self._closed:
            raise ChatSessionError("session_closed", "chat session is closed")
        if not isinstance(request, str) or not request.strip():
            raise ChatSessionError("empty_request", "chat request must not be empty")
        if self._turn >= self.bootstrap.max_turns:
            raise ChatSessionError("turn_budget", "chat turn budget is exhausted")
        self._turn += 1
        session_suffix = self.bootstrap.session_id[len("chat_session_") :]
        request_id = f"chat_request_{session_suffix}_{self._turn:02d}"
        provider_result = invoke_provider(
            self.provider,
            InterpretIntentRequest(request_id, request, budget=self.budget),
            cancellation=cancellation or self._cancellation,
        )
        policy: IntentPolicyDecision | None = None
        if (
            provider_result.outcome is ProviderOutcome.SUPPORTED
            and isinstance(provider_result.output, IntentProviderOutput)
        ):
            from modelsurgeon.search.intent_policy import evaluate_intent_policy

            policy = evaluate_intent_policy(provider_result.output.intent)
        return ChatTurnResult(
            self.bootstrap.session_id,
            self._turn,
            request_id,
            request,
            provider_result,
            policy,
        )

    def cancel(self) -> None:
        self._cancellation.cancel()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.provider.close()

    def __enter__(self) -> ChatSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "CHAT_SESSION_SCHEMA_VERSION",
    "CHAT_TURN_SCHEMA_VERSION",
    "DEFAULT_CHAT_MAX_TURNS",
    "ChatSession",
    "ChatSessionBootstrap",
    "ChatSessionError",
    "ChatTurnResult",
    "bootstrap_chat_session",
]

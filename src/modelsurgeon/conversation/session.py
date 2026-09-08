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
from typing import TYPE_CHECKING, Literal, cast

from modelsurgeon.experiments import HardwareProfile, build_default_hardware_profile
from modelsurgeon.provider_kind import ProviderKind

from .clarification import (
    ClarificationAnswer,
    ClarificationError,
    ClarificationState,
    apply_clarification_answer,
    build_clarification_state,
)
from .execution import ChatExecutionRecord, ChatOptimizeAdapter, ProgressCallback
from .inspection import (
    ChatInspectionContext,
    ChatInspectionError,
    inspect_local_chat_model,
    no_provider_inspection_context,
)
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
    JSONValue,
    NullTextModelProvider,
    ProviderBudget,
    ProviderOutcome,
    ProviderResult,
    TextModelProvider,
    invoke_provider,
)

if TYPE_CHECKING:
    from modelsurgeon.search.intent_policy import IntentPolicyDecision
    from modelsurgeon.search.spec_preview import SpecPreview

CHAT_SESSION_SCHEMA_VERSION: Literal[1] = 1
CHAT_TURN_SCHEMA_VERSION: Literal[2] = 2
DEFAULT_CHAT_MAX_TURNS = 8
_MAX_CHAT_TURNS = 64


class ChatSessionError(ValueError):
    """Raised when a chat session cannot be safely bootstrapped."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return f"sha256:{hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()}"


def _installed_runtime_revision() -> str:
    try:
        return importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError as error:
        raise ChatSessionError(
            "provider_unavailable",
            "llama-cpp-python is not installed; install a separately managed local runtime",
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
    inspection_context: Mapping[str, object]
    max_turns: int
    schema_version: Literal[1] = CHAT_SESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHAT_SESSION_SCHEMA_VERSION:
            raise ChatSessionError("schema_version", "unsupported chat session schema version")
        if not self.session_id.startswith("chat_session_"):
            raise ChatSessionError("session_id", "invalid chat session identity")
        if not 0 < self.max_turns <= _MAX_CHAT_TURNS:
            raise ChatSessionError("max_turns", "chat turn budget is outside the supported range")
        if self.inspection_context.get("record_type") != "chat_inspection_context":
            raise ChatSessionError("inspection_context", "invalid chat inspection context")

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
            "inspection_context": dict(self.inspection_context),
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
    spec_preview: SpecPreview | None = None
    execution: ChatExecutionRecord | None = None
    clarification: ClarificationState | None = None
    schema_version: Literal[2] = CHAT_TURN_SCHEMA_VERSION

    @property
    def outcome(self) -> str:
        if self.policy_decision is not None:
            return self.policy_decision.outcome.value
        return self.provider_result.outcome.value

    def to_record(self) -> dict[str, object]:
        intent: dict[str, object] | None = None
        if self.provider_result.outcome is ProviderOutcome.SUPPORTED and isinstance(
            self.provider_result.output, IntentProviderOutput
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
            "spec_preview": (None if self.spec_preview is None else self.spec_preview.to_record()),
            "execution": (
                "not_requested" if self.execution is None else self.execution.to_record()
            ),
            "clarification": (
                None if self.clarification is None else self.clarification.to_record()
            ),
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
    validated_model: tuple[Path, str, str] | None = None,
) -> tuple[TextModelProvider, Path | None, str | None, str | None, str | None]:
    if provider_kind is ProviderKind.NONE:
        return NullTextModelProvider(), None, None, None, None
    if provider_kind is not ProviderKind.LOCAL:
        raise ChatSessionError(
            "provider_unsupported",
            f"provider kind {provider_kind.value!r} has no chat adapter in this release",
        )
    if validated_model is None:
        raise ChatSessionError("inspection_context", "local model was not inspected")
    resolved, computed_revision, architecture = validated_model
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
    provider = LocalGGUFProvider(config) if provider_factory is None else provider_factory(config)
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
    hardware_profile_factory: Callable[[str], HardwareProfile] | None = None,
    execution_adapter: ChatOptimizeAdapter | None = None,
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
    inspection_factory = hardware_profile_factory or build_default_hardware_profile
    inspection: ChatInspectionContext
    validated_model: tuple[Path, str, str] | None = None
    if provider_kind is ProviderKind.NONE:
        inspection = no_provider_inspection_context(
            hardware_profile_factory=inspection_factory,
            path=model,
        )
    else:
        try:
            inspection = inspect_local_chat_model(
                model,
                model_revision=model_revision,
                runtime_revision=runtime_revision,
                hardware_profile_factory=inspection_factory,
            )
        except ChatInspectionError as error:
            raise ChatSessionError(error.code, str(error)) from error
        validated_model = (
            Path(str(inspection.model["path"])),
            str(inspection.model["revision"]),
            str(inspection.model["family"]),
        )
    provider, resolved, revision, architecture, selected_runtime_revision = _build_provider(
        model,
        provider_kind=provider_kind,
        model_revision=model_revision,
        runtime_revision=runtime_revision,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_wall_seconds=max_wall_seconds,
        provider_factory=provider_factory,
        validated_model=validated_model,
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
    if selected_runtime_revision is not None:
        inspection = inspection.with_provider(
            runtime_revision=selected_runtime_revision,
            capability_card=card,
        )
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
        inspection.to_record(),
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
        execution_adapter,
    )


class ChatSession:
    """A lifecycle-owned, interpretation-only interaction loop."""

    def __init__(
        self,
        bootstrap: ChatSessionBootstrap,
        provider: TextModelProvider,
        budget: ProviderBudget,
        execution_adapter: ChatOptimizeAdapter | None = None,
    ) -> None:
        self.bootstrap = bootstrap
        self.provider = provider
        self.budget = budget
        self._turn = 0
        self._closed = False
        self._cancellation = CancellationToken()
        self.execution_adapter = execution_adapter

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
            InterpretIntentRequest(
                request_id,
                request,
                budget=self.budget,
                inspection_context=cast(Mapping[str, JSONValue], self.bootstrap.inspection_context),
            ),
            cancellation=cancellation or self._cancellation,
        )
        policy: IntentPolicyDecision | None = None
        preview: SpecPreview | None = None
        clarification: ClarificationState | None = None
        if provider_result.outcome is ProviderOutcome.SUPPORTED and isinstance(
            provider_result.output, IntentProviderOutput
        ):
            from modelsurgeon.search.intent_policy import evaluate_intent_policy
            from modelsurgeon.search.spec_preview import build_spec_preview

            policy = evaluate_intent_policy(provider_result.output.intent)
            preview = build_spec_preview(provider_result.output.intent, decision=policy)
            clarification = build_clarification_state(provider_result.output.intent, policy)
        return ChatTurnResult(
            self.bootstrap.session_id,
            self._turn,
            request_id,
            request,
            provider_result,
            policy,
            preview,
            None,
            clarification,
        )

    def answer_clarification(
        self, turn: ChatTurnResult, answer: ClarificationAnswer
    ) -> ChatTurnResult:
        """Apply one typed answer and replay canonical policy before continuing."""

        if self._closed:
            raise ChatSessionError("session_closed", "chat session is closed")
        if turn.session_id != self.bootstrap.session_id:
            raise ChatSessionError("turn_identity", "chat turn belongs to another session")
        if turn.clarification is None:
            raise ChatSessionError(
                "clarification_unavailable", "the turn has no clarification state"
            )
        try:
            state = apply_clarification_answer(turn.clarification, answer)
        except ClarificationError as error:
            raise ChatSessionError("clarification_failed", str(error)) from error
        preview: SpecPreview | None = None
        if state.policy is not None:
            from modelsurgeon.search.spec_preview import build_spec_preview

            preview = build_spec_preview(
                state.intent,
                decision=state.policy,
                previous=turn.spec_preview,
            )
        return ChatTurnResult(
            self.bootstrap.session_id,
            turn.turn,
            turn.request_id,
            turn.request,
            turn.provider_result,
            state.policy,
            preview,
            None,
            state,
        )

    answer = answer_clarification

    def cancel_clarification(self, turn: ChatTurnResult) -> ChatTurnResult:
        """Cancel clarification without changing the canonical intent or policy."""

        if turn.session_id != self.bootstrap.session_id:
            raise ChatSessionError("turn_identity", "chat turn belongs to another session")
        if turn.clarification is None:
            raise ChatSessionError(
                "clarification_unavailable", "the turn has no clarification state"
            )
        state = turn.clarification
        from .clarification import ClarificationMachine

        cancelled = ClarificationMachine(
            high_confidence=state.high_confidence,
            medium_confidence=state.medium_confidence,
        ).cancel(state)
        return ChatTurnResult(
            self.bootstrap.session_id,
            turn.turn,
            turn.request_id,
            turn.request,
            turn.provider_result,
            turn.policy_decision,
            turn.spec_preview,
            None,
            cancelled,
        )

    def preview_plan(self, turn: ChatTurnResult) -> ChatTurnResult:
        """Call the trusted read-only optimize planner for one interpreted turn."""

        if self.execution_adapter is None:
            raise ChatSessionError("execution_unavailable", "chat optimization is not configured")
        if turn.session_id != self.bootstrap.session_id:
            raise ChatSessionError("turn_identity", "chat turn belongs to another session")
        if turn.spec_preview is None:
            raise ChatSessionError("preview_unavailable", "the turn did not produce a spec preview")
        try:
            execution = self.execution_adapter.preview(
                self.bootstrap.session_id,
                turn.request_id,
                turn.spec_preview,
                provider_context={
                    "record_type": "chat_provider_context",
                    "provider": dict(self.bootstrap.provider),
                    "capability_card": dict(self.bootstrap.capability_card),
                    "runtime_revision": self.bootstrap.runtime_revision,
                },
            )
        except ValueError as error:
            if isinstance(error, ChatSessionError):
                raise
            raise ChatSessionError("execution_preview_failed", str(error)) from error
        return ChatTurnResult(
            turn.session_id,
            turn.turn,
            turn.request_id,
            turn.request,
            turn.provider_result,
            turn.policy_decision,
            turn.spec_preview,
            execution,
        )

    def execute_plan(
        self,
        turn: ChatTurnResult,
        approval_id: str | None,
        *,
        resume: bool | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ChatTurnResult:
        """Submit only the exact confirmed preview to the trusted executor."""

        if self.execution_adapter is None:
            raise ChatSessionError("execution_unavailable", "chat optimization is not configured")
        if turn.session_id != self.bootstrap.session_id:
            raise ChatSessionError("turn_identity", "chat turn belongs to another session")
        if turn.spec_preview is None:
            raise ChatSessionError(
                "execution_unavailable", "the turn did not produce a spec preview"
            )
        try:
            execution = self.execution_adapter.execute(
                self.bootstrap.session_id,
                turn.request_id,
                turn.spec_preview,
                approval_id,
                resume=resume,
                progress_callback=progress_callback,
                provider_context={
                    "record_type": "chat_provider_context",
                    "provider": dict(self.bootstrap.provider),
                    "capability_card": dict(self.bootstrap.capability_card),
                    "runtime_revision": self.bootstrap.runtime_revision,
                },
            )
        except ValueError as error:
            if isinstance(error, ChatSessionError):
                raise
            raise ChatSessionError("execution_failed", str(error)) from error
        return ChatTurnResult(
            turn.session_id,
            turn.turn,
            turn.request_id,
            turn.request,
            turn.provider_result,
            turn.policy_decision,
            turn.spec_preview,
            execution,
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

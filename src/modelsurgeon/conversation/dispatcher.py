"""Fail-closed execution for capability-scoped conversational tools.

The schema module deliberately contains no executor.  This module is the
small trusted runtime boundary that may be used by an engine adapter.  Model
records are decoded and negotiated before a registered handler is reached;
handlers never come from a request payload and receive a cooperative budget
and cancellation context.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from modelsurgeon.conversation.tools import (
    DEFAULT_TOOL_CATALOG,
    MAX_TOOL_RESULT_BYTES,
    JSONValue,
    ToolBudget,
    ToolCatalog,
    ToolContractError,
    ToolDefinition,
    ToolEvidenceStatus,
    ToolFailure,
    ToolFailureCode,
    ToolNegotiation,
    ToolOutcome,
    ToolProvenance,
    ToolRequest,
    ToolResult,
    tool_request_digest,
)
from modelsurgeon.experiments.identity import canonical_identity_json


class ToolDispatchError(RuntimeError):
    """Raised for an invalid trusted dispatcher configuration."""


class ToolExecutionError(Exception):
    """A bounded, typed failure raised by a trusted handler."""

    def __init__(
        self,
        code: ToolFailureCode,
        detail: str,
        *,
        retryable: bool = False,
        raw_payload: Mapping[str, JSONValue] | None = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = (
            detail if isinstance(detail, str) and detail.strip() else "tool execution failed"
        )
        self.retryable = retryable
        self.raw_payload = raw_payload


class ToolCancellationToken:
    """Thread-safe cancellation signal shared with a trusted handler."""

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()


@dataclass(frozen=True, slots=True)
class ToolExecutionResponse:
    """Engine-owned output and evidence lineage returned by a handler."""

    output: Mapping[str, JSONValue]
    evidence_status: ToolEvidenceStatus = ToolEvidenceStatus.UNVERIFIED
    source_digest: str | None = None
    evidence_id: str | None = None
    artifact_id: str | None = None
    campaign_id: str | None = None
    observed_at: str | None = None


@dataclass(frozen=True, slots=True)
class ToolUsage:
    """Measured or explicitly reserved consumption for one dispatch."""

    wall_seconds: float
    memory_bytes: int
    evaluation_count: int
    output_bytes: int

    def __post_init__(self) -> None:
        if self.wall_seconds < 0 or self.memory_bytes < 0:
            raise ToolDispatchError("tool usage cannot be negative")
        if self.evaluation_count < 0 or self.output_bytes < 0:
            raise ToolDispatchError("tool usage cannot be negative")

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "wall_seconds": self.wall_seconds,
            "memory_bytes": self.memory_bytes,
            "evaluation_count": self.evaluation_count,
            "output_bytes": self.output_bytes,
        }


@dataclass(frozen=True, slots=True)
class ToolExecutionReceipt:
    """Non-authoritative dispatch telemetry kept beside the typed result."""

    attempts: int
    usage: ToolUsage
    replayed: bool = False

    def __post_init__(self) -> None:
        if self.attempts <= 0:
            raise ToolDispatchError("tool dispatch attempts must be positive")


@dataclass(frozen=True, slots=True)
class ToolDispatchResult:
    """Result of decoding, negotiating, and possibly executing a request.

    Malformed untrusted records intentionally have no ``request`` or typed
    ``result`` because there is no canonical request identity to bind them to.
    """

    negotiation: ToolNegotiation
    request: ToolRequest | None
    result: ToolResult | None
    receipt: ToolExecutionReceipt | None = None


@runtime_checkable
class ToolHandler(Protocol):
    """Trusted engine adapter for one catalog entry."""

    def __call__(self, context: ToolExecutionContext) -> ToolExecutionResponse | Mapping[
        str, JSONValue
    ]: ...


@runtime_checkable
class ToolApprovalPolicy(Protocol):
    """Trusted approval binding checked independently of tool arguments."""

    def __call__(self, request: ToolRequest) -> bool: ...


class ToolExecutionContext:
    """Cooperative resource and cancellation boundary for a trusted handler."""

    def __init__(
        self,
        request: ToolRequest,
        budget: ToolBudget,
        cancellation: ToolCancellationToken,
        clock: Callable[[], float],
    ) -> None:
        self.request = request
        self.budget = budget
        self.cancellation = cancellation
        self._clock = clock
        self._started_at = clock()
        # Entering a handler is one bounded evaluation. Nested work must be
        # charged explicitly with consume_evaluations().
        self._evaluation_count = 1
        self._memory_bytes = 0

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._started_at)

    @property
    def memory_bytes(self) -> int:
        return self._memory_bytes

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    def check_cancelled(self) -> None:
        if self.cancellation.cancelled:
            raise ToolExecutionError(ToolFailureCode.CANCELLED, "tool execution was cancelled")
        if self.elapsed_seconds > self.budget.max_wall_seconds:
            raise ToolExecutionError(ToolFailureCode.TIMEOUT, "tool wall-time budget was exhausted")

    def consume_evaluations(self, count: int = 1) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ToolDispatchError("evaluation count must be a positive integer")
        self.check_cancelled()
        if self._evaluation_count + count > self.budget.max_evaluation_count:
            raise ToolExecutionError(
                ToolFailureCode.BUDGET_EXCEEDED,
                "tool evaluation budget was exhausted",
            )
        self._evaluation_count += count

    def reserve_memory(self, byte_count: int) -> None:
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise ToolDispatchError("reserved memory must be a non-negative integer")
        self.check_cancelled()
        if byte_count > self.budget.max_memory_bytes - self._memory_bytes:
            raise ToolExecutionError(
                ToolFailureCode.BUDGET_EXCEEDED,
                "tool memory budget was exhausted",
            )
        self._memory_bytes += byte_count


def _budget_exceeds(requested: ToolBudget, ceiling: ToolBudget) -> bool:
    return (
        requested.max_wall_seconds > ceiling.max_wall_seconds
        or requested.max_memory_bytes > ceiling.max_memory_bytes
        or requested.max_evaluation_count > ceiling.max_evaluation_count
        or requested.max_output_bytes > ceiling.max_output_bytes
    )


def _failure_outcome(code: ToolFailureCode, fallback: ToolOutcome) -> ToolOutcome:
    if code is ToolFailureCode.UNKNOWN_TOOL:
        return ToolOutcome.UNKNOWN
    if code in {ToolFailureCode.UNSUPPORTED_CAPABILITY, ToolFailureCode.HANDLER_UNAVAILABLE}:
        return ToolOutcome.UNSUPPORTED
    if code is ToolFailureCode.TIMEOUT:
        return ToolOutcome.TIMEOUT
    if code is ToolFailureCode.CANCELLED:
        return ToolOutcome.CANCELLED
    if fallback in {ToolOutcome.UNKNOWN, ToolOutcome.UNSUPPORTED, ToolOutcome.REFUSED}:
        return fallback
    return ToolOutcome.FAILED


class ToolDispatcher:
    """Validate and execute only registered, budgeted tool operations.

    The dispatcher is intentionally handler-agnostic. The engine registers a
    trusted adapter for a named catalog entry; a request can never select a
    Python callback, command, path, or provider. ``max_retries`` applies only
    to handlers that raise a retryable :class:`ToolExecutionError` and is
    bounded at construction time.
    """

    def __init__(
        self,
        handlers: Mapping[str, ToolHandler] | None = None,
        *,
        catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
        approval_policy: ToolApprovalPolicy | None = None,
        budget_ceiling: ToolBudget | None = None,
        max_retries: int = 0,
        max_replay_entries: int = 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or not 0 <= max_retries <= 3
        ):
            raise ToolDispatchError("max_retries must be between zero and three")
        if (
            isinstance(max_replay_entries, bool)
            or not isinstance(max_replay_entries, int)
            or max_replay_entries <= 0
        ):
            raise ToolDispatchError("max_replay_entries must be positive")
        self.catalog = catalog
        self.approval_policy = approval_policy
        self.budget_ceiling = budget_ceiling
        self.max_retries = max_retries
        self.max_replay_entries = max_replay_entries
        self._clock = clock
        self._handlers: dict[str, ToolHandler] = {}
        for name, handler in (handlers or {}).items():
            self.register(name, handler)
        self._lock = threading.Lock()
        self._completed: dict[str, tuple[ToolResult, ToolExecutionReceipt]] = {}
        self._inflight: set[str] = set()

    def register(self, name: str, handler: ToolHandler) -> None:
        """Register a trusted handler for an existing allowlisted tool."""

        definition = self.catalog.definition(name)
        if definition is None:
            raise ToolDispatchError("cannot register a handler for an unknown tool")
        if not callable(handler):
            raise ToolDispatchError("tool handler must be callable")
        if name in self._handlers:
            raise ToolDispatchError("tool handler is already registered")
        self._handlers[name] = handler

    def dispatch_record(self, payload: object) -> ToolDispatchResult:
        """Decode an untrusted record and fail closed before handler lookup."""

        try:
            request = ToolRequest.from_record(payload)
        except ToolContractError:
            negotiation = self.catalog.negotiate_record(payload)
            return ToolDispatchResult(negotiation, None, None)
        return self.dispatch(request)

    def dispatch(
        self,
        request: ToolRequest,
        *,
        cancellation: ToolCancellationToken | None = None,
    ) -> ToolDispatchResult:
        """Run one typed request, returning a grounded typed result envelope."""

        if not isinstance(request, ToolRequest):
            raise ToolContractError("tool dispatcher requires a typed ToolRequest")
        negotiation = self.catalog.negotiate(request)
        definition = negotiation.tool or self.catalog.definition(request.name)
        if negotiation.outcome is not ToolOutcome.SUPPORTED or definition is None:
            return self._finish_refusal(request, negotiation, definition)
        if self.budget_ceiling is not None and _budget_exceeds(request.budget, self.budget_ceiling):
            return self._finish_failure(
                request,
                definition,
                ToolOutcome.REFUSED,
                ToolFailureCode.BUDGET_EXCEEDED,
                "request exceeds the dispatcher budget ceiling",
            )
        if definition.approval_required:
            if self.approval_policy is None:
                return self._finish_failure(
                    request,
                    definition,
                    ToolOutcome.REFUSED,
                    ToolFailureCode.APPROVAL_REQUIRED,
                    "consequential dispatch requires a trusted approval policy",
                )
            try:
                approved = self.approval_policy(request)
            except Exception:
                approved = False
            if approved is not True:
                return self._finish_failure(
                    request,
                    definition,
                    ToolOutcome.REFUSED,
                    ToolFailureCode.APPROVAL_INVALID,
                    "approval policy rejected the request",
                )
        handler = self._handlers.get(request.name)
        if handler is None:
            return self._finish_failure(
                request,
                definition,
                ToolOutcome.UNSUPPORTED,
                ToolFailureCode.HANDLER_UNAVAILABLE,
                "no trusted engine handler is registered",
            )
        if cancellation is not None and cancellation.cancelled:
            return self._finish_failure(
                request,
                definition,
                ToolOutcome.CANCELLED,
                ToolFailureCode.CANCELLED,
                "tool execution was cancelled",
            )

        request_digest = tool_request_digest(request)
        with self._lock:
            replay = self._completed.get(request.request_id)
            if replay is not None:
                prior_result, prior_receipt = replay
                if prior_result.provenance.request_digest != request_digest:
                    return self._finish_failure(
                        request,
                        definition,
                        ToolOutcome.REFUSED,
                        ToolFailureCode.REPLAYED_REQUEST,
                        "request identity is already bound to different evidence",
                    )
                receipt = ToolExecutionReceipt(
                    prior_receipt.attempts,
                    prior_receipt.usage,
                    replayed=True,
                )
                return ToolDispatchResult(negotiation, request, prior_result, receipt)
            if request.request_id in self._inflight:
                return self._finish_failure(
                    request,
                    definition,
                    ToolOutcome.REFUSED,
                    ToolFailureCode.REPLAYED_REQUEST,
                    "request is already executing",
                )
            if len(self._completed) >= self.max_replay_entries:
                return self._finish_failure(
                    request,
                    definition,
                    ToolOutcome.REFUSED,
                    ToolFailureCode.REPLAY_LEDGER_FULL,
                    "replay ledger capacity is exhausted",
                )
            self._inflight.add(request.request_id)

        token = cancellation or ToolCancellationToken()
        result, receipt = self._execute(request, definition, handler, token)
        with self._lock:
            self._inflight.discard(request.request_id)
            self._completed[request.request_id] = (result, receipt)
        return ToolDispatchResult(negotiation, request, result, receipt)

    def _execute(
        self,
        request: ToolRequest,
        definition: ToolDefinition,
        handler: ToolHandler,
        cancellation: ToolCancellationToken,
    ) -> tuple[ToolResult, ToolExecutionReceipt]:
        started = self._clock()
        attempts = 0
        last_failure: ToolExecutionError | None = None
        for attempts in range(1, self.max_retries + 2):
            if attempts > 1 and self._clock() - started >= request.budget.max_wall_seconds:
                cancellation.cancel()
                context = ToolExecutionContext(request, request.budget, cancellation, self._clock)
                result = self._failure_result(
                    request,
                    definition,
                    ToolOutcome.TIMEOUT,
                    ToolFailureCode.TIMEOUT,
                    "tool wall-time budget was exhausted",
                )
                return result, self._receipt(attempts - 1, context, started, 0)
            context = ToolExecutionContext(request, request.budget, cancellation, self._clock)
            outcome_queue: queue.Queue[tuple[object, BaseException | None]] = queue.Queue(maxsize=1)
            worker = threading.Thread(
                target=self._invoke_handler,
                args=(handler, context, outcome_queue),
                daemon=True,
            )
            worker.start()
            while worker.is_alive():
                elapsed = max(0.0, self._clock() - started)
                remaining = request.budget.max_wall_seconds - elapsed
                if cancellation.cancelled:
                    cancellation.cancel()
                    result = self._failure_result(
                        request,
                        definition,
                        ToolOutcome.CANCELLED,
                        ToolFailureCode.CANCELLED,
                        "tool execution was cancelled",
                    )
                    return result, self._receipt(attempts, context, started, 0)
                if remaining <= 0:
                    cancellation.cancel()
                    result = self._failure_result(
                        request,
                        definition,
                        ToolOutcome.TIMEOUT,
                        ToolFailureCode.TIMEOUT,
                        "tool wall-time budget was exhausted",
                    )
                    return result, self._receipt(attempts, context, started, 0)
                worker.join(min(remaining, 0.01))
            elapsed = max(0.0, self._clock() - started)
            if elapsed > request.budget.max_wall_seconds:
                cancellation.cancel()
                result = self._failure_result(
                    request,
                    definition,
                    ToolOutcome.TIMEOUT,
                    ToolFailureCode.TIMEOUT,
                    "tool wall-time budget was exhausted",
                )
                return result, self._receipt(attempts, context, started, 0)
            try:
                value, error = outcome_queue.get_nowait()
            except queue.Empty:
                error = ToolExecutionError(
                    ToolFailureCode.EXECUTION_FAILED,
                    "tool handler returned no result",
                )
                value = None
            if error is not None:
                failure = self._as_execution_error(error)
                if failure.retryable and attempts <= self.max_retries:
                    last_failure = failure
                    continue
                result = self._failure_result(
                    request,
                    definition,
                    _failure_outcome(failure.code, ToolOutcome.FAILED),
                    failure.code,
                    failure.detail,
                    retryable=failure.retryable,
                    raw_payload=failure.raw_payload,
                )
                return result, self._receipt(attempts, context, started, 0)
            try:
                if isinstance(value, ToolExecutionResponse):
                    response = value
                elif isinstance(value, Mapping):
                    response = ToolExecutionResponse(value)
                else:
                    raise ToolExecutionError(
                        ToolFailureCode.OUTPUT_INVALID,
                        "tool handler returned a non-object output",
                    )
                definition.validate_output(response.output)
                output_bytes = len(canonical_identity_json(dict(response.output)).encode("utf-8"))
                if output_bytes > request.budget.max_output_bytes:
                    raise ToolExecutionError(
                        ToolFailureCode.BUDGET_EXCEEDED,
                        "tool output budget was exhausted",
                    )
                if context.memory_bytes > request.budget.max_memory_bytes:
                    raise ToolExecutionError(
                        ToolFailureCode.BUDGET_EXCEEDED,
                        "tool memory budget was exhausted",
                    )
                if context.evaluation_count > request.budget.max_evaluation_count:
                    raise ToolExecutionError(
                        ToolFailureCode.BUDGET_EXCEEDED,
                        "tool evaluation budget was exhausted",
                    )
                provenance = ToolProvenance(
                    definition.owner,
                    definition.tool_id,
                    tool_request_digest(request),
                    response.source_digest,
                    response.evidence_id,
                    response.artifact_id,
                    response.campaign_id,
                    response.observed_at,
                    response.evidence_status,
                )
                result = ToolResult(
                    request.request_id,
                    request.name,
                    ToolOutcome.SUPPORTED,
                    provenance,
                    response.output,
                )
                if len(result.canonical_json().encode("utf-8")) > MAX_TOOL_RESULT_BYTES:
                    raise ToolExecutionError(
                        ToolFailureCode.BUDGET_EXCEEDED,
                        "tool result hard size limit was exhausted",
                    )
            except ToolExecutionError as execution_failure:
                result = self._failure_result(
                    request,
                    definition,
                    _failure_outcome(execution_failure.code, ToolOutcome.FAILED),
                    execution_failure.code,
                    execution_failure.detail,
                    retryable=execution_failure.retryable,
                    raw_payload=execution_failure.raw_payload,
                )
                return result, self._receipt(attempts, context, started, 0)
            except (ToolContractError, TypeError, ValueError):
                result = self._failure_result(
                    request,
                    definition,
                    ToolOutcome.FAILED,
                    ToolFailureCode.OUTPUT_INVALID,
                    "tool handler returned output outside its declared schema",
                )
                return result, self._receipt(attempts, context, started, 0)
            return result, self._receipt(attempts, context, started, output_bytes)

        exhausted = last_failure or ToolExecutionError(
            ToolFailureCode.EXECUTION_FAILED, "tool handler failed"
        )
        result = self._failure_result(
            request,
            definition,
            ToolOutcome.FAILED,
            exhausted.code,
            exhausted.detail,
            retryable=exhausted.retryable,
            raw_payload=exhausted.raw_payload,
        )
        context = ToolExecutionContext(request, request.budget, cancellation, self._clock)
        return result, self._receipt(attempts, context, started, 0)

    @staticmethod
    def _invoke_handler(
        handler: ToolHandler,
        context: ToolExecutionContext,
        outcome_queue: queue.Queue[tuple[object, BaseException | None]],
    ) -> None:
        try:
            outcome_queue.put((handler(context), None))
        except BaseException as error:  # transport the trusted handler failure safely
            outcome_queue.put((None, error))

    def _receipt(
        self,
        attempts: int,
        context: ToolExecutionContext,
        started: float,
        output_bytes: int,
        *,
        replayed: bool = False,
    ) -> ToolExecutionReceipt:
        return ToolExecutionReceipt(
            attempts,
            ToolUsage(
                max(0.0, self._clock() - started),
                context.memory_bytes,
                context.evaluation_count,
                output_bytes,
            ),
            replayed,
        )

    @staticmethod
    def _as_execution_error(error: BaseException) -> ToolExecutionError:
        if isinstance(error, ToolExecutionError):
            return error
        return ToolExecutionError(ToolFailureCode.EXECUTION_FAILED, "tool handler failed")

    def _finish_refusal(
        self,
        request: ToolRequest,
        negotiation: ToolNegotiation,
        definition: ToolDefinition | None,
    ) -> ToolDispatchResult:
        failure = negotiation.failure
        if failure is None:
            raise ToolDispatchError("refused negotiation did not contain a failure")
        result = self._failure_result(
            request,
            definition,
            negotiation.outcome,
            failure.code,
            failure.detail,
            retryable=failure.retryable,
        )
        receipt = ToolExecutionReceipt(1, ToolUsage(0.0, 0, 0, 0))
        return ToolDispatchResult(negotiation, request, result, receipt)

    def _finish_failure(
        self,
        request: ToolRequest,
        definition: ToolDefinition,
        outcome: ToolOutcome,
        code: ToolFailureCode,
        detail: str,
    ) -> ToolDispatchResult:
        negotiation = ToolNegotiation(
            request.request_id,
            request.name,
            outcome,
            tool=None,
            failure=ToolFailure(code, detail, request.request_id),
        )
        result = self._failure_result(request, definition, outcome, code, detail)
        return ToolDispatchResult(
            negotiation,
            request,
            result,
            ToolExecutionReceipt(1, ToolUsage(0.0, 0, 0, 0)),
        )

    @staticmethod
    def _failure_result(
        request: ToolRequest,
        definition: ToolDefinition | None,
        outcome: ToolOutcome,
        code: ToolFailureCode,
        detail: str,
        *,
        retryable: bool = False,
        raw_payload: Mapping[str, JSONValue] | None = None,
    ) -> ToolResult:
        owner = definition.owner if definition is not None else "modelsurgeon.conversation"
        tool_id = definition.tool_id if definition is not None else "unknown"
        bounded_raw = None
        if raw_payload is not None:
            try:
                encoded = canonical_identity_json(dict(raw_payload)).encode("utf-8")
                if len(encoded) <= MAX_TOOL_RESULT_BYTES // 4:
                    bounded_raw = raw_payload
            except (TypeError, ValueError):
                bounded_raw = None
        return ToolResult(
            request.request_id,
            request.name,
            _failure_outcome(code, outcome),
            ToolProvenance(
                owner,
                tool_id,
                tool_request_digest(request),
                evidence_status=ToolEvidenceStatus.UNAVAILABLE,
            ),
            failure=ToolFailure(code, detail, request.request_id, retryable),
            raw_payload=bounded_raw,
        )


__all__ = [
    "ToolApprovalPolicy",
    "ToolCancellationToken",
    "ToolDispatchError",
    "ToolDispatchResult",
    "ToolDispatcher",
    "ToolExecutionContext",
    "ToolExecutionError",
    "ToolExecutionReceipt",
    "ToolExecutionResponse",
    "ToolHandler",
    "ToolUsage",
]

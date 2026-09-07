"""Fail-closed transaction handles for conversational tool dispatch.

The conversational layer owns only the control-plane lifecycle.  A trusted
engine adapter may bind a consequential handle to its already-active mutation
transaction through :class:`ToolTransactionParticipant`; the chat layer never
receives a mutation primitive or a model object.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from modelsurgeon.conversation.tools import (
    ToolAccess,
    ToolContractError,
    ToolDefinition,
    ToolRequest,
    tool_request_digest,
)


class ToolTransactionError(ToolContractError):
    """Raised when a conversational transaction boundary is violated."""


class ToolTransactionState(StrEnum):
    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class ToolTransactionParticipant(Protocol):
    """Trusted engine-owned transaction already prepared for one tool call."""

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ToolTransactionHandle:
    """Immutable capability for one transaction generation.

    Handles are intentionally stale after a lifecycle transition.  Callers
    must use the handle returned by ``commit``, ``rollback`` or ``cancel``;
    retaining an earlier handle cannot operate on a later transaction.
    """

    transaction_id: str
    request_id: str
    request_digest: str
    access: ToolAccess
    generation: int
    state: ToolTransactionState


@dataclass(slots=True)
class ToolTransactionContext:
    """The only transaction capability exposed to a trusted tool handler."""

    _boundary: ToolTransactionBoundary
    _handle: ToolTransactionHandle

    @property
    def handle(self) -> ToolTransactionHandle:
        return self._handle

    @property
    def transaction_id(self) -> str:
        return self._handle.transaction_id

    @property
    def state(self) -> ToolTransactionState:
        return self._handle.state

    @property
    def access(self) -> ToolAccess:
        return self._handle.access

    def require_active(self) -> None:
        self._boundary.require_active(self._handle)

    def commit(self) -> None:
        self._handle = self._boundary.commit(self._handle)

    def rollback(self) -> None:
        self._handle = self._boundary.rollback(self._handle)

    def cancel(self) -> None:
        self._handle = self._boundary.cancel(self._handle)


class ToolTransactionBoundary:
    """Manage bounded conversational lifecycle and stale-handle protection.

    Read-only transactions never accept an engine participant.  Consequential
    transactions may receive one from trusted engine code; its commit and
    rollback methods are the only physical lifecycle hooks at this boundary.
    """

    def __init__(
        self,
        *,
        max_active: int = 64,
        participant_factory: Callable[
            [ToolRequest, ToolDefinition, int], ToolTransactionParticipant | None
        ]
        | None = None,
    ) -> None:
        if isinstance(max_active, bool) or not isinstance(max_active, int) or max_active <= 0:
            raise ToolTransactionError("max_active transactions must be positive")
        self.max_active = max_active
        self.participant_factory = participant_factory
        self._lock = threading.Lock()
        self._active: dict[
            str, tuple[ToolTransactionHandle, ToolTransactionParticipant | None]
        ] = {}

    def begin(
        self,
        request: ToolRequest,
        definition: ToolDefinition,
        attempt: int,
    ) -> ToolTransactionContext:
        if not isinstance(request, ToolRequest) or not isinstance(definition, ToolDefinition):
            raise ToolTransactionError("transaction requires typed request and definition")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
            raise ToolTransactionError("transaction attempt must be positive")
        request_digest = tool_request_digest(request)
        identity = f"{request_digest}:{definition.access.value}:{attempt}"
        transaction_id = "tool_transaction_" + hashlib.sha256(identity.encode()).hexdigest()
        participant = None
        if definition.access is ToolAccess.CONSEQUENTIAL and self.participant_factory is not None:
            try:
                participant = self.participant_factory(request, definition, attempt)
            except BaseException as error:
                raise ToolTransactionError(
                    "trusted transaction participant could not be prepared"
                ) from error
            if participant is not None and (
                not callable(getattr(participant, "commit", None))
                or not callable(getattr(participant, "rollback", None))
            ):
                raise ToolTransactionError("transaction participant lacks commit/rollback")
        handle = ToolTransactionHandle(
            transaction_id,
            request.request_id,
            request_digest,
            definition.access,
            0,
            ToolTransactionState.ACTIVE,
        )
        with self._lock:
            if len(self._active) >= self.max_active:
                raise ToolTransactionError("transaction boundary capacity is exhausted")
            if request.request_id in {item[0].request_id for item in self._active.values()}:
                raise ToolTransactionError("request already owns an active transaction")
            self._active[transaction_id] = (handle, participant)
        return ToolTransactionContext(self, handle)

    def require_active(self, handle: ToolTransactionHandle) -> None:
        self._lookup(handle)

    def commit(self, handle: ToolTransactionHandle) -> ToolTransactionHandle:
        current, participant = self._lookup(handle)
        if participant is not None:
            try:
                participant.commit()
            except BaseException as error:
                self._transition(current, ToolTransactionState.ROLLED_BACK)
                raise ToolTransactionError("engine transaction commit failed") from error
        return self._transition(current, ToolTransactionState.COMMITTED)

    def rollback(self, handle: ToolTransactionHandle) -> ToolTransactionHandle:
        current, participant = self._lookup(handle)
        if participant is not None:
            try:
                participant.rollback()
            except BaseException as error:
                self._transition(current, ToolTransactionState.ROLLED_BACK)
                raise ToolTransactionError("engine transaction rollback failed") from error
        return self._transition(current, ToolTransactionState.ROLLED_BACK)

    def cancel(self, handle: ToolTransactionHandle) -> ToolTransactionHandle:
        current, participant = self._lookup(handle)
        if participant is not None:
            try:
                participant.rollback()
            except BaseException as error:
                self._transition(current, ToolTransactionState.CANCELLED)
                raise ToolTransactionError(
                    "engine transaction cancellation rollback failed"
                ) from error
        return self._transition(current, ToolTransactionState.CANCELLED)

    def close_failure(self, context: ToolTransactionContext, *, cancelled: bool = False) -> None:
        """Rollback an active call; preserve committed state as a hard failure."""

        if context.state is not ToolTransactionState.ACTIVE:
            return
        if cancelled:
            context.cancel()
        else:
            context.rollback()

    def _lookup(
        self, handle: ToolTransactionHandle
    ) -> tuple[ToolTransactionHandle, ToolTransactionParticipant | None]:
        with self._lock:
            current = self._active.get(handle.transaction_id)
            if current is None:
                raise ToolTransactionError("transaction handle is stale or unknown")
            current_handle, participant = current
            if current_handle != handle or current_handle.state is not ToolTransactionState.ACTIVE:
                raise ToolTransactionError("transaction handle is stale or no longer active")
            return current_handle, participant

    def _transition(
        self, current: ToolTransactionHandle, state: ToolTransactionState
    ) -> ToolTransactionHandle:
        next_handle = ToolTransactionHandle(
            current.transaction_id,
            current.request_id,
            current.request_digest,
            current.access,
            current.generation + 1,
            state,
        )
        with self._lock:
            stored = self._active.get(current.transaction_id)
            if stored is None or stored[0] != current:
                raise ToolTransactionError("transaction handle became stale during transition")
            del self._active[current.transaction_id]
        return next_handle


__all__ = [
    "ToolTransactionBoundary",
    "ToolTransactionContext",
    "ToolTransactionError",
    "ToolTransactionHandle",
    "ToolTransactionParticipant",
    "ToolTransactionState",
]

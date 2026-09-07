"""Transaction, retry, cancellation, and fixture-campaign boundary coverage."""

from __future__ import annotations

import threading
import time

import pytest

from modelsurgeon.conversation import (
    DEFAULT_TOOL_CATALOG,
    ToolCancellationToken,
    ToolDispatcher,
    ToolExecutionContext,
    ToolExecutionError,
    ToolFailureCode,
    ToolOutcome,
    ToolRequest,
    ToolTransactionBoundary,
    ToolTransactionError,
    ToolTransactionState,
)


class FixtureCampaign:
    """Tiny campaign/artifact pair used to exercise the real dispatcher path."""

    def __init__(self) -> None:
        self.artifact = "source"
        self.participants: list[FixtureParticipant] = []

    def prepare(
        self, _request: ToolRequest, _definition: object, _attempt: int
    ) -> FixtureParticipant:
        participant = FixtureParticipant(self)
        self.participants.append(participant)
        return participant


class FixtureParticipant:
    def __init__(self, campaign: FixtureCampaign) -> None:
        self.campaign = campaign
        self.before = campaign.artifact
        self.commit_calls = 0
        self.rollback_calls = 0

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1
        self.campaign.artifact = self.before


def _request() -> ToolRequest:
    definition = DEFAULT_TOOL_CATALOG.definition("execute_approved_plan")
    assert definition is not None
    return ToolRequest.create(
        definition,
        {
            "plan_id": "plan.fixture",
            "plan_digest": "sha256:" + "a" * 64,
            "approval_id": "approval.fixture",
        },
        approval_id="approval.fixture",
    )


def _output() -> dict[str, object]:
    return {
        "run_id": "run.fixture",
        "outcome": "supported",
        "evidence_refs": [],
        "artifact_ref": "artifact.fixture",
    }


def _dispatcher(campaign: FixtureCampaign, handler: object, **kwargs: object) -> ToolDispatcher:
    return ToolDispatcher(
        {"execute_approved_plan": handler},
        approval_policy=lambda _request: True,
        transaction_boundary=ToolTransactionBoundary(
            participant_factory=campaign.prepare,
        ),
        **kwargs,
    )


def test_transaction_handles_transition_and_stale_handles_fail_closed() -> None:
    request = _request()
    definition = DEFAULT_TOOL_CATALOG.definition(request.name)
    assert definition is not None
    boundary = ToolTransactionBoundary()
    context = boundary.begin(request, definition, 1)
    original = context.handle

    assert context.state is ToolTransactionState.ACTIVE
    context.commit()
    assert context.state is ToolTransactionState.COMMITTED
    assert context.handle.generation == original.generation + 1

    with pytest.raises(ToolTransactionError, match="stale"):
        boundary.rollback(original)
    with pytest.raises(ToolTransactionError, match="stale"):
        context.require_active()


def test_read_only_transaction_has_no_mutable_participant_or_mutation_method() -> None:
    request = ToolRequest.create(
        DEFAULT_TOOL_CATALOG.definition("inspect_model"),  # type: ignore[arg-type]
        {"model_ref": "fixture.model"},
    )
    definition = DEFAULT_TOOL_CATALOG.definition(request.name)
    assert definition is not None
    prepared = 0

    def participant_factory(_request: ToolRequest, _definition: object, _attempt: int) -> None:
        nonlocal prepared
        prepared += 1
        return None

    boundary = ToolTransactionBoundary(participant_factory=participant_factory)
    context = boundary.begin(request, definition, 1)
    assert prepared == 0
    assert not hasattr(context, "mutate")
    context.commit()


def test_uncommitted_consequential_handler_rolls_back_fixture_campaign() -> None:
    campaign = FixtureCampaign()

    def handler(_context: ToolExecutionContext) -> dict[str, object]:
        campaign.artifact = "child"
        return _output()

    dispatched = _dispatcher(campaign, handler).dispatch(_request())

    assert dispatched.result is not None
    assert dispatched.result.outcome is ToolOutcome.FAILED
    assert dispatched.result.failure is not None
    assert dispatched.result.failure.code is ToolFailureCode.TRANSACTION_REQUIRED
    assert campaign.artifact == "source"
    assert campaign.participants[0].rollback_calls == 1
    assert campaign.participants[0].commit_calls == 0


def test_committed_consequential_handler_publishes_fixture_artifact() -> None:
    campaign = FixtureCampaign()

    def handler(context: ToolExecutionContext) -> dict[str, object]:
        campaign.artifact = "child"
        context.transaction.commit()
        return _output()

    dispatched = _dispatcher(campaign, handler).dispatch(_request())

    assert dispatched.result is not None
    assert dispatched.result.outcome is ToolOutcome.SUPPORTED
    assert campaign.artifact == "child"
    assert campaign.participants[0].rollback_calls == 0
    assert campaign.participants[0].commit_calls == 1
    assert dispatched.receipt is not None
    assert dispatched.receipt.transaction_id is not None


def test_consequential_retry_is_rejected_without_idempotency_declaration() -> None:
    campaign = FixtureCampaign()
    calls = 0

    def handler(_context: ToolExecutionContext) -> dict[str, object]:
        nonlocal calls
        calls += 1
        campaign.artifact = "partial"
        raise ToolExecutionError(
            ToolFailureCode.EXECUTION_FAILED,
            "fixture failed after applying work",
            retryable=True,
        )

    dispatched = _dispatcher(campaign, handler, max_retries=2).dispatch(_request())

    assert calls == 1
    assert dispatched.result is not None
    assert dispatched.result.failure is not None
    assert dispatched.result.failure.code is ToolFailureCode.RETRY_NOT_SAFE
    assert campaign.artifact == "source"
    assert len(campaign.participants) == 1
    assert campaign.participants[0].rollback_calls == 1


def test_idempotent_consequential_retry_rolls_back_then_commits_once() -> None:
    campaign = FixtureCampaign()
    calls = 0

    def handler(context: ToolExecutionContext) -> dict[str, object]:
        nonlocal calls
        calls += 1
        campaign.artifact = f"attempt-{calls}"
        if calls == 1:
            raise ToolExecutionError(
                ToolFailureCode.EXECUTION_FAILED,
                "retryable fixture failure",
                retryable=True,
                idempotent=True,
            )
        context.transaction.commit()
        return _output()

    dispatched = _dispatcher(campaign, handler, max_retries=1).dispatch(_request())

    assert calls == 2
    assert dispatched.result is not None
    assert dispatched.result.outcome is ToolOutcome.SUPPORTED
    assert campaign.artifact == "attempt-2"
    assert campaign.participants[0].rollback_calls == 1
    assert campaign.participants[1].commit_calls == 1
    assert dispatched.receipt is not None and dispatched.receipt.attempts == 2


def test_cancellation_rolls_back_active_consequential_transaction() -> None:
    campaign = FixtureCampaign()
    started = threading.Event()
    cancellation = ToolCancellationToken()

    def handler(_context: ToolExecutionContext) -> dict[str, object]:
        campaign.artifact = "cancelled-child"
        started.set()
        time.sleep(0.1)
        return _output()

    result_holder: list[object] = []

    def run() -> None:
        result_holder.append(
            _dispatcher(campaign, handler).dispatch(_request(), cancellation=cancellation)
        )

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(1)
    cancellation.cancel()
    worker.join(2)

    assert not worker.is_alive()
    dispatched = result_holder[0]
    assert dispatched.result is not None  # type: ignore[attr-defined]
    assert dispatched.result.outcome is ToolOutcome.CANCELLED  # type: ignore[attr-defined]
    assert campaign.artifact == "source"
    assert campaign.participants[0].rollback_calls == 1

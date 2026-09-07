"""Runtime enforcement tests for the conversational tool dispatcher."""

from __future__ import annotations

import threading
import time

from modelsurgeon.conversation import (
    DEFAULT_TOOL_CATALOG,
    ToolBudget,
    ToolCancellationToken,
    ToolDispatcher,
    ToolEvidenceStatus,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionResponse,
    ToolFailureCode,
    ToolOutcome,
    ToolRequest,
    deterministic_tool_request_id,
    tool_request_digest,
)


def _request(
    name: str = "inspect_model",
    *,
    budget: ToolBudget | None = None,
    approval_id: str | None = None,
) -> ToolRequest:
    definition = DEFAULT_TOOL_CATALOG.definition(name)
    assert definition is not None
    if name == "execute_approved_plan":
        input: dict[str, object] = {
            "plan_id": "plan.fixture",
            "plan_digest": "sha256:" + "a" * 64,
            "approval_id": approval_id or "approval.fixture",
        }
    else:
        input = {"model_ref": "fixture.model"}
    return ToolRequest.create(definition, input, budget=budget, approval_id=approval_id)


def _inspect_output() -> dict[str, object]:
    return {
        "model_ref": "fixture.model",
        "status": "supported",
        "capability_refs": [],
        "provenance_ref": "evidence.fixture",
    }


def test_untrusted_records_are_rejected_before_handler_lookup() -> None:
    calls = 0

    def handler(_context: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _inspect_output()

    request = _request()
    payload = request.to_record()
    assert isinstance(payload["input"], dict)
    payload["input"]["unexpected"] = "adversarial"
    dispatched = ToolDispatcher({"inspect_model": handler}).dispatch_record(payload)

    assert dispatched.request is None
    assert dispatched.result is None
    assert dispatched.negotiation.failure is not None
    assert dispatched.negotiation.failure.code is ToolFailureCode.INVALID_INPUT
    assert calls == 0

    oversized = request.to_record()
    oversized["input"] = {"model_ref": "x" * (1 << 20)}
    refused = ToolDispatcher({"inspect_model": handler}).dispatch_record(oversized)
    assert refused.result is None
    assert refused.negotiation.failure is not None
    assert refused.negotiation.failure.code is ToolFailureCode.INVALID_INPUT
    assert calls == 0


def test_unknown_tool_and_missing_handler_fail_closed() -> None:
    request = _request()
    unknown_budget = ToolBudget(1.0, 1024, 1, 1024)
    unknown = ToolRequest(
        deterministic_tool_request_id(
            "not_allowlisted", "not_allowlisted", {}, unknown_budget
        ),
        "not_allowlisted",
        "not_allowlisted",
        {},
        unknown_budget,
    )
    unknown_result = ToolDispatcher().dispatch(unknown)
    assert unknown_result.result is not None
    assert unknown_result.result.outcome is ToolOutcome.UNKNOWN
    assert unknown_result.result.failure is not None
    assert unknown_result.result.failure.code is ToolFailureCode.UNKNOWN_TOOL

    missing = ToolDispatcher().dispatch(request)
    assert missing.result is not None
    assert missing.result.outcome is ToolOutcome.UNSUPPORTED
    assert missing.result.failure is not None
    assert missing.result.failure.code is ToolFailureCode.HANDLER_UNAVAILABLE


def test_handler_receives_isolated_request_copy() -> None:
    request = _request()

    def handler(context: ToolExecutionContext) -> ToolExecutionResponse:
        context.request.input["model_ref"] = "provider-controlled.model"
        return ToolExecutionResponse(_inspect_output())

    dispatched = ToolDispatcher({"inspect_model": handler}).dispatch(request)
    assert dispatched.result is not None
    assert dispatched.result.outcome is ToolOutcome.SUPPORTED
    assert request.input["model_ref"] == "fixture.model"
    assert dispatched.result.provenance.request_digest == tool_request_digest(request)


def test_approval_is_bound_to_arguments_and_trusted_policy() -> None:
    request = _request("execute_approved_plan", approval_id="approval.fixture")
    definition = DEFAULT_TOOL_CATALOG.definition("execute_approved_plan")
    assert definition is not None
    mismatched = ToolRequest.create(
        definition,
        {
            "plan_id": "plan.fixture",
            "plan_digest": "sha256:" + "a" * 64,
            "approval_id": "approval.argument",
        },
        approval_id="approval.top_level",
    )
    mismatch = ToolDispatcher().dispatch(mismatched)
    assert mismatch.result is not None
    assert mismatch.result.failure is not None
    assert mismatch.result.failure.code is ToolFailureCode.APPROVAL_MISMATCH

    calls = 0

    def handler(context: ToolExecutionContext) -> dict[str, object]:
        nonlocal calls
        calls += 1
        context.transaction.commit()
        return {
            "run_id": "run.fixture",
            "outcome": "supported",
            "evidence_refs": [],
            "artifact_ref": "artifact.fixture",
        }

    without_policy = ToolDispatcher({"execute_approved_plan": handler}).dispatch(request)
    assert without_policy.result is not None
    assert without_policy.result.failure is not None
    assert without_policy.result.failure.code is ToolFailureCode.APPROVAL_REQUIRED
    assert calls == 0

    rejected = ToolDispatcher(
        {"execute_approved_plan": handler}, approval_policy=lambda _request: False
    ).dispatch(request)
    assert rejected.result is not None
    assert rejected.result.failure is not None
    assert rejected.result.failure.code is ToolFailureCode.APPROVAL_INVALID
    assert calls == 0

    accepted = ToolDispatcher(
        {"execute_approved_plan": handler}, approval_policy=lambda _request: True
    ).dispatch(request)
    assert accepted.result is not None
    assert accepted.result.outcome is ToolOutcome.SUPPORTED
    assert calls == 1


def test_budget_exhaustion_is_typed_and_handlers_can_charge_nested_work() -> None:
    def too_large(_context: object) -> dict[str, object]:
        return _inspect_output()

    output_limited = ToolDispatcher({"inspect_model": too_large}).dispatch(
        _request(budget=ToolBudget(5.0, 1024, 1, 1))
    )
    assert output_limited.result is not None
    assert output_limited.result.failure is not None
    assert output_limited.result.failure.code is ToolFailureCode.BUDGET_EXCEEDED

    def too_many(context: object) -> dict[str, object]:
        assert hasattr(context, "consume_evaluations")
        context.consume_evaluations()  # type: ignore[attr-defined]
        return _inspect_output()

    evaluation_limited = ToolDispatcher({"inspect_model": too_many}).dispatch(
        _request(budget=ToolBudget(5.0, 1024, 1, 64 * 1024))
    )
    assert evaluation_limited.result is not None
    assert evaluation_limited.result.failure is not None
    assert evaluation_limited.result.failure.code is ToolFailureCode.BUDGET_EXCEEDED

    def too_much_memory(context: object) -> dict[str, object]:
        assert hasattr(context, "reserve_memory")
        context.reserve_memory(2048)  # type: ignore[attr-defined]
        return _inspect_output()

    memory_limited = ToolDispatcher({"inspect_model": too_much_memory}).dispatch(
        _request(budget=ToolBudget(5.0, 1024, 1, 64 * 1024))
    )
    assert memory_limited.result is not None
    assert memory_limited.result.failure is not None
    assert memory_limited.result.failure.code is ToolFailureCode.BUDGET_EXCEEDED


def test_timeout_and_cancellation_do_not_publish_success() -> None:
    started = threading.Event()

    def slow(_context: object) -> dict[str, object]:
        started.set()
        time.sleep(0.05)
        return _inspect_output()

    timed_out = ToolDispatcher({"inspect_model": slow}).dispatch(
        _request(budget=ToolBudget(0.01, 1024, 1, 64 * 1024))
    )
    assert started.wait(1)
    assert timed_out.result is not None
    assert timed_out.result.outcome is ToolOutcome.TIMEOUT
    assert timed_out.result.failure is not None
    assert timed_out.result.failure.code is ToolFailureCode.TIMEOUT

    cancellation = ToolCancellationToken()
    cancellation.cancel()
    cancelled = ToolDispatcher({"inspect_model": slow}).dispatch(
        _request(), cancellation=cancellation
    )
    assert cancelled.result is not None
    assert cancelled.result.outcome is ToolOutcome.CANCELLED
    assert cancelled.result.failure is not None
    assert cancelled.result.failure.code is ToolFailureCode.CANCELLED


def test_replay_is_idempotent_and_retry_count_is_bounded() -> None:
    calls = 0

    def handler(_context: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _inspect_output()

    dispatcher = ToolDispatcher({"inspect_model": handler})
    first = dispatcher.dispatch(_request())
    replay = dispatcher.dispatch(_request())
    assert first.result is not None and replay.result is not None
    assert first.result.result_id == replay.result.result_id
    assert replay.receipt is not None and replay.receipt.replayed
    assert calls == 1

    attempts = 0

    def flaky(_context: object) -> ToolExecutionResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ToolExecutionError(
                ToolFailureCode.EXECUTION_FAILED,
                "transient engine failure",
                retryable=True,
            )
        return ToolExecutionResponse(
            _inspect_output(),
            ToolEvidenceStatus.CANONICAL,
            source_digest="sha256:" + "b" * 64,
            evidence_id="evidence.fixture",
            observed_at="2026-09-07T12:00:00Z",
        )

    retried = ToolDispatcher({"inspect_model": flaky}, max_retries=1).dispatch(_request())
    assert retried.result is not None
    assert retried.result.outcome is ToolOutcome.SUPPORTED
    assert retried.receipt is not None and retried.receipt.attempts == 2
    assert retried.result.provenance.request_digest == tool_request_digest(_request())

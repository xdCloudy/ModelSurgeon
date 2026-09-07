"""Contract tests for the capability-scoped conversational tool boundary."""

from __future__ import annotations

import hashlib

import pytest

from modelsurgeon.conversation import (
    DEFAULT_TOOL_CATALOG,
    ToolAccess,
    ToolBudget,
    ToolCapability,
    ToolContractError,
    ToolDefinition,
    ToolFailureCode,
    ToolName,
    ToolOutcome,
    ToolProvenance,
    ToolRequest,
    ToolResult,
    deterministic_tool_request_id,
)
from modelsurgeon.experiments.identity import canonical_identity_json


def _request_without_definition(
    name: str, capability: str, input: dict[str, object], *, tool_id: str | None = None
) -> ToolRequest:
    budget = ToolBudget(1.0, 1024, 1, 1024)
    request_id = deterministic_tool_request_id(
        name, capability, input, budget, tool_id=tool_id
    )
    return ToolRequest(request_id, name, capability, input, budget, tool_id)


def test_every_allowlisted_tool_has_a_complete_deterministic_contract() -> None:
    definitions = DEFAULT_TOOL_CATALOG.definitions
    assert [item.name for item in definitions] == list(ToolName)
    assert len({item.tool_id for item in definitions}) == len(definitions)
    for definition in definitions:
        assert definition.owner
        assert definition.capability.value == definition.name.value
        assert definition.budget.max_wall_seconds > 0
        assert ToolOutcome.SUPPORTED not in definition.failure_semantics
        assert set(definition.failure_semantics) == {
            ToolOutcome.CANCELLED,
            ToolOutcome.FAILED,
            ToolOutcome.REFUSED,
            ToolOutcome.TIMEOUT,
            ToolOutcome.UNKNOWN,
            ToolOutcome.UNSUPPORTED,
        }
        assert definition.to_record() == definition.to_record()
        assert "tensor" not in definition.to_record().__repr__().lower()
        assert "command" not in definition.to_record().__repr__().lower()


def test_request_ids_are_replay_stable_and_round_trip() -> None:
    definition = DEFAULT_TOOL_CATALOG.definition("inspect_model")
    assert definition is not None
    request = ToolRequest.create(definition, {"model_ref": "fixture.model"})
    replay = ToolRequest.create(definition, {"model_ref": "fixture.model"})
    assert request.request_id == replay.request_id
    assert ToolRequest.from_record(request.to_record()) == request
    assert request.request_id.startswith("tool_request_")

    changed = ToolRequest.create(definition, {"model_ref": "fixture.other"})
    assert changed.request_id != request.request_id


def test_unknown_schema_version_is_refused_before_execution() -> None:
    definition = DEFAULT_TOOL_CATALOG.definition("inspect_model")
    assert definition is not None
    payload = ToolRequest.create(definition, {"model_ref": "fixture.model"}).to_record()
    payload["schema_version"] = 99

    negotiation = DEFAULT_TOOL_CATALOG.negotiate_record(payload)
    assert negotiation.outcome is ToolOutcome.REFUSED
    assert negotiation.failure is not None
    assert negotiation.failure.code is ToolFailureCode.UNKNOWN_SCHEMA_VERSION
    assert not hasattr(DEFAULT_TOOL_CATALOG, "execute")

    with pytest.raises(ToolContractError, match="unsupported"):
        ToolRequest.from_record(payload)


def test_unknown_and_wrong_capability_requests_fail_closed() -> None:
    unknown = _request_without_definition("not_allowlisted", "not_allowlisted", {})
    unknown_result = DEFAULT_TOOL_CATALOG.negotiate(unknown)
    assert unknown_result.outcome is ToolOutcome.UNKNOWN
    assert unknown_result.failure is not None
    assert unknown_result.failure.code is ToolFailureCode.UNKNOWN_TOOL

    definition = DEFAULT_TOOL_CATALOG.definition("inspect_model")
    assert definition is not None
    wrong_capability = _request_without_definition(
        definition.name.value, "query_evidence", {"model_ref": "fixture.model"}
    )
    result = DEFAULT_TOOL_CATALOG.negotiate(wrong_capability)
    assert result.outcome is ToolOutcome.UNSUPPORTED
    assert result.failure is not None
    assert result.failure.code is ToolFailureCode.UNSUPPORTED_CAPABILITY


def test_schemas_reject_unknown_fields_and_direct_tensor_authority() -> None:
    definition = DEFAULT_TOOL_CATALOG.definition("inspect_model")
    assert definition is not None
    extra_field = _request_without_definition(
        definition.name.value,
        definition.capability.value,
        {"model_ref": "fixture.model", "unexpected": True},
        tool_id=definition.tool_id,
    )
    result = DEFAULT_TOOL_CATALOG.negotiate(extra_field)
    assert result.outcome is ToolOutcome.REFUSED
    assert result.failure is not None
    assert result.failure.code is ToolFailureCode.INVALID_INPUT

    safe_output = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    with pytest.raises(ToolContractError, match="forbidden"):
        ToolDefinition(
            ToolName.INSPECT_MODEL,
            "modelsurgeon.test",
            ToolCapability.INSPECT_MODEL,
            ToolAccess.READ_ONLY,
            {
                "type": "object",
                "properties": {"tensor_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["tensor_ids"],
                "additionalProperties": False,
            },
            safe_output,
            ToolBudget(1.0, 1024, 1, 1024),
        )


def test_consequential_tool_requires_approval_but_carries_no_tensor_selector() -> None:
    definition = DEFAULT_TOOL_CATALOG.definition("execute_approved_plan")
    assert definition is not None
    input = {
        "plan_id": "plan.fixture",
        "plan_digest": "sha256:" + "a" * 64,
        "approval_id": "approval.fixture",
    }
    without_approval = ToolRequest.create(definition, input)
    refused = DEFAULT_TOOL_CATALOG.negotiate(without_approval)
    assert refused.outcome is ToolOutcome.REFUSED
    assert refused.failure is not None
    assert refused.failure.code is ToolFailureCode.APPROVAL_REQUIRED

    approved = ToolRequest.create(definition, input, approval_id="approval.fixture")
    accepted = DEFAULT_TOOL_CATALOG.negotiate(approved)
    assert accepted.outcome is ToolOutcome.SUPPORTED
    assert accepted.tool == definition
    assert definition.access is ToolAccess.CONSEQUENTIAL
    assert definition.approval_required


def test_typed_result_retains_provenance_and_failure_identity() -> None:
    definition = DEFAULT_TOOL_CATALOG.definition("inspect_model")
    assert definition is not None
    request = ToolRequest.create(definition, {"model_ref": "fixture.model"})
    request_digest = "sha256:" + hashlib.sha256(
        canonical_identity_json(request.to_record()).encode()
    ).hexdigest()
    provenance = ToolProvenance(definition.owner, definition.tool_id, request_digest)
    output = {
        "model_ref": "fixture.model",
        "status": "supported",
        "capability_refs": [],
        "provenance_ref": "evidence.fixture",
    }
    definition.validate_output(output)
    result = ToolResult(
        request.request_id,
        request.name,
        ToolOutcome.SUPPORTED,
        provenance,
        output,
    )
    assert result.canonical_json() == result.canonical_json()
    assert result.to_record()["outcome"] == "supported"

    with pytest.raises(ToolContractError, match="failure"):
        ToolResult(
            request.request_id,
            request.name,
            ToolOutcome.FAILED,
            provenance,
        )

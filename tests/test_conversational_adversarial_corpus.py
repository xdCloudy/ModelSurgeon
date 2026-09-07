"""Deterministic adversarial corpus for the conversational tool boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modelsurgeon.conversation import (
    DEFAULT_TOOL_CATALOG,
    ToolAccess,
    ToolBudget,
    ToolCapability,
    ToolContractError,
    ToolDefinition,
    ToolDispatcher,
    ToolEvidenceStatus,
    ToolExecutionError,
    ToolExecutionResponse,
    ToolFailureCode,
    ToolName,
    ToolOutcome,
    ToolRequest,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = json.loads(
    (ROOT / "tests" / "fixtures" / "conversational_tool_boundary_adversarial_v1.json").read_text(
        encoding="utf-8"
    )
)


def _request() -> ToolRequest:
    definition = DEFAULT_TOOL_CATALOG.definition("inspect_model")
    assert definition is not None
    return ToolRequest.create(definition, {"model_ref": "fixture.model"})


def _output() -> dict[str, Any]:
    return {
        "model_ref": "fixture.model",
        "status": "supported",
        "capability_refs": [],
        "provenance_ref": "evidence.fixture",
    }


def _request_payload(case: dict[str, Any]) -> dict[str, Any]:
    payload = _request().to_record()
    payload.update(case.get("top_level_patch", {}))
    input_payload = dict(payload["input"])
    input_payload.update(case.get("input_patch", {}))
    input_payload.update(case.get("input_replace", {}))
    payload["input"] = input_payload
    return payload


def test_corpus_metadata_and_threat_classes_are_complete() -> None:
    assert CORPUS["schema_version"] == 1
    assert CORPUS["corpus_revision"] == "v2.6-adversarial-v1"
    cases = CORPUS["cases"]
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert {case["threat_class"] for case in cases} >= {
        "shell_code_request",
        "path_traversal",
        "secret_exfiltration",
        "prompt_injection_model_metadata",
        "malformed_schema",
        "forged_measurement",
        "contradictory_measurement",
        "replayed_identifier",
    }


def test_request_attacks_fail_before_the_trusted_handler() -> None:
    cases = [case for case in CORPUS["cases"] if case["route"] == "request"]
    calls = 0

    def handler(_context: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _output()

    dispatcher = ToolDispatcher({"inspect_model": handler})
    for case in cases:
        dispatched = dispatcher.dispatch_record(_request_payload(case))
        assert dispatched.negotiation.outcome.value == case["expected_outcome"], case["case_id"]
        assert dispatched.negotiation.failure is not None, case["case_id"]
        assert (
            dispatched.negotiation.failure.code.value == case["expected_failure_code"]
        ), case["case_id"]
        assert dispatched.request is None
        assert dispatched.result is None
    assert calls == 0


def test_forged_and_contradictory_measurements_cannot_become_canonical() -> None:
    cases = {case["case_id"]: case for case in CORPUS["cases"] if case["route"] == "response"}

    def forged(_context: object) -> ToolExecutionResponse:
        case = cases["forged-measurement-is-unverified"]
        return ToolExecutionResponse(
            _output(),
            ToolEvidenceStatus(case["response"]["evidence_status"]),
        )

    unverified = ToolDispatcher({"inspect_model": forged}).dispatch(_request())
    assert unverified.result is not None
    assert unverified.result.outcome is ToolOutcome.SUPPORTED
    assert unverified.result.provenance.evidence_status is ToolEvidenceStatus.UNVERIFIED
    assert unverified.result.provenance.source_digest is None
    assert unverified.result.provenance.evidence_id is None

    def contradictory(_context: object) -> ToolExecutionResponse:
        case = cases["contradictory-canonical-measurement"]
        return ToolExecutionResponse(
            _output(),
            ToolEvidenceStatus(case["response"]["evidence_status"]),
        )

    rejected = ToolDispatcher({"inspect_model": contradictory}).dispatch(_request())
    assert rejected.result is not None
    assert rejected.result.outcome is ToolOutcome.FAILED
    assert rejected.result.failure is not None
    assert rejected.result.failure.code is ToolFailureCode.OUTPUT_INVALID


def test_replay_case_returns_the_original_result_without_reinvoking_the_handler() -> None:
    calls = 0

    def handler(_context: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _output()

    dispatcher = ToolDispatcher({"inspect_model": handler})
    first = dispatcher.dispatch(_request())
    replay = dispatcher.dispatch(_request())
    assert first.result is not None and replay.result is not None
    assert first.result.result_id == replay.result.result_id
    assert replay.receipt is not None
    assert replay.receipt.replayed is CORPUS["cases"][-1]["expected_replayed"]
    assert calls == 1


def test_failure_diagnostics_are_retained_but_secret_shaped_values_are_redacted() -> None:
    def failing(_context: object) -> ToolExecutionResponse:
        raise ToolExecutionError(
            ToolFailureCode.EXECUTION_FAILED,
            "provider api_key=live-secret encountered authorization: Bearer-live-token",
            raw_payload={
                "error": "secret=raw-secret",
                "api_key": "live-secret",
                "outcome": "supported",
            },
        )

    dispatched = ToolDispatcher({"inspect_model": failing}).dispatch(_request())
    assert dispatched.result is not None
    assert dispatched.result.outcome is ToolOutcome.FAILED
    assert dispatched.result.failure is not None
    assert "live-secret" not in dispatched.result.failure.detail
    assert "Bearer-live-token" not in dispatched.result.failure.detail
    assert dispatched.result.raw_payload == {
        "error": "secret=<redacted>",
        "api_key": "<redacted>",
        "outcome": "supported",
    }


def test_malformed_tool_schemas_are_rejected_before_registration() -> None:
    output_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    malformed_inputs = (
        {
            "type": "object",
            "properties": {"value": {"type": "string", "minLength": 1.5}},
            "required": ["value"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"value": {"type": "string", "pattern": "["}},
            "required": ["value"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"value": {"type": "string", "minLength": 4, "maxLength": 2}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )
    for input_schema in malformed_inputs:
        try:
            ToolDefinition(
                ToolName.INSPECT_MODEL,
                "modelsurgeon.test",
                ToolCapability.INSPECT_MODEL,
                ToolAccess.READ_ONLY,
                input_schema,
                output_schema,
                ToolBudget(1.0, 1024, 1, 1024),
            )
        except ToolContractError:
            continue
        raise AssertionError("malformed schema was accepted")

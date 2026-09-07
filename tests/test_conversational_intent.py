"""Canonical conversational intent record tests."""

from __future__ import annotations

import json

import pytest

from modelsurgeon.conversation import (
    AmbiguityRecord,
    IntentField,
    IntentOutcome,
    IntentProvenance,
    IntentRecord,
    IntentRecordError,
    InterpretationStep,
    SourceSpan,
)

REQUEST = "reduce latency while retaining at least 0.98 quality"
PROVENANCE = IntentProvenance(
    "conversation-request-v1",
    "local-provider",
    "provider-revision-1",
    "tool-revision-1",
    ("evidence:request",),
)


def _record(*, outcome: IntentOutcome = IntentOutcome.EXECUTABLE) -> IntentRecord:
    spans = (
        SourceSpan("span-quality", 21, 44, "retaining at least 0.98"),
        SourceSpan("span-request", 0, len(REQUEST), REQUEST),
    )
    fields = (
        IntentField("objective.metric", "latency", None, 0.99, ("span-request",)),
        IntentField("quality.minimum", 0.98, "ratio", 0.99, ("span-quality",), True),
    )
    ambiguities = ()
    emitted = {"contract_id": "objective_contract_example", "schema_version": 1}
    return IntentRecord(
        REQUEST,
        spans,
        fields,
        ambiguities,
        (
            InterpretationStep(
                "step-normalize",
                "normalize",
                ("span-request",),
                ("objective.metric", "quality.minimum"),
                0.99,
                ("evidence:request",),
            ),
        ),
        PROVENANCE,
        outcome,
        ("request normalized without provider policy",),
        emitted if outcome is IntentOutcome.EXECUTABLE else None,
    )


def test_canonical_json_is_byte_stable_and_round_trips() -> None:
    record = _record()
    payload = record.canonical_json()
    assert payload == record.canonical_json()
    assert IntentRecord.from_json(payload).canonical_json() == payload
    decoded = json.loads(payload)
    assert decoded["request_digest"] == record.request_digest
    assert decoded["spec_digest"] == record.spec_digest
    assert decoded["intent_id"] == record.intent_id


def test_original_text_and_provenance_are_not_authoritative_policy() -> None:
    record = _record()
    assert record.original_request == REQUEST
    assert record.emitted_spec == {
        "contract_id": "objective_contract_example",
        "schema_version": 1,
    }
    assert record.provenance.provider_id == "local-provider"
    assert record.intent_id == IntentRecord(
        record.original_request,
        record.source_spans,
        record.fields,
        record.ambiguities,
        record.interpretation_steps,
        IntentProvenance(
            "conversation-request-v1",
            "different-provider",
            "provider-revision-2",
            "tool-revision-2",
            ("evidence:request",),
        ),
        record.outcome,
        record.diagnostics,
        record.emitted_spec,
    ).intent_id


def test_unknown_schema_and_tampered_span_are_rejected() -> None:
    record = _record()
    root = json.loads(record.canonical_json())
    root["schema_version"] = 99
    with pytest.raises(IntentRecordError, match="unsupported"):
        IntentRecord.from_json(json.dumps(root))

    root = json.loads(record.canonical_json())
    root["source_spans"][0]["text"] = "retaining at least 0.97"
    with pytest.raises(IntentRecordError, match="span text"):
        IntentRecord.from_json(json.dumps(root))


def test_non_executable_record_cannot_emit_a_spec() -> None:
    record = _record(outcome=IntentOutcome.CLARIFICATION_REQUIRED)
    assert record.spec_digest is None
    with pytest.raises(IntentRecordError, match="non-executable"):
        IntentRecord(
            record.original_request,
            record.source_spans,
            record.fields,
            (
                AmbiguityRecord(
                    "ambiguity-quality",
                    "quality.minimum",
                    "missing",
                    "minimum quality was not stated",
                    ("ask user",),
                ),
            ),
            record.interpretation_steps,
            record.provenance,
            IntentOutcome.CLARIFICATION_REQUIRED,
            ("clarification required",),
            {"unsafe": True},
        )

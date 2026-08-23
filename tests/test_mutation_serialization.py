"""Tests for canonical mutation plan, outcome, and mapping records."""

from __future__ import annotations

import json

import pytest

from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery import (
    REDACTED_LOCAL_PATH,
    MutationDelta,
    MutationIdentityMapping,
    MutationKind,
    MutationOutcome,
    MutationOutcomeStatus,
    MutationPlan,
    MutationPrecondition,
    MutationProvenance,
    MutationRecordError,
    MutationRequest,
    MutationRunRecord,
)

OLD = ComponentId.parse("model.layers.2")
NEW = ComponentId.parse("model.layers.1")


def _record() -> MutationRunRecord:
    request = MutationRequest(
        MutationKind.REMOVE,
        (OLD,),
        (("index", 2), ("strategy", "physical")),
    )
    plan = MutationPlan(
        request,
        (OLD,),
        (MutationPrecondition("input_revision", "a" * 40),),
        MutationDelta(parameters=-1024, flops=-2048, storage_bytes=-4096),
    )
    outcome = MutationOutcome(
        MutationOutcomeStatus.APPLIED,
        MutationDelta(parameters=-1024, flops=-2048, storage_bytes=-4000),
        (MutationIdentityMapping(OLD, (NEW,), "layer renumbered"),),
    )
    return MutationRunRecord(
        plan,
        MutationProvenance("a" * 40, "modelsurgeon-0.1", r"C:\private\model.gguf"),
        outcome,
    )


def test_canonical_round_trip_preserves_mutation_identity_and_outcome() -> None:
    original = _record()
    payload = original.canonical_json(redact_local_paths=False)
    restored = MutationRunRecord.from_json(payload)

    assert restored == original
    assert restored.mutation_id == original.mutation_id
    assert restored.canonical_json(redact_local_paths=False) == payload
    assert restored.outcome is not None
    assert restored.outcome.identity_mappings[0].targets == (NEW,)


def test_sensitive_local_paths_are_redacted_by_default() -> None:
    record = _record()
    redacted = MutationRunRecord.from_json(record.canonical_json())
    assert redacted.provenance.input_path == REDACTED_LOCAL_PATH
    assert "private" not in record.canonical_json()
    assert "private" in record.canonical_json(redact_local_paths=False)
    assert redacted.mutation_id == record.mutation_id


def test_removed_identity_mapping_round_trips_without_silent_retargeting() -> None:
    mapping = MutationIdentityMapping(OLD, (), "component removed")
    assert mapping.removed is True
    assert mapping.to_record()["targets"] == []


def test_tampered_identity_and_mapping_flags_fail_closed() -> None:
    value = json.loads(_record().canonical_json())
    value["mutation_id"] = "0" * 64
    with pytest.raises(MutationRecordError, match="identity does not match"):
        MutationRunRecord.from_json(json.dumps(value))

    value = json.loads(_record().canonical_json())
    value["outcome"]["identity_mappings"][0]["removed"] = True
    with pytest.raises(MutationRecordError, match="removed flag"):
        MutationRunRecord.from_json(json.dumps(value))


def test_unknown_fields_and_applied_outcome_without_delta_are_rejected() -> None:
    value = json.loads(_record().canonical_json())
    value["unexpected"] = True
    with pytest.raises(MutationRecordError, match="unknown fields"):
        MutationRunRecord.from_json(json.dumps(value))
    with pytest.raises(MutationRecordError, match="require actual deltas"):
        MutationOutcome(MutationOutcomeStatus.APPLIED, None)

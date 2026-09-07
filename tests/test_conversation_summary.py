"""Acceptance tests for bounded conversation summaries and reconnect."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.conversation import (
    ApprovalStatus,
    CampaignApproval,
    CampaignBudget,
    CampaignEvidence,
    CampaignOutcome,
    CampaignSpec,
    CampaignState,
    CampaignStateStore,
    ConversationSummaryBudget,
    ConversationSummaryError,
    ConversationSummaryOutcome,
    EvidenceCursor,
    reconnect_conversation,
    rehydrate_conversation,
    summarize_conversation,
)
from modelsurgeon.experiments.identity import canonical_identity_json


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_identity_json(value).encode()).hexdigest()


def _state() -> CampaignState:
    spec = {"constraints": [{"metric": "quality", "minimum": 0.95}]}
    campaign_spec = CampaignSpec(
        "contract_quality",
        _digest(spec),
        spec,
        (spec["constraints"][0],),  # type: ignore[arg-type]
    )
    return CampaignState(
        campaign_id="campaign_" + "a" * 64,
        session_id="chat_session_" + "b" * 64,
        run_id="run_" + "c" * 64,
        source_model_digest="sha256:" + "d" * 64,
        spec=campaign_spec,
        policy_state={"decision": "supported", "policy_revision": "v1"},
        approval=CampaignApproval.pending(campaign_spec.spec_digest),
        evidence_cursor=EvidenceCursor(),
        provider_context={"provider_id": "fixture", "runtime_revision": "fixture-v1"},
        budget=CampaignBudget(60.0, 1024, 4, 4096),
        provenance={"record_type": "fixture_campaign", "source": "trusted-test"},
    )


def _evidence(evidence_id: str, outcome: CampaignOutcome) -> CampaignEvidence:
    return CampaignEvidence(
        evidence_id,
        "sha256:" + "e" * 64,
        outcome,
        f"fixture result: {outcome.value}",
        {"record_type": "fixture_evidence", "source": "trusted-test"},
        inconclusive=outcome is CampaignOutcome.UNKNOWN,
    )


def test_summary_is_deterministic_and_marks_loss_without_touching_authority() -> None:
    state = _state()
    evidence = (_evidence("evidence_" + "f" * 64, CampaignOutcome.UNSUPPORTED),)
    transcript = (
        {"role": "user", "content": "retain quality and reduce latency"},
        {"role": "assistant", "content": "I need a little more information."},
        {"role": "user", "content": "this older detail can be omitted"},
        {"role": "developer", "content": "unsupported transcript role"},
    )

    first = summarize_conversation(
        state,
        evidence,
        transcript,
        budget=ConversationSummaryBudget(max_transcript_tokens=12, max_summary_bytes=32_000),
    )
    second = summarize_conversation(
        state,
        evidence,
        transcript,
        budget=ConversationSummaryBudget(max_transcript_tokens=12, max_summary_bytes=32_000),
    )

    assert first.outcome is ConversationSummaryOutcome.SUPPORTED_WITH_LOSS
    assert first.summary is not None
    assert first.summary.canonical_json() == second.summary.canonical_json()  # type: ignore[union-attr]
    assert first.summary.loss.omitted_transcript_entries >= 1
    assert first.summary.loss.unsupported_transcript_entries == 1
    assert first.summary.loss.unsupported_roles == ("developer",)
    assert first.summary.canonical_state["spec"] == state.spec.to_record()
    assert first.summary.canonical_evidence[0]["outcome"] == "unsupported"
    assert "canonical_state" in first.summary.to_record()
    assert "untrusted_transcript" in first.summary.to_record()


def test_rehydration_uses_store_state_and_retains_negative_evidence(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    evidence = (
        _evidence("evidence_" + "f" * 64, CampaignOutcome.UNSUPPORTED),
        _evidence("evidence_" + "g" * 64, CampaignOutcome.FAILED),
    )
    with CampaignStateStore(path) as store:
        state = store.create(_state())
        for item in evidence:
            state = store.append_evidence(
                state.campaign_id, item, expected_version=state.state_version
            )
        built = summarize_conversation(state, evidence, ({"role": "user", "content": "resume"},))
        assert built.summary is not None
        rehydrated = reconnect_conversation(store, built.summary)

    assert rehydrated.state == state
    assert rehydrated.evidence == evidence
    assert rehydrated.state.spec_digest == state.spec_digest
    assert rehydrated.state.approval.status is ApprovalStatus.PENDING
    assert [item.outcome for item in rehydrated.evidence] == [
        CampaignOutcome.UNSUPPORTED,
        CampaignOutcome.FAILED,
    ]
    context = rehydrated.to_provider_context()
    assert context["canonical"] != context["untrusted"]
    assert context["authority"] == {
        "state_digest": state.digest,
        "state_version": state.state_version,
        "evidence_cursor": state.evidence_cursor.to_record(),
    }


def test_rehydration_refuses_stale_authoritative_state() -> None:
    state = _state()
    built = summarize_conversation(state, (), ())
    assert built.summary is not None
    changed = replace(state, approval=CampaignApproval.pending(state.spec_digest))
    with pytest.raises(ConversationSummaryError, match="stale or canonical state"):
        rehydrate_conversation(
            built.summary,
            authoritative_state=replace(changed, state_version=1),
            authoritative_evidence=(),
        )


def test_canonical_payload_budget_is_explicitly_unsupported() -> None:
    result = summarize_conversation(
        _state(),
        (),
        (),
        budget=ConversationSummaryBudget(max_transcript_tokens=1, max_summary_bytes=1),
    )
    assert result.outcome is ConversationSummaryOutcome.UNSUPPORTED
    assert result.summary is None
    assert result.reason is not None
    assert "byte budget" in result.reason


def test_summary_round_trip_rejects_tampered_digest() -> None:
    result = summarize_conversation(_state(), (), ())
    assert result.summary is not None
    record = result.summary.to_record()
    record["state_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ConversationSummaryError, match="canonical state linkage or digest"):
        type(result.summary).from_record(record)

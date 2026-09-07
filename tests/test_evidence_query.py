"""Acceptance tests for the canonical conversational evidence query path."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.conversation import (
    CampaignBudget,
    CampaignEvidence,
    CampaignOutcome,
    CampaignSpec,
    CampaignStateStore,
    EvidenceCursor,
    EvidenceQuery,
    EvidenceQueryAccessError,
    EvidenceQueryEngine,
    EvidenceQueryIntegrityError,
    EvidenceQueryLimits,
    EvidenceQueryOutcome,
    EvidenceQueryResourceError,
    EvidenceQueryStaleError,
    EvidenceSnapshot,
    direct_evidence_report,
    new_campaign_state,
)
from modelsurgeon.experiments.identity import canonical_identity_json


def _digest(value: object) -> str:
    return (
        "sha256:" + hashlib.sha256(canonical_identity_json(value).encode()).hexdigest()
    )


def _state() -> object:
    payload = {"constraints": [{"metric": "quality", "minimum": 0.95}]}
    spec = CampaignSpec(
        "contract_fixture",
        _digest(payload),
        payload,
        (payload["constraints"][0],),  # type: ignore[arg-type]
    )
    return new_campaign_state(
        session_id="chat_session_fixture",
        run_id="run_fixture",
        source_model_digest="sha256:" + "a" * 64,
        spec=spec,
        policy_state={"record_type": "fixture_policy", "revision": "v1"},
        provider_context={"provider_id": "fixture"},
        budget=CampaignBudget(60.0, 4096, 4, 8192),
        campaign_id="campaign_fixture",
    )


def _evidence(
    evidence_id: str,
    outcome: CampaignOutcome,
    *,
    decision: str | None = None,
    measured: bool = False,
) -> CampaignEvidence:
    provenance: dict[str, object] = {
        "record_type": "fixture_evidence",
        "source": "trusted-test",
        "run_id": "run_fixture",
        "plan_id": "plan_fixture",
    }
    if decision is not None:
        provenance["decision"] = decision
    if measured:
        provenance["measurements"] = {
            "quality": {
                "value": 0.97,
                "unit": "score",
                "uncertainty": {
                    "lower_bound": 0.96,
                    "upper_bound": 0.98,
                    "confidence": 0.95,
                    "sample_count": 3,
                },
            }
        }
        provenance["observed_at"] = "2026-09-07T12:00:00Z"
    return CampaignEvidence(
        evidence_id,
        "sha256:" + "a" * 64,
        outcome,
        "fixture evidence: " + outcome.value,
        provenance,
        artifact_digest="sha256:" + "b" * 64 if decision == "accepted" else None,
        inconclusive=outcome is CampaignOutcome.UNKNOWN,
    )


def _snapshot() -> EvidenceSnapshot:
    state = _state()
    evidence = (
        _evidence(
            "evidence_accepted",
            CampaignOutcome.SUPPORTED,
            decision="accepted",
            measured=True,
        ),
        _evidence(
            "evidence_rolled_back", CampaignOutcome.FAILED, decision="rolled_back"
        ),
        _evidence("evidence_unsupported", CampaignOutcome.UNSUPPORTED),
    )
    state = replace(
        state,  # type: ignore[arg-type]
        evidence_cursor=EvidenceCursor(len(evidence), evidence[-1].evidence_id),
        state_version=len(evidence),
    )
    return EvidenceSnapshot(
        state,  # type: ignore[arg-type]
        evidence,
    )


def test_query_is_deterministic_and_direct_report_parity_is_exact() -> None:
    snapshot = _snapshot()
    query = EvidenceQuery(
        snapshot.campaign.campaign_id,
        fields=(
            "artifact_digest",
            "detail",
            "measurements",
            "outcome",
            "provenance_refs",
        ),
        expected_snapshot_id=snapshot.snapshot_id,
        expected_state_digest=snapshot.state_digest,
    )
    first = EvidenceQueryEngine(snapshot).query(query)
    second = direct_evidence_report(snapshot, query)

    assert first.query.query_id == second.query.query_id
    assert first.canonical_json() == second.canonical_json()
    assert first.response_digest == second.response_digest
    assert [item.outcome for item in first.records] == [
        EvidenceQueryOutcome.ACCEPTED,
        EvidenceQueryOutcome.ROLLED_BACK,
        EvidenceQueryOutcome.UNSUPPORTED,
    ]
    assert first.records[0].measurements[0].uncertainty is not None
    assert first.records[0].measurements[0].value == 0.97


def test_query_joins_campaign_state_and_provenance_without_raw_provider_data() -> None:
    snapshot = _snapshot()
    query = EvidenceQuery(
        snapshot.campaign.campaign_id,
        evidence_ids=("evidence_accepted",),
        fields=(
            "campaign_id",
            "provenance_refs",
            "run_id",
            "source_digest",
            "spec_digest",
        ),
    )
    response = EvidenceQueryEngine(snapshot).query(query)
    record = response.records[0]
    assert record.campaign_id == snapshot.campaign.campaign_id
    assert record.fields["run_id"] == snapshot.campaign.run_id
    assert record.fields["spec_digest"] == snapshot.campaign.spec_digest
    assert "run_fixture" in record.provenance_refs
    assert "provider_context" not in response.canonical_json()
    assert "transcript" not in response.canonical_json()


def test_missing_measurements_are_explicit_and_never_inferred() -> None:
    snapshot = _snapshot()
    query = EvidenceQuery(
        snapshot.campaign.campaign_id, fields=("measurements", "uncertainty")
    )
    response = EvidenceQueryEngine(snapshot).query(query)
    assert response.status.value == "partial"
    assert "measurements" in response.missing_fields
    assert response.records[1].measurements == ()
    assert response.records[1].outcome is EvidenceQueryOutcome.ROLLED_BACK


def test_query_current_rejects_a_stale_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    snapshot = _snapshot()
    with CampaignStateStore(path) as store:
        state = store.create(_state())
        for item in snapshot.evidence:
            state = store.append_evidence(
                state.campaign_id,
                item,
                expected_version=state.state_version,
            )
        current_snapshot = EvidenceSnapshot.from_store(store, state.campaign_id)
        engine = EvidenceQueryEngine(current_snapshot)
        store.append_evidence(
            state.campaign_id,
            _evidence("evidence_new", CampaignOutcome.UNKNOWN),
            expected_version=store.load(state.campaign_id).state_version,
        )
        with pytest.raises(EvidenceQueryStaleError, match="stale"):
            engine.query_current(
                store, EvidenceQuery(current_snapshot.campaign.campaign_id)
            )


def test_snapshot_digest_tampering_fails_closed() -> None:
    snapshot = _snapshot()
    record = snapshot.to_record()
    record["state_digest"] = "sha256:" + "0" * 64
    with pytest.raises(EvidenceQueryIntegrityError, match="digest"):
        EvidenceSnapshot.from_record(record)


def test_query_rejects_arbitrary_file_and_llm_access() -> None:
    with pytest.raises(EvidenceQueryAccessError):
        EvidenceQuery("campaign_fixture", fields=("file_path",))
    with pytest.raises(EvidenceQueryAccessError):
        EvidenceQuery("campaign_fixture", fields=("llm_answer",))


def test_query_resource_limits_are_hard() -> None:
    snapshot = _snapshot()
    with pytest.raises(EvidenceQueryResourceError):
        EvidenceQueryEngine(snapshot, limits=EvidenceQueryLimits(max_records=1)).query(
            EvidenceQuery(snapshot.campaign.campaign_id, max_records=2)
        )
    with pytest.raises(EvidenceQueryResourceError):
        EvidenceQueryEngine(
            snapshot, limits=EvidenceQueryLimits(max_output_bytes=128)
        ).query(EvidenceQuery(snapshot.campaign.campaign_id))


def test_snapshot_round_trip_preserves_identity() -> None:
    snapshot = _snapshot()
    restored = EvidenceSnapshot.from_record(snapshot.to_record())
    assert restored.to_record() == snapshot.to_record()
    assert replace(restored.campaign, state_version=1).digest != restored.state_digest

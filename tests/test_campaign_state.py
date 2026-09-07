"""Acceptance tests for canonical conversational campaign state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from modelsurgeon.conversation import (
    ApprovalStatus,
    CampaignApproval,
    CampaignBudget,
    CampaignEvidence,
    CampaignLifecycle,
    CampaignOutcome,
    CampaignSpec,
    CampaignState,
    CampaignStateError,
    CampaignStateStore,
    new_campaign_state,
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
        ({"metric": "quality", "minimum": 0.95},),
    )
    return new_campaign_state(
        session_id="chat_session_" + "a" * 64,
        run_id="run_" + "b" * 64,
        source_model_digest="sha256:" + "c" * 64,
        spec=campaign_spec,
        policy_state={"decision": "supported", "policy_revision": "v1"},
        provider_context={"provider_id": "fixture", "runtime_revision": "fixture-v1"},
        budget=CampaignBudget(60.0, 1024, 4, 4096),
        provenance={"record_type": "fixture_campaign", "source": "trusted-test"},
    )


def test_state_serialization_is_canonical_and_excludes_transcript() -> None:
    state = _state()
    assert state.canonical_json() == state.canonical_json()
    record = state.to_record()
    assert "transcript" not in record
    assert "history" not in record
    assert record["spec"] == state.spec.to_record()
    assert state.hard_constraints == state.spec.hard_constraints


def test_restart_reconnect_recovers_state_without_chat_replay(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    initial = _state()
    with CampaignStateStore(path) as store:
        created = store.create(initial)
        running = store.transition(
            created.campaign_id,
            expected_version=0,
            kind="started",
            provenance={"record_type": "engine_transition", "source": "campaign-runner"},
            lifecycle=CampaignLifecycle.RUNNING,
        )

    with CampaignStateStore(path) as restarted:
        recovered = restarted.reconnect(running.campaign_id, running.session_id)
        assert recovered == running
        assert restarted.load_for_session(running.session_id) == running
        assert restarted.history(running.campaign_id)[-1].to_record()["kind"] == "started"


def test_material_spec_change_invalidates_prior_approval_and_is_versioned(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    initial = _state()
    approved = CampaignApproval(
        ApprovalStatus.APPROVED,
        initial.spec_digest,
        approval_id="approval_" + "d" * 64,
        recorded_by="operator_fixture",
        expires_at="2030-01-01T00:00:00+00:00",
        provenance={"record_type": "approval", "source": "fixture"},
    )
    initial = new_campaign_state(
        session_id=initial.session_id,
        run_id=initial.run_id,
        source_model_digest=initial.source_model_digest,
        spec=initial.spec,
        policy_state=initial.policy_state,
        provider_context=initial.provider_context,
        budget=initial.budget,
        approval=approved,
        provenance=initial.provenance,
    )
    changed_payload = {"constraints": [{"metric": "quality", "minimum": 0.97}]}
    changed_spec = CampaignSpec(
        "contract_quality_v2",
        _digest(changed_payload),
        changed_payload,
        ({"metric": "quality", "minimum": 0.97},),
    )
    with CampaignStateStore(path) as store:
        created = store.create(initial)
        changed = store.transition(
            created.campaign_id,
            expected_version=0,
            kind="spec_amended",
            provenance={"record_type": "spec_amendment", "source": "operator"},
            spec=changed_spec,
        )
        assert changed.spec_digest == changed_spec.spec_digest
        assert changed.approval.status is ApprovalStatus.PENDING
        assert changed.state_version == 1


def test_failed_and_inconclusive_evidence_is_retained_and_duplicate_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.sqlite3"
    evidence = CampaignEvidence(
        "evidence_" + "d" * 64,
        "sha256:" + "e" * 64,
        CampaignOutcome.FAILED,
        "runtime was unavailable",
        {"record_type": "runtime_probe", "source": "fixture"},
        inconclusive=True,
    )
    with CampaignStateStore(path) as store:
        created = store.create(_state())
        retained = store.append_evidence(
            created.campaign_id, evidence, expected_version=created.state_version
        )
        duplicate = store.append_evidence(
            created.campaign_id, evidence, expected_version=created.state_version
        )
        assert retained.evidence_cursor.evidence_id == evidence.evidence_id
        assert duplicate == retained
        assert store.evidence(created.campaign_id) == (evidence,)


def test_stale_transition_and_forbidden_ephemeral_fields_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    with CampaignStateStore(path) as store:
        created = store.create(_state())
        store.transition(
            created.campaign_id,
            expected_version=0,
            kind="started",
            provenance={"record_type": "engine_transition", "source": "fixture"},
            lifecycle=CampaignLifecycle.RUNNING,
        )
        with pytest.raises(CampaignStateError, match="stale campaign state"):
            store.transition(
                created.campaign_id,
                expected_version=0,
                kind="stale_transcript_update",
                provenance={"record_type": "chat", "source": "untrusted"},
            )
        with pytest.raises(CampaignStateError, match="ephemeral"):
            store.transition(
                created.campaign_id,
                expected_version=1,
                kind="bad_context",
                provenance={"record_type": "engine", "source": "fixture"},
                provider_context={"metadata": {"transcript": "must not become state"}},
            )


def test_schema_migration_and_corruption_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    with CampaignStateStore(path):
        pass
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE campaign_state_schema_migrations SET checksum = 'tampered' WHERE version = 1"
    )
    connection.commit()
    connection.close()
    with pytest.raises(CampaignStateError, match=r"missing, corrupt, or unsupported|checksum"):
        CampaignStateStore(path)

    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_text("not a sqlite database", encoding="utf-8")
    with pytest.raises(CampaignStateError, match="missing, corrupt, or unsupported"):
        CampaignStateStore(corrupt)


def test_concurrent_reader_sees_last_committed_canonical_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    first = CampaignStateStore(path)
    second = CampaignStateStore(path)
    try:
        created = first.create(_state())
        with second.reader() as reader:
            row = reader.execute(
                "SELECT state_json FROM campaign_state_records WHERE campaign_id = ?",
                (created.campaign_id,),
            ).fetchone()
        assert row is not None
        assert json.loads(str(row[0])) == created.to_record()
        updated = first.transition(
            created.campaign_id,
            expected_version=0,
            kind="paused",
            provenance={"record_type": "engine_transition", "source": "fixture"},
            lifecycle=CampaignLifecycle.PAUSED,
        )
        assert second.load(created.campaign_id) == updated
    finally:
        first.close()
        second.close()


def test_lifecycle_commands_are_replayable_and_terminal_cancel_fails_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "campaign.sqlite3"
    with CampaignStateStore(path) as store:
        created = store.create(_state())
        approved = CampaignApproval(
            ApprovalStatus.APPROVED,
            created.spec_digest,
            approval_id="approval_lifecycle",
            recorded_by="operator_fixture",
            expires_at="2030-01-01T00:00:00+00:00",
            provenance={"record_type": "approval", "source": "fixture"},
        )
        running = store.transition(
            created.campaign_id,
            expected_version=created.state_version,
            kind="started",
            provenance={
                "record_type": "campaign_transition",
                "source": "fixture",
                "operation_id": "start_fixture",
            },
            lifecycle=CampaignLifecycle.RUNNING,
            approval=approved,
        )
        paused = store.pause(
            running.campaign_id,
            running.session_id,
            operation_id="pause_fixture",
        )
        assert (
            store.pause(
                running.campaign_id,
                running.session_id,
                operation_id="pause_fixture",
            )
            == paused
        )
        resumed = store.resume(
            paused.campaign_id,
            paused.session_id,
            operation_id="resume_fixture",
        )
        cancelled = store.cancel(
            resumed.campaign_id,
            resumed.session_id,
            operation_id="cancel_fixture",
        )
        assert cancelled.lifecycle is CampaignLifecycle.CANCELLED
        assert cancelled.state_version == resumed.state_version + 1
        with pytest.raises(CampaignStateError, match="cannot resume"):
            store.resume(cancelled.campaign_id, cancelled.session_id)


def test_expired_approval_is_recorded_and_resume_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    initial = _state()
    approved = CampaignApproval(
        ApprovalStatus.APPROVED,
        initial.spec_digest,
        approval_id="approval_expired",
        recorded_by="operator_fixture",
        expires_at="2020-01-01T00:00:00+00:00",
        provenance={"record_type": "approval", "source": "fixture"},
    )
    initial = new_campaign_state(
        session_id=initial.session_id,
        run_id=initial.run_id,
        source_model_digest=initial.source_model_digest,
        spec=initial.spec,
        policy_state=initial.policy_state,
        provider_context=initial.provider_context,
        budget=initial.budget,
        approval=approved,
        provenance=initial.provenance,
    )
    with CampaignStateStore(path) as store:
        created = store.create(initial)
        paused = store.transition(
            created.campaign_id,
            expected_version=created.state_version,
            kind="paused",
            provenance={"record_type": "fixture", "source": "test"},
            lifecycle=CampaignLifecycle.PAUSED,
        )
        with pytest.raises(CampaignStateError, match="expired"):
            store.resume(paused.campaign_id, paused.session_id)
        expired = store.load(paused.campaign_id)
        assert expired.lifecycle is CampaignLifecycle.PAUSED
        assert expired.approval.status is ApprovalStatus.EXPIRED
        assert expired.state_version == paused.state_version + 1

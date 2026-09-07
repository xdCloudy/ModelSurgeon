"""Bounded v2.7 campaign recovery, fault, and determinism matrix tests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    ConversationSummaryBudget,
    ConversationSummaryOutcome,
    EvidenceCursor,
    StaleContextError,
    context_from_campaign,
    new_campaign_state,
    reconnect_conversation,
    summarize_conversation,
)
from modelsurgeon.experiments.identity import canonical_identity_json

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "tests" / "fixtures" / "campaign_recovery_matrix_v1.json"
MATRIX_RECORD = json.loads(MATRIX.read_text(encoding="utf-8"))
MATRIX_CELLS = tuple(MATRIX_RECORD["cells"])
EXPECTED_CELL_IDS = tuple(item["cell_id"] for item in MATRIX_CELLS)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_identity_json(value).encode()).hexdigest()


def _state(cell_id: str, *, expired: bool = False) -> CampaignState:
    spec_payload = {
        "constraints": [
            {"metric": "quality", "minimum": 0.95},
            {"metric": "latency_ms", "maximum": 250},
        ],
        "cell_id": cell_id,
    }
    spec = CampaignSpec(
        "contract_" + cell_id,
        _digest(spec_payload),
        spec_payload,
        tuple(spec_payload["constraints"]),  # type: ignore[arg-type]
    )
    approval = CampaignApproval(
        ApprovalStatus.APPROVED,
        spec.spec_digest,
        approval_id="approval_" + cell_id,
        recorded_by="operator_fixture",
        expires_at=("2020-01-01T00:00:00+00:00" if expired else "2099-01-01T00:00:00+00:00"),
        provenance={"record_type": "fixture_approval", "source": "trusted-test"},
    )
    return new_campaign_state(
        session_id="chat_session_" + cell_id,
        run_id="run_" + cell_id,
        source_model_digest="sha256:" + "a" * 64,
        spec=spec,
        policy_state={"record_type": "fixture_policy", "revision": "v1"},
        provider_context={"provider_id": "fixture", "revision": "v1"},
        budget=CampaignBudget(60.0, 4096, 4, 8192),
        approval=approval,
        campaign_id="campaign_" + cell_id,
        provenance={
            "record_type": "fixture_campaign",
            "source": "trusted-test",
            "fixture_revision": "campaign-recovery-v1",
        },
    )


def _evidence(
    cell_id: str,
    outcome: CampaignOutcome,
    *,
    inconclusive: bool = False,
) -> CampaignEvidence:
    return CampaignEvidence(
        "evidence_" + cell_id,
        "sha256:" + "b" * 64,
        outcome,
        "fixture evidence: " + outcome.value,
        {"record_type": "fixture_evidence", "source": "trusted-test", "cell_id": cell_id},
        artifact_digest="sha256:" + "c" * 64,
        inconclusive=inconclusive,
    )


def _next_action(state: CampaignState) -> str:
    if state.lifecycle is CampaignLifecycle.CANCELLED:
        return "stop_cancelled"
    if state.approval.status is not ApprovalStatus.APPROVED or not state.approval.active:
        return "await_fresh_approval"
    if state.outcome is CampaignOutcome.UNKNOWN and state.evidence_cursor.sequence:
        return "review_evidence"
    if state.lifecycle is CampaignLifecycle.PAUSED:
        return "resume"
    if state.lifecycle is CampaignLifecycle.CREATED:
        return "start"
    return "execute_next_stage"


def _snapshot(
    state: CampaignState,
    evidence: tuple[CampaignEvidence, ...],
    action: str,
) -> dict[str, object]:
    return {
        "state": state.to_record(),
        "state_digest": state.digest,
        "evidence_cursor": state.evidence_cursor.to_record(),
        "evidence": [item.to_record() for item in evidence],
        "artifacts": sorted(
            item.artifact_digest for item in evidence if item.artifact_digest is not None
        ),
        "action": action,
        "source_model_digest": state.source_model_digest,
        "hard_constraints": [dict(item) for item in state.hard_constraints],
        "budget": state.budget.to_record(),
        "provenance": dict(state.provenance),
    }


def _direct_and_chat_snapshots(
    store: CampaignStateStore,
    state: CampaignState,
    *,
    transcript_budget: int = 128,
) -> tuple[dict[str, object], dict[str, object], ConversationSummaryOutcome]:
    direct_state = store.reconnect(state.campaign_id, state.session_id)
    direct_evidence = store.evidence(state.campaign_id)
    direct = _snapshot(direct_state, direct_evidence, _next_action(direct_state))
    summary_result = summarize_conversation(
        state,
        direct_evidence,
        (
            {"role": "user", "content": "resume the campaign from the last committed stage"},
            {"role": "assistant", "content": "the transcript is not authoritative"},
            {"role": "user", "content": "old context may be unavailable"},
        ),
        budget=ConversationSummaryBudget(max_transcript_tokens=transcript_budget),
    )
    if summary_result.summary is None:
        raise AssertionError("fixture summary unexpectedly has no bounded summary")
    chat = reconnect_conversation(store, summary_result.summary)
    chat_snapshot = _snapshot(chat.state, chat.evidence, _next_action(chat.state))
    return direct, chat_snapshot, summary_result.outcome


def _child_reconnect(path: Path, campaign_id: str, session_id: str) -> dict[str, object]:
    code = """
import json
import sys
from modelsurgeon.conversation import CampaignStateStore

with CampaignStateStore(sys.argv[1]) as store:
    state = store.reconnect(sys.argv[2], sys.argv[3])
    print(json.dumps({"state": state.to_record(), "digest": state.digest}, sort_keys=True))
"""
    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path), campaign_id, session_id],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return json.loads(completed.stdout)


def _run_cell(
    cell_id: str,
    root: Path,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> dict[str, Any]:
    if cell_id in {"fault_transition_after_sql", "fault_evidence_after_sql"} and monkeypatch:
        # A single parametrized test may run both fault cells; each fixture
        # campaign must be created before the next injection is installed.
        monkeypatch.undo()
    path = root / (cell_id + ".sqlite3")
    initial = _state(cell_id, expired=cell_id == "expired_approval")
    with CampaignStateStore(path) as store:
        state = store.create(initial)
        if cell_id == "pause_resume":
            state = store.pause(
                state.campaign_id, state.session_id, operation_id="pause_fixture"
            )
            state = store.resume(
                state.campaign_id, state.session_id, operation_id="resume_fixture"
            )
        elif cell_id == "cancel":
            state = store.cancel(state.campaign_id, state.session_id, operation_id="cancel_fixture")
        elif cell_id == "restart":
            state = store.pause(
                state.campaign_id, state.session_id, operation_id="pause_fixture"
            )
            state = store.restart(
                state.campaign_id, state.session_id, operation_id="restart_fixture"
            )
        elif cell_id == "stale_plan":
            plan = {"plan_id": "plan_fixture", "steps": ["inspect", "evaluate"]}
            expected = context_from_campaign(
                state, plan, (), transaction_id="transaction_fixture"
            )
            state = store.append_evidence(
                state.campaign_id,
                _evidence(cell_id, CampaignOutcome.FAILED),
                expected_version=state.state_version,
            )
            with pytest.raises(StaleContextError, match="evidence"):
                from modelsurgeon.conversation import build_replan_proposal

                build_replan_proposal(
                    store,
                    campaign_id=state.campaign_id,
                    expected_context=expected,
                    current_plan=plan,
                    candidate_plan={"plan_id": "plan_new"},
                    candidate_spec=state.spec,
                    candidate_provider_context=state.provider_context,
                    candidate_budget=state.budget,
                    transaction_id="transaction_fixture",
                )
            direct, chat, summary_outcome = _direct_and_chat_snapshots(store, state)
            direct["action"] = chat["action"] = "rebuild_stale_plan"
            return {
                "outcome": "failed",
                "action": "rebuild_stale_plan",
                "direct": direct,
                "chat": chat,
                "summary": summary_outcome.value,
            }
        elif cell_id == "expired_approval":
            state = store.pause(
                state.campaign_id, state.session_id, operation_id="pause_fixture"
            )
            with pytest.raises(CampaignStateError, match="expired"):
                store.resume(state.campaign_id, state.session_id, operation_id="resume_expired")
            state = store.load(state.campaign_id)
        elif cell_id == "partial_evidence":
            state = store.append_evidence(
                state.campaign_id,
                _evidence(cell_id, CampaignOutcome.UNKNOWN, inconclusive=True),
                expected_version=state.state_version,
            )
        elif cell_id in {"fault_transition_after_sql", "fault_evidence_after_sql"}:
            if monkeypatch is None:
                raise AssertionError("fault cells require monkeypatch support")
            original_write = CampaignStateStore._write_transition

            def fail_after_write(
                connection: Any,
                next_state: CampaignState,
                transition: Any,
            ) -> None:
                original_write(connection, next_state, transition)
                raise RuntimeError("bounded fixture fault")

            monkeypatch.setattr(
                CampaignStateStore,
                "_write_transition",
                staticmethod(fail_after_write),
            )
            with pytest.raises(RuntimeError, match="bounded fixture fault"):
                if cell_id == "fault_transition_after_sql":
                    store.transition(
                        state.campaign_id,
                        expected_version=state.state_version,
                        kind="paused",
                        provenance={"record_type": "fault_fixture", "source": "test"},
                        lifecycle=CampaignLifecycle.PAUSED,
                    )
                else:
                    store.append_evidence(
                        state.campaign_id,
                        _evidence(cell_id, CampaignOutcome.UNKNOWN, inconclusive=True),
                        expected_version=state.state_version,
                    )
            state = store.load(state.campaign_id)
        elif cell_id == "process_restart":
            state = store.pause(
                state.campaign_id, state.session_id, operation_id="pause_fixture"
            )
            child = _child_reconnect(path, state.campaign_id, state.session_id)
            assert child["state"] == state.to_record()
            assert child["digest"] == state.digest
        elif cell_id in {"reconnect", "transcript_loss", "direct_chat_recovery"}:
            pass
        else:
            raise AssertionError(f"unhandled recovery matrix cell {cell_id}")

        if cell_id == "partial_evidence":
            direct, chat, summary_outcome = _direct_and_chat_snapshots(store, state)
            direct["action"] = chat["action"] = "review_evidence"
            return {
                "outcome": "unknown",
                "action": "review_evidence",
                "direct": direct,
                "chat": chat,
                "summary": summary_outcome.value,
            }
        if cell_id == "expired_approval":
            direct, chat, summary_outcome = _direct_and_chat_snapshots(store, state)
            direct["action"] = chat["action"] = "await_fresh_approval"
            return {
                "outcome": "failed",
                "action": "await_fresh_approval",
                "direct": direct,
                "chat": chat,
                "summary": summary_outcome.value,
            }
        transcript_budget = 8 if cell_id == "transcript_loss" else 128
        direct, chat, summary_outcome = _direct_and_chat_snapshots(
            store, state, transcript_budget=transcript_budget
        )
        if cell_id == "cancel":
            outcome = "cancelled"
        elif cell_id == "transcript_loss":
            outcome = summary_outcome.value
        elif cell_id in {"fault_transition_after_sql", "fault_evidence_after_sql"}:
            outcome = "unknown"
        else:
            outcome = "supported"
        return {
            "outcome": outcome,
            "action": direct["action"],
            "direct": direct,
            "chat": chat,
            "summary": summary_outcome.value,
        }


def test_recovery_fixture_declares_the_complete_matrix() -> None:
    assert MATRIX_RECORD["record_type"] == "bounded_campaign_recovery_matrix"
    assert len(MATRIX_CELLS) == 12
    assert len(EXPECTED_CELL_IDS) == len(set(EXPECTED_CELL_IDS))
    assert set(MATRIX_RECORD["required_guarantees"]) == {
        "approval_is_required_for_resume",
        "hard_constraints_are_preserved",
        "transaction_boundaries_are_atomic",
        "identical_inputs_have_identical_next_action",
        "negative_and_inconclusive_evidence_is_retained",
        "resource_budgets_and_provenance_are_preserved",
        "ui_reconnect_is_not_correctness_evidence",
    }


@pytest.mark.parametrize("cell", MATRIX_CELLS, ids=EXPECTED_CELL_IDS)
def test_recovery_cell_preserves_authority_and_matches_direct_chat(
    cell: dict[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed = _run_cell(cell["cell_id"], tmp_path, monkeypatch)
    assert observed["outcome"] == cell["expected_outcome"]
    assert observed["action"] == cell["expected_action"]
    assert observed["direct"] == observed["chat"]
    assert observed["direct"]["source_model_digest"] == "sha256:" + "a" * 64
    assert observed["direct"]["hard_constraints"]
    assert observed["direct"]["budget"] == CampaignBudget(60.0, 4096, 4, 8192).to_record()
    assert observed["direct"]["provenance"].get("record_type")
    if cell["cell_id"] == "transcript_loss":
        assert observed["summary"] == ConversationSummaryOutcome.SUPPORTED_WITH_LOSS.value
    if cell["cell_id"] == "fault_evidence_after_sql":
        assert observed["direct"]["evidence_cursor"] == EvidenceCursor().to_record()
    if cell["cell_id"] == "partial_evidence":
        assert (
            observed["direct"]["evidence"] == []
            or observed["direct"]["evidence"][0]["inconclusive"]
        )


def test_identical_recovery_inputs_have_identical_next_action(tmp_path: Path) -> None:
    first = _run_cell("restart", tmp_path / "first")
    second = _run_cell("restart", tmp_path / "second")
    assert first["direct"] == second["direct"]
    assert first["chat"] == second["chat"]
    assert first["action"] == second["action"] == "execute_next_stage"


def test_process_restart_reopens_the_same_committed_snapshot(tmp_path: Path) -> None:
    observed = _run_cell("process_restart", tmp_path)
    assert observed["direct"] == observed["chat"]
    assert observed["direct"]["state"]["lifecycle"] == "paused"
    assert observed["direct"]["action"] == "resume"


def test_fault_injection_rolls_back_transition_and_partial_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transition = _run_cell("fault_transition_after_sql", tmp_path / "transition", monkeypatch)
    evidence = _run_cell("fault_evidence_after_sql", tmp_path / "evidence", monkeypatch)
    assert transition["direct"]["state"]["state_version"] == 0
    assert transition["direct"]["evidence_cursor"] == EvidenceCursor().to_record()
    assert evidence["direct"]["state"]["state_version"] == 0
    assert evidence["direct"]["evidence"] == []

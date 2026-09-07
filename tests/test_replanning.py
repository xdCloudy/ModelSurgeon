"""Acceptance tests for deterministic stale-context replanning."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.conversation import (
    CampaignBudget,
    CampaignEvidence,
    CampaignLifecycle,
    CampaignOutcome,
    CampaignSpec,
    CampaignState,
    CampaignStateStore,
    ReplanApprovalRequired,
    ReplanReplayError,
    StaleContextError,
    apply_replan,
    build_replan_approval_request,
    build_replan_proposal,
    context_from_campaign,
    detect_stale_context,
    new_campaign_state,
    replay_replan,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.optimization_package import ApprovalDecision, ApprovalDecisionKind
from modelsurgeon.explain import (
    CandidateDisposition,
    CandidateEvidenceStatus,
    CanonicalEvidenceArchive,
    FeasibilityCandidateEvidence,
    FeasibilityProvenance,
    build_feasibility_explanation,
)
from modelsurgeon.search import (
    ContractConstraintDirection,
    ContractMetric,
    ContractObjectiveDirection,
    HardConstraint,
    MetricObservation,
    MetricUnit,
    ObjectiveContract,
    SoftObjective,
    approve_objective_amendment,
    propose_objective_amendment,
)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_identity_json(value).encode()).hexdigest()


def _state() -> CampaignState:
    payload = {"constraints": [{"metric": "quality", "minimum": 0.95}]}
    spec = CampaignSpec(
        "contract_quality", _digest(payload), payload, tuple(payload["constraints"])
    )
    return new_campaign_state(
        session_id="chat_session_" + "a" * 64,
        run_id="run_" + "b" * 64,
        source_model_digest="sha256:" + "c" * 64,
        spec=spec,
        policy_state={"record_type": "fixture_policy", "plan_version": 1},
        provider_context={"provider_id": "fixture", "revision": "v1"},
        budget=CampaignBudget(60.0, 1024, 4, 4096),
        provenance={"record_type": "fixture_campaign", "source": "trusted-test"},
    )


def _plan(name: str, quality: float = 0.95) -> dict[str, object]:
    return {
        "plan_id": name,
        "schema_version": 1,
        "objective": {"quality": quality},
        "steps": ["inspect", "evaluate"],
    }


def _evidence() -> CampaignEvidence:
    return CampaignEvidence(
        "evidence_" + "d" * 64,
        "sha256:" + "e" * 64,
        CampaignOutcome.FAILED,
        "provider was unavailable",
        {"record_type": "fixture_probe", "provider_revision": "v1"},
        inconclusive=True,
    )


def _candidate_spec() -> CampaignSpec:
    payload = {"constraints": [{"metric": "quality", "minimum": 0.97}]}
    return CampaignSpec(
        "contract_quality_v2",
        _digest(payload),
        payload,
        tuple(payload["constraints"]),
    )


def _objective_amendment() -> object:
    original = ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                ContractConstraintDirection.MINIMUM,
                0.95,
                MetricUnit.RATIO,
            ),
        ),
        objectives=(
            SoftObjective(
                ContractMetric.QUALITY,
                ContractObjectiveDirection.MAXIMIZE,
                MetricUnit.RATIO,
                weight=1.0,
            ),
        ),
    )
    proposed = ObjectiveContract(
        constraints=original.constraints,
        objectives=(
            SoftObjective(
                ContractMetric.QUALITY,
                ContractObjectiveDirection.MAXIMIZE,
                MetricUnit.RATIO,
                weight=2.0,
            ),
        ),
    )
    source = "sha256:" + "a" * 64
    candidate = FeasibilityCandidateEvidence(
        "candidate_fixture",
        "evidence_fixture",
        CandidateEvidenceStatus.MEASURED,
        source,
        observations=(MetricObservation(ContractMetric.QUALITY, 0.94, MetricUnit.RATIO),),
        disposition=CandidateDisposition.REJECTED,
    )
    archive = CanonicalEvidenceArchive.build(original, (candidate,))
    explanation = build_feasibility_explanation(
        original,
        archive,
        provenance=FeasibilityProvenance(original.contract_id, source, archive.archive_id),
    )
    pending = propose_objective_amendment(
        original,
        proposed,
        rationale="fixture objective amendment",
        evidence=explanation,
        operator_id="operator_fixture",
        requested_at="2026-09-07T10:00:00Z",
        expires_at="2030-01-01T00:00:00Z",
        parent_campaign_id="campaign_fixture",
        session_id="chat_session_fixture",
        run_id="run_fixture",
    )
    return approve_objective_amendment(
        pending,
        operator_id="operator_fixture",
        decided_at="2026-09-07T10:01:00Z",
    )


def _objective_campaign(amendment: object) -> CampaignState:
    original = amendment.original_objective
    payload = original.to_record()
    return new_campaign_state(
        session_id="chat_session_fixture",
        run_id="run_fixture",
        source_model_digest="sha256:" + "a" * 64,
        spec=CampaignSpec(
            original.contract_id,
            _digest(payload),
            payload,
            tuple(item.to_record() for item in original.constraints),
        ),
        policy_state={"record_type": "fixture_policy", "plan_version": 1},
        provider_context={"provider_id": "fixture", "revision": "v1"},
        budget=CampaignBudget(60.0, 1024, 4, 4096),
        provenance={"record_type": "fixture_campaign", "source": "trusted-test"},
    )


def test_stale_report_distinguishes_evidence_provider_resource_and_transaction() -> None:
    state = _state()
    old = context_from_campaign(state, _plan("plan_old"), (), transaction_id="tool_transaction_old")
    newer = replace(
        old,
        evidence_ids=("evidence_" + "d" * 64,),
        evidence_digest=_digest([("evidence_" + "d" * 64, "sha256:" + "f" * 64)]),
        provider_digest=_digest({"provider_id": "fixture", "revision": "v2"}),
        resource_digest=_digest(CampaignBudget(30.0, 1024, 4, 4096).to_record()),
        transaction_id="tool_transaction_new",
    )
    report = detect_stale_context(old, newer)
    assert report.stale
    assert report.reasons == ("evidence", "provider", "resource", "transaction")
    assert report.to_record() == report.to_record()


def test_new_evidence_is_stale_until_context_is_refreshed(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    plan = _plan("plan_old")
    with CampaignStateStore(path) as store:
        created = store.create(_state())
        expected = context_from_campaign(created, plan, (), transaction_id="tool_transaction_1")
        updated = store.append_evidence(created.campaign_id, _evidence(), expected_version=0)
        current = context_from_campaign(
            updated, plan, store.evidence(created.campaign_id), transaction_id="tool_transaction_1"
        )
        assert detect_stale_context(expected, current).reasons == ("evidence",)
        with pytest.raises(StaleContextError, match="evidence"):
            build_replan_proposal(
                store,
                campaign_id=created.campaign_id,
                expected_context=expected,
                current_plan=plan,
                candidate_plan=plan,
                candidate_spec=created.spec,
                candidate_provider_context=created.provider_context,
                candidate_budget=created.budget,
                transaction_id="tool_transaction_1",
            )


def test_noop_replan_does_not_create_child_or_consume_approval(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    plan = _plan("plan_old")
    with CampaignStateStore(path) as store:
        created = store.create(_state())
        expected = context_from_campaign(created, plan, (), transaction_id="tool_transaction_1")
        result = apply_replan(
            store,
            build_replan_proposal(
                store,
                campaign_id=created.campaign_id,
                expected_context=expected,
                current_plan=plan,
                candidate_plan=plan,
                candidate_spec=created.spec,
                candidate_provider_context=created.provider_context,
                candidate_budget=created.budget,
                transaction_id="tool_transaction_1",
            ),
            candidate_spec=created.spec,
            candidate_policy_state=created.policy_state,
            candidate_provider_context=created.provider_context,
            candidate_budget=created.budget,
        )
        assert not result.material
        assert result.child_campaign_id is None
        assert store.campaigns_for_session(created.session_id) == (created,)


def test_material_replan_requires_fresh_approval_and_preserves_lineage(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    old_plan = _plan("plan_old")
    new_plan = _plan("plan_new", 0.97)
    with CampaignStateStore(path) as store:
        created = store.create(_state())
        retained = store.append_evidence(created.campaign_id, _evidence(), expected_version=0)
        expected = context_from_campaign(
            retained,
            old_plan,
            store.evidence(created.campaign_id),
            transaction_id="tool_transaction_1",
        )
        proposal = build_replan_proposal(
            store,
            campaign_id=created.campaign_id,
            expected_context=expected,
            current_plan=old_plan,
            candidate_plan=new_plan,
            candidate_spec=_candidate_spec(),
            candidate_provider_context=retained.provider_context,
            candidate_budget=retained.budget,
            transaction_id="tool_transaction_1",
        )
        assert proposal.material
        with pytest.raises(ReplanApprovalRequired):
            apply_replan(
                store,
                proposal,
                candidate_spec=_candidate_spec(),
                candidate_policy_state={"plan_id": "plan_new"},
                candidate_provider_context=retained.provider_context,
                candidate_budget=retained.budget,
            )
        request = build_replan_approval_request(
            proposal,
            requested_at="2026-09-07T10:00:00Z",
            expires_at="2030-01-01T00:00:00Z",
            operator_id="operator_fixture",
        )
        proposal = replace(proposal, approval_request=request)
        decision = ApprovalDecision(
            request.request_id,
            request.code,
            request.plan_id,
            request.plan_digest,
            request.diff_id,
            ApprovalDecisionKind.APPROVED,
            "2026-09-07T10:01:00Z",
            "operator_fixture",
            "approved after evidence review",
        )
        result = apply_replan(
            store,
            proposal,
            candidate_spec=_candidate_spec(),
            candidate_policy_state={"plan_id": "plan_new"},
            candidate_provider_context=retained.provider_context,
            candidate_budget=retained.budget,
            approval_decision=decision,
        )
        assert result.child_campaign_id is not None
        assert result.state.approval.approval_id == decision.decision_id
        assert result.state.approval.spec_digest == _candidate_spec().spec_digest
        assert store.evidence(created.campaign_id) == (_evidence(),)
        assert store.evidence(result.state.campaign_id) == ()
        assert store.lineage(result.state.campaign_id) == (
            result.state,
            store.load(created.campaign_id),
        )
        assert store.children(created.campaign_id) == (result.state,)


def test_objective_amendment_gates_material_replan_without_reusing_its_approval(
    tmp_path: Path,
) -> None:
    amendment = _objective_amendment()
    parent = _objective_campaign(amendment)
    proposed_payload = amendment.proposed_amendment.to_record()
    candidate_spec = CampaignSpec(
        amendment.proposed_spec_identity,
        _digest(proposed_payload),
        proposed_payload,
        tuple(item.to_record() for item in amendment.proposed_amendment.constraints),
    )
    old_plan = _plan("plan_old")
    new_plan = _plan("plan_amended", 0.97)
    path = tmp_path / "campaign.sqlite3"
    with CampaignStateStore(path) as store:
        created = store.create(parent)
        expected = context_from_campaign(
            created, old_plan, (), transaction_id="tool_transaction_amendment"
        )
        proposal = build_replan_proposal(
            store,
            campaign_id=created.campaign_id,
            expected_context=expected,
            current_plan=old_plan,
            candidate_plan=new_plan,
            candidate_spec=candidate_spec,
            candidate_provider_context=created.provider_context,
            candidate_budget=created.budget,
            transaction_id="tool_transaction_amendment",
            amendment=amendment,
        )
        request = build_replan_approval_request(
            proposal,
            requested_at="2026-09-07T10:02:00Z",
            expires_at="2030-01-01T00:00:00Z",
            operator_id="operator_fixture",
        )
        applied = apply_replan(
            store,
            replace(proposal, approval_request=request),
            candidate_spec=candidate_spec,
            candidate_policy_state={"record_type": "amended_policy"},
            candidate_provider_context=created.provider_context,
            candidate_budget=created.budget,
        )
        assert applied.state.provenance["objective_amendment_id"] == amendment.amendment_id
        assert applied.state.approval.status.value == "pending"
        assert applied.state.spec_identity == amendment.proposed_spec_identity


def test_provider_and_budget_changes_are_material_and_budget_is_preserved(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    plan = _plan("plan_old")
    with CampaignStateStore(path) as store:
        created = store.create(_state())
        expected = context_from_campaign(created, plan, (), transaction_id="tool_transaction_1")
        current = store.transition(
            created.campaign_id,
            expected_version=0,
            kind="provider_changed",
            provenance={"record_type": "provider_fixture", "source": "test"},
            provider_context={"provider_id": "fixture", "revision": "v2"},
        )
        refreshed = context_from_campaign(
            current, plan, (), transaction_id="tool_transaction_1"
        )
        assert detect_stale_context(expected, refreshed).reasons == ("provider",)
        smaller_budget = CampaignBudget(30.0, 1024, 2, 2048)
        proposal = build_replan_proposal(
            store,
            campaign_id=created.campaign_id,
            expected_context=refreshed,
            current_plan=plan,
            candidate_plan=plan,
            candidate_spec=current.spec,
            candidate_provider_context=current.provider_context,
            candidate_budget=smaller_budget,
            transaction_id="tool_transaction_1",
        )
        assert proposal.diff.changed_paths == ("resource_budget",)
        request = build_replan_approval_request(
            proposal,
            requested_at="2026-09-07T10:00:00Z",
            expires_at="2030-01-01T00:00:00Z",
            operator_id="operator_fixture",
        )
        result = apply_replan(
            store,
            replace(proposal, approval_request=request),
            candidate_spec=current.spec,
            candidate_policy_state={"record_type": "budget_policy"},
            candidate_provider_context=current.provider_context,
            candidate_budget=smaller_budget,
        )
        assert result.state.budget == smaller_budget
        assert result.state.provider_context == current.provider_context


def test_restart_does_not_make_semantically_equal_context_stale(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    plan = _plan("plan_old")
    with CampaignStateStore(path) as store:
        created = store.create(_state())
        paused = store.transition(
            created.campaign_id,
            expected_version=0,
            kind="paused",
            provenance={"record_type": "restart_fixture", "source": "test"},
            lifecycle=CampaignLifecycle.PAUSED,
        )
        restarted = store.transition(
            paused.campaign_id,
            expected_version=1,
            kind="restarted",
            provenance={"record_type": "restart_fixture", "source": "test"},
            lifecycle=CampaignLifecycle.RUNNING,
        )
        old = context_from_campaign(created, plan, (), transaction_id="tool_transaction_1")
        current = context_from_campaign(restarted, plan, (), transaction_id="tool_transaction_1")
        assert not detect_stale_context(old, current).stale


def test_cancelled_campaign_and_stale_replay_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    plan = _plan("plan_old")
    with CampaignStateStore(path) as store:
        created = store.create(_state())
        cancelled = store.cancel(
            created.campaign_id, created.session_id, operation_id="cancel_fixture"
        )
        expected = context_from_campaign(cancelled, plan, (), transaction_id="tool_transaction_1")
        with pytest.raises(ReplanReplayError, match="cancelled"):
            build_replan_proposal(
                store,
                campaign_id=created.campaign_id,
                expected_context=expected,
                current_plan=plan,
                candidate_plan=_plan("plan_new"),
                candidate_spec=_candidate_spec(),
                candidate_provider_context=cancelled.provider_context,
                candidate_budget=cancelled.budget,
                transaction_id="tool_transaction_1",
            )


def test_replay_is_idempotent_and_stale_parent_replay_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    plan = _plan("plan_old")
    candidate_plan = _plan("plan_new")
    with CampaignStateStore(path) as store:
        created = store.create(_state())
        expected = context_from_campaign(created, plan, (), transaction_id="tool_transaction_1")
        proposal = build_replan_proposal(
            store,
            campaign_id=created.campaign_id,
            expected_context=expected,
            current_plan=plan,
            candidate_plan=candidate_plan,
            candidate_spec=_candidate_spec(),
            candidate_provider_context=created.provider_context,
            candidate_budget=created.budget,
            transaction_id="tool_transaction_1",
        )
        request = build_replan_approval_request(
            proposal,
            requested_at="2026-09-07T10:00:00Z",
            expires_at="2030-01-01T00:00:00Z",
            operator_id="operator_fixture",
        )
        proposal = replace(proposal, approval_request=request)
        first = apply_replan(
            store,
            proposal,
            candidate_spec=_candidate_spec(),
            candidate_policy_state={"record_type": "replay_policy"},
            candidate_provider_context=created.provider_context,
            candidate_budget=created.budget,
        )
        replayed = replay_replan(
            store,
            proposal,
            candidate_spec=_candidate_spec(),
            candidate_policy_state={"record_type": "replay_policy"},
            candidate_provider_context=created.provider_context,
            candidate_budget=created.budget,
        )
        assert replayed.state == first.state

        stale_created = store.create(
            replace(_state(), campaign_id="campaign_stale", run_id="run_" + "f" * 64)
        )
        stale_expected = context_from_campaign(
            stale_created, plan, (), transaction_id="tool_transaction_stale"
        )
        stale_proposal = build_replan_proposal(
            store,
            campaign_id=stale_created.campaign_id,
            expected_context=stale_expected,
            current_plan=plan,
            candidate_plan=candidate_plan,
            candidate_spec=_candidate_spec(),
            candidate_provider_context=stale_created.provider_context,
            candidate_budget=stale_created.budget,
            transaction_id="tool_transaction_stale",
        )
        changed = store.transition(
            stale_created.campaign_id,
            expected_version=0,
            kind="paused",
            provenance={"record_type": "replay_fixture", "source": "test"},
            lifecycle=CampaignLifecycle.PAUSED,
        )
        assert changed.state_version == 1
        with pytest.raises(ReplanReplayError, match="stale"):
            replay_replan(
                store,
                stale_proposal,
                candidate_spec=_candidate_spec(),
                candidate_policy_state={"record_type": "replay_policy"},
                candidate_provider_context=stale_created.provider_context,
                candidate_budget=stale_created.budget,
            )


def test_plan_diff_and_identifiers_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "campaign.sqlite3"
    plan = _plan("plan_old")
    with CampaignStateStore(path) as store:
        created = store.create(_state())
        expected = context_from_campaign(created, plan, (), transaction_id="tool_transaction_1")
        first = build_replan_proposal(
            store,
            campaign_id=created.campaign_id,
            expected_context=expected,
            current_plan=plan,
            candidate_plan=_plan("plan_new"),
            candidate_spec=_candidate_spec(),
            candidate_provider_context=created.provider_context,
            candidate_budget=created.budget,
            transaction_id="tool_transaction_1",
        )
        second = build_replan_proposal(
            store,
            campaign_id=created.campaign_id,
            expected_context=expected,
            current_plan=plan,
            candidate_plan=_plan("plan_new"),
            candidate_spec=_candidate_spec(),
            candidate_provider_context=created.provider_context,
            candidate_budget=created.budget,
            transaction_id="tool_transaction_1",
        )
        assert first.diff.to_record() == second.diff.to_record()
        assert first.diff.diff_id == second.diff.diff_id
        assert first.diff.changed_paths == tuple(sorted(first.diff.changed_paths))

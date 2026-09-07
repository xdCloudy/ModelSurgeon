"""Immutable objective-amendment approval, replay, and history contracts."""

from __future__ import annotations

import pytest

from modelsurgeon.explain import (
    CandidateDisposition,
    CandidateEvidenceStatus,
    CanonicalEvidenceArchive,
    FeasibilityCandidateEvidence,
    FeasibilityProvenance,
    build_feasibility_explanation,
)
from modelsurgeon.search import (
    AmendmentStatus,
    ContractConstraintDirection,
    ContractMetric,
    ContractObjectiveDirection,
    HardConstraint,
    MetricObservation,
    MetricUnit,
    ObjectiveAmendmentError,
    ObjectiveAmendmentLedger,
    ObjectiveContract,
    ObjectiveMode,
    SoftObjective,
    apply_objective_amendment,
    approve_objective_amendment,
    cancel_objective_amendment,
    diff_objective_contracts,
    expire_objective_amendment,
    propose_objective_amendment,
    reject_objective_amendment,
    replay_objective_amendment,
)

SOURCE = "sha256:" + "a" * 64


def _contract(*, weight: float = 1.0, latency_limit: float = 100.0) -> ObjectiveContract:
    return ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                direction=ContractConstraintDirection.MINIMUM,
                threshold=0.95,
                unit=MetricUnit.RATIO,
            ),
            HardConstraint(
                ContractMetric.LATENCY,
                direction=ContractConstraintDirection.MAXIMUM,
                threshold=latency_limit,
                unit=MetricUnit.MILLISECONDS,
            ),
        ),
        objectives=(
            SoftObjective(
                ContractMetric.QUALITY,
                ContractObjectiveDirection.MAXIMIZE,
                MetricUnit.RATIO,
                weight=weight,
            ),
        ),
        mode=ObjectiveMode.WEIGHTED,
    )


def _evidence(contract: ObjectiveContract):
    candidate = FeasibilityCandidateEvidence(
        "candidate_near",
        "evidence_near",
        CandidateEvidenceStatus.MEASURED,
        SOURCE,
        observations=(
            MetricObservation(ContractMetric.LATENCY, 101.0, MetricUnit.MILLISECONDS),
            MetricObservation(ContractMetric.QUALITY, 0.94, MetricUnit.RATIO),
        ),
        disposition=CandidateDisposition.REJECTED,
    )
    archive = CanonicalEvidenceArchive.build(contract, (candidate,))
    return build_feasibility_explanation(
        contract,
        archive,
        provenance=FeasibilityProvenance(contract.contract_id, SOURCE, archive.archive_id),
    )


def _proposal(
    *,
    original: ObjectiveContract | None = None,
    proposed: ObjectiveContract | None = None,
    expires_at: str = "2026-09-07T11:00:00+00:00",
):
    original = original or _contract()
    proposed = proposed or _contract(weight=2.0)
    return propose_objective_amendment(
        original,
        proposed,
        rationale="measured near-miss evidence supports changing the soft priority",
        evidence=_evidence(original),
        operator_id="operator_alice",
        requested_at="2026-09-07T10:00:00+00:00",
        expires_at=expires_at,
        parent_campaign_id="campaign_original",
        session_id="session_fixture",
        run_id="run_fixture",
        operator_context={"ticket": "issue-460"},
    )


def test_material_diff_is_visible_and_approval_scope_covers_every_path() -> None:
    amendment = _proposal()

    assert amendment.status is AmendmentStatus.PENDING
    assert amendment.diff.material
    assert "objective.objectives[0].weight" in amendment.diff.changed_paths
    assert amendment.diff.diff_id == amendment.approval_request.diff_id
    assert set(amendment.diff.changed_paths) <= set(amendment.approval_scope)
    assert amendment.evidence_archive_id.startswith("evidence_archive_")
    assert amendment.preserved_evidence_ids == ("evidence_near",)
    assert amendment.hard_constraints == amendment.original_objective.constraints


def test_canonical_reordering_is_non_material_and_keeps_spec_identity() -> None:
    original = _contract()
    reordered = ObjectiveContract(
        constraints=tuple(reversed(original.constraints)),
        objectives=tuple(reversed(original.objectives)),
        mode=original.mode,
        approval_policy=original.approval_policy,
    )
    diff = diff_objective_contracts(original, reordered)
    amendment = _proposal(original=original, proposed=reordered)

    assert diff.non_material
    assert diff.changed_paths == ()
    assert not amendment.diff.material
    approved = approve_objective_amendment(
        amendment,
        operator_id="operator_alice",
        decided_at="2026-09-07T10:05:00+00:00",
    )
    applied = apply_objective_amendment(
        approved,
        current_objective=original,
        current_campaign_id="campaign_original",
        applied_at="2026-09-07T10:06:00+00:00",
    )
    assert applied.effective_spec_identity == original.contract_id
    assert applied.downstream_campaign_id == "campaign_original"


def test_approve_reject_expire_and_cancel_are_terminal_and_immutable() -> None:
    approved = approve_objective_amendment(
        _proposal(),
        operator_id="operator_alice",
        decided_at="2026-09-07T10:05:00+00:00",
    )
    assert approved.status is AmendmentStatus.APPROVED
    assert approved.approval_id is not None

    rejected = reject_objective_amendment(
        _proposal(),
        operator_id="operator_bob",
        decided_at="2026-09-07T10:05:00+00:00",
        reason="the proposed trade-off is not accepted",
    )
    assert rejected.status is AmendmentStatus.REJECTED

    expired = expire_objective_amendment(_proposal())
    assert expired.status is AmendmentStatus.EXPIRED

    cancelled = cancel_objective_amendment(
        _proposal(),
        operator_id="operator_alice",
        cancelled_at="2026-09-07T10:05:00+00:00",
        reason="the user withdrew the request",
    )
    assert cancelled.status is AmendmentStatus.CANCELLED

    with pytest.raises(ObjectiveAmendmentError, match="rejected state"):
        approve_objective_amendment(
            rejected,
            operator_id="operator_bob",
            decided_at="2026-09-07T10:06:00+00:00",
        )


def test_material_apply_creates_new_campaign_and_preserves_prior_evidence() -> None:
    amendment = approve_objective_amendment(
        _proposal(),
        operator_id="operator_alice",
        decided_at="2026-09-07T10:05:00+00:00",
    )
    applied = apply_objective_amendment(
        amendment,
        current_objective=amendment.original_objective,
        current_campaign_id="campaign_original",
        applied_at="2026-09-07T10:06:00+00:00",
    )

    assert applied.material
    assert applied.original_campaign_id == "campaign_original"
    assert applied.downstream_campaign_id is not None
    assert applied.downstream_campaign_id != "campaign_original"
    assert applied.effective_spec_identity == amendment.proposed_spec_identity
    assert applied.evidence_archive_id == amendment.evidence_archive_id
    assert applied.preserved_evidence_ids == amendment.preserved_evidence_ids


def test_stale_and_expired_replay_fail_closed_but_applied_replay_is_idempotent() -> None:
    approved = approve_objective_amendment(
        _proposal(),
        operator_id="operator_alice",
        decided_at="2026-09-07T10:05:00+00:00",
    )
    with pytest.raises(ObjectiveAmendmentError, match="stale amendment"):
        apply_objective_amendment(
            approved,
            current_objective=_contract(weight=3.0),
            current_campaign_id="campaign_original",
            applied_at="2026-09-07T10:06:00+00:00",
        )

    expired = expire_objective_amendment(_proposal(expires_at="2026-09-07T11:00:00+00:00"))
    with pytest.raises(ObjectiveAmendmentError, match="expired"):
        apply_objective_amendment(
            expired,
            current_objective=expired.original_objective,
            applied_at="2026-09-07T12:06:00+00:00",
        )

    application = apply_objective_amendment(
        approved,
        current_objective=approved.original_objective,
        current_campaign_id="campaign_original",
        applied_at="2026-09-07T10:06:00+00:00",
    )
    applied = approved.__class__(
        approved.original_objective,
        approved.proposed_amendment,
        approved.rationale,
        approved.evidence,
        approved.approval_request,
        approved.diff,
        approved.source_model_digest,
        approved.parent_campaign_id,
        approved.session_id,
        approved.run_id,
        AmendmentStatus.APPLIED,
        approved.decision,
        application=application,
    )
    assert replay_objective_amendment(
        applied,
        current_objective=approved.proposed_amendment,
        replayed_at="2026-09-07T10:07:00+00:00",
    ) == application


def test_ledger_exposes_immutable_history_and_idempotent_replay() -> None:
    amendment = _proposal()
    ledger = ObjectiveAmendmentLedger()
    ledger.propose(amendment)
    ledger.propose(amendment)
    ledger.approve(
        amendment.amendment_id,
        operator_id="operator_alice",
        decided_at="2026-09-07T10:05:00+00:00",
    )
    application = ledger.apply(
        amendment.amendment_id,
        current_objective=amendment.original_objective,
        current_campaign_id="campaign_original",
        applied_at="2026-09-07T10:06:00+00:00",
    )
    assert len(ledger.history.entries) == 3
    assert ledger.inspect(amendment.amendment_id).status is AmendmentStatus.APPLIED
    assert ledger.inspect(amendment.amendment_id).application == application
    assert ledger.history.for_campaign("campaign_original")
    with pytest.raises(ObjectiveAmendmentError, match="replay or transition"):
        ledger.history.append(reject_objective_amendment(
            _proposal(),
            operator_id="operator_bob",
            decided_at="2026-09-07T10:05:00+00:00",
            reason="conflicting replay",
        ))


def test_hard_constraint_changes_are_explicitly_material_and_scope_bound() -> None:
    amendment = _proposal(proposed=_contract(latency_limit=90.0))

    assert amendment.diff.material
    assert amendment.diff.hard_constraints_changed
    assert "hard_constraints" in amendment.approval_scope
    assert amendment.original_objective.constraints != amendment.proposed_amendment.constraints

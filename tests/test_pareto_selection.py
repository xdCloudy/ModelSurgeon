"""Golden and fail-closed tests for measured Pareto selection explanations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.explain import (
    CandidateDisposition,
    CandidateEvidenceStatus,
    CanonicalEvidenceArchive,
    FeasibilityCandidateEvidence,
    FeasibilityProvenance,
    ParetoSelectionError,
    ParetoSelectionOutcome,
    build_feasibility_explanation,
    build_pareto_selection_explanation,
    render_pareto_selection_explanation,
    replay_pareto_selection_explanation,
)
from modelsurgeon.search import (
    ContractConstraintDirection,
    ContractMetric,
    ContractObjectiveDirection,
    ContractObjectiveNormalization,
    HardConstraint,
    MetricObservation,
    MetricUnit,
    ObjectiveContract,
    ObjectiveMode,
    SoftObjective,
    apply_objective_amendment,
    approve_objective_amendment,
    propose_objective_amendment,
)

SOURCE = "sha256:" + "c" * 64
ROOT = Path(__file__).resolve().parents[1]


def _contract(
    *, mode: ObjectiveMode = ObjectiveMode.PARETO, weight: float = 1.0
) -> ObjectiveContract:
    return ObjectiveContract(
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
                weight=weight,
                baseline=1.0,
            ),
            SoftObjective(
                ContractMetric.LATENCY,
                ContractObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                normalization=ContractObjectiveNormalization.IDENTITY,
            ),
        ),
        mode=mode,
    )


def _candidate(
    name: str,
    *,
    quality: float | None = None,
    latency: float | None = None,
    quality_lower: float | None = None,
    quality_upper: float | None = None,
    status: CandidateEvidenceStatus = CandidateEvidenceStatus.MEASURED,
    disposition: CandidateDisposition = CandidateDisposition.ACCEPTED,
) -> FeasibilityCandidateEvidence:
    observations: list[MetricObservation] = []
    if quality is not None:
        observations.append(
            MetricObservation(
                ContractMetric.QUALITY,
                quality,
                MetricUnit.RATIO,
                lower=quality_lower,
                upper=quality_upper,
            )
        )
    if latency is not None:
        observations.append(
            MetricObservation(ContractMetric.LATENCY, latency, MetricUnit.MILLISECONDS)
        )
    return FeasibilityCandidateEvidence(
        f"candidate_{name}",
        f"evidence_{name}",
        status,
        SOURCE,
        tuple(sorted(observations, key=lambda item: item.metric))
        if status is CandidateEvidenceStatus.MEASURED
        else (),
        disposition=disposition,
        evaluation_id=f"evaluation_{name}",
        provenance={"fixture": "pareto-selection-v1", "candidate": name},
    )


def _build(
    contract: ObjectiveContract,
    candidates: tuple[FeasibilityCandidateEvidence, ...],
    **kwargs: object,
):
    archive = CanonicalEvidenceArchive.build(contract, candidates)
    return build_pareto_selection_explanation(
        contract,
        archive,
        provenance=FeasibilityProvenance(
            contract.contract_id,
            SOURCE,
            archive.archive_id,
            approval_id="approval_selection_fixture",
            approval_provenance={"operator": "operator_fixture"},
        ),
        **kwargs,
    )


def test_golden_tied_dominated_and_uncertain_frontier() -> None:
    result = _build(
        _contract(),
        (
            _candidate("a", quality=0.96, latency=100.0),
            _candidate("tie", quality=0.96, latency=100.0),
            _candidate("dominated", quality=0.95, latency=110.0),
            _candidate(
                "uncertain",
                quality=0.96,
                quality_lower=0.95,
                quality_upper=0.97,
                latency=90.0,
            ),
        ),
    )

    golden = json.loads(
        (ROOT / "tests" / "golden" / "pareto_selection_v1.json").read_text(encoding="utf-8")
    )
    direct = result.direct_report()
    assert direct["record_type"] == golden["record_type"]
    assert direct["schema_version"] == golden["schema_version"]
    assert direct["frontier_candidate_ids"] == golden["frontier_candidate_ids"]
    assert direct["selected_candidate_id"] == golden["selected_candidate_id"]
    assert direct["outcome"] == golden["outcome"]
    assert {item["candidate_id"]: item["status"] for item in direct["alternatives"]} == golden[
        "statuses"
    ]
    assert golden["selection_rule_contains"] in result.selection_rule
    assert result.objective_terms[0].kind == "hard_constraint"
    assert result.objective_terms[1].kind == "soft_objective"
    assert result.candidate_scores[-1].uncertain
    assert "tied" in result.rationale


def test_infeasible_frontier_never_selects_a_near_miss() -> None:
    result = _build(
        _contract(),
        (_candidate("bad", quality=0.90, latency=80.0),),
    )

    assert result.outcome is ParetoSelectionOutcome.NO_FEASIBLE_FRONTIER
    assert result.selected_candidate_id is None
    assert result.frontier_candidate_ids == ()
    assert result.alternatives[0].status.value == "constraint_violation"
    assert "no candidate was selected" in result.rationale.lower()


def test_final_selection_evidence_may_choose_a_measured_frontier_tradeoff() -> None:
    result = _build(
        _contract(),
        (
            _candidate("a", quality=0.96, latency=100.0),
            _candidate(
                "uncertain",
                quality=0.96,
                quality_lower=0.95,
                quality_upper=0.97,
                latency=90.0,
            ),
        ),
        final_selection_evidence=(
            {
                "candidate_id": "candidate_a",
                "measured": True,
                "complete": True,
                "constraints_passed": True,
                "score": 0.0,
            },
            {
                "candidate_id": "candidate_uncertain",
                "measured": True,
                "complete": True,
                "constraints_passed": True,
                "score": 1.0,
            },
        ),
    )

    assert result.selected_candidate_id == "candidate_a"
    assert "final-selection evidence" in result.rationale
    assert result.decision_replay.selected_candidate_id == "candidate_a"


def test_replay_and_direct_report_are_byte_stable() -> None:
    contract = _contract(mode=ObjectiveMode.WEIGHTED)
    candidates = (
        _candidate("a", quality=0.96, latency=100.0),
        _candidate("b", quality=0.97, latency=120.0),
    )
    first = _build(contract, candidates)
    replayed = replay_pareto_selection_explanation(
        contract,
        tuple(reversed(candidates)),
        provenance=first.provenance,
        expected_report=first.direct_report(),
    )
    assert first.direct_report() == replayed.direct_report()
    assert render_pareto_selection_explanation(
        first, format="direct"
    ) == render_pareto_selection_explanation(first, format="direct")
    assert first.decision_replay.deterministic


def test_selection_rejects_a_candidate_outside_the_measured_frontier() -> None:
    contract = _contract()
    candidates = (
        _candidate("good", quality=0.96, latency=80.0),
        _candidate("bad", quality=0.90, latency=70.0),
    )
    with pytest.raises(ParetoSelectionError, match="does not reproduce"):
        _build(contract, candidates, selected_candidate_id="candidate_bad")


def test_amended_objective_has_new_identity_and_new_rationale() -> None:
    original = _contract(mode=ObjectiveMode.WEIGHTED, weight=1.0)
    proposed = _contract(mode=ObjectiveMode.WEIGHTED, weight=2.0)
    candidates = (
        _candidate("a", quality=0.96, latency=100.0),
        _candidate("b", quality=0.97, latency=120.0),
    )
    original_archive = CanonicalEvidenceArchive.build(original, candidates)
    feasibility = build_feasibility_explanation(
        original,
        original_archive,
        provenance=FeasibilityProvenance(original.contract_id, SOURCE, original_archive.archive_id),
    )
    proposal = propose_objective_amendment(
        original,
        proposed,
        rationale="the approved soft priority changed",
        evidence=feasibility,
        operator_id="operator_fixture",
        requested_at="2026-09-07T10:00:00+00:00",
        expires_at="2026-09-07T11:00:00+00:00",
        parent_campaign_id="campaign_original",
        session_id="session_fixture",
        run_id="run_fixture",
    )
    approved = approve_objective_amendment(
        proposal,
        operator_id="operator_fixture",
        decided_at="2026-09-07T10:05:00+00:00",
    )
    application = apply_objective_amendment(
        approved,
        current_objective=original,
        current_campaign_id="campaign_original",
        applied_at="2026-09-07T10:06:00+00:00",
    )
    amended = _build(
        proposed,
        candidates,
        amendment=application,
    )

    assert amended.objective_identity.original_objective_id == original.contract_id
    assert amended.objective_identity.effective_objective_id == proposed.contract_id
    assert amended.objective_identity.amendment_id == application.amendment_id
    assert amended.objective_identity.downstream_campaign_id == application.downstream_campaign_id
    assert amended.approval_id == application.approval_id
    assert amended.rationale != _build(original, candidates).rationale
    assert proposed.contract_id in amended.rationale

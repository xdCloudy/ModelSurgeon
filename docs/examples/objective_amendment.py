"""Small deterministic example for an explicit objective amendment."""

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
    apply_objective_amendment,
    approve_objective_amendment,
    propose_objective_amendment,
)

SOURCE = "sha256:" + "a" * 64


def contract(weight: float = 1.0) -> ObjectiveContract:
    return ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                ContractConstraintDirection.MINIMUM,
                0.95,
                MetricUnit.RATIO,
            ),
            HardConstraint(
                ContractMetric.LATENCY,
                ContractConstraintDirection.MAXIMUM,
                100.0,
                MetricUnit.MILLISECONDS,
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
    )


def main() -> None:
    original = contract()
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
    archive = CanonicalEvidenceArchive.build(original, (candidate,))
    explanation = build_feasibility_explanation(
        original,
        archive,
        provenance=FeasibilityProvenance(original.contract_id, SOURCE, archive.archive_id),
    )
    proposal = propose_objective_amendment(
        original,
        contract(weight=2.0),
        rationale="the measured near miss is an explicit user-approved trade-off",
        evidence=explanation,
        operator_id="operator-alice",
        requested_at="2026-09-07T10:00:00+00:00",
        expires_at="2026-09-07T11:00:00+00:00",
        parent_campaign_id="campaign_original",
        session_id="session-example",
        run_id="run-example",
    )
    approved = approve_objective_amendment(
        proposal,
        operator_id="operator-alice",
        decided_at="2026-09-07T10:05:00+00:00",
    )
    application = apply_objective_amendment(
        approved,
        current_objective=original,
        current_campaign_id="campaign_original",
        applied_at="2026-09-07T10:06:00+00:00",
    )
    print(application.to_record())


if __name__ == "__main__":
    main()

"""Small deterministic example for measured infeasibility explanations."""

from modelsurgeon.explain import (
    CandidateEvidenceStatus,
    CanonicalEvidenceArchive,
    FeasibilityCandidateEvidence,
    FeasibilityProvenance,
    build_feasibility_explanation,
)
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    ContractMetric,
    HardConstraint,
    MetricObservation,
    MetricUnit,
    ObjectiveContract,
    ObjectiveDirection,
    SoftObjective,
)


def main() -> None:
    contract = ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                ConstraintDirection.MINIMUM,
                0.95,
                MetricUnit.RATIO,
            ),
            HardConstraint(
                ContractMetric.LATENCY,
                ConstraintDirection.MAXIMUM,
                100.0,
                MetricUnit.MILLISECONDS,
            ),
        ),
        objectives=(
            SoftObjective(
                ContractMetric.QUALITY,
                ObjectiveDirection.MAXIMIZE,
                MetricUnit.RATIO,
                baseline=1.0,
            ),
        ),
    )
    source_digest = "sha256:" + "a" * 64
    candidate = FeasibilityCandidateEvidence(
        "candidate_tiny",
        "evidence_tiny",
        CandidateEvidenceStatus.MEASURED,
        source_digest,
        (
            MetricObservation(ContractMetric.LATENCY, 101.0, MetricUnit.MILLISECONDS),
            MetricObservation(ContractMetric.QUALITY, 0.94, MetricUnit.RATIO),
        ),
        evaluation_id="evaluation_tiny",
        provenance={"runtime": "fixture-v1", "dataset": "heldout-tiny"},
    )
    archive = CanonicalEvidenceArchive.build(contract, (candidate,))
    result = build_feasibility_explanation(
        contract,
        archive,
        provenance=FeasibilityProvenance(
            "spec_tiny",
            source_digest,
            archive.archive_id,
            approval_id="approval_tiny",
            approval_provenance={"operator": "fixture"},
        ),
    )
    print(result.outcome.value)
    for item in result.closest_candidates:
        print(item.candidate_id, item.distance)


if __name__ == "__main__":
    main()

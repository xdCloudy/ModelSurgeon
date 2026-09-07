"""Small deterministic example for measured Pareto final selection."""

from modelsurgeon.explain import (
    CandidateDisposition,
    CandidateEvidenceStatus,
    CanonicalEvidenceArchive,
    FeasibilityCandidateEvidence,
    FeasibilityProvenance,
    build_pareto_selection_explanation,
    render_pareto_selection_explanation,
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
)

SOURCE = "sha256:" + "b" * 64


def main() -> None:
    contract = ObjectiveContract(
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
                baseline=1.0,
            ),
            SoftObjective(
                ContractMetric.LATENCY,
                ContractObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                normalization=ContractObjectiveNormalization.IDENTITY,
            ),
        ),
        mode=ObjectiveMode.PARETO,
    )
    candidates = (
        FeasibilityCandidateEvidence(
            "candidate_fast",
            "evidence_fast",
            CandidateEvidenceStatus.MEASURED,
            SOURCE,
            observations=(
                MetricObservation(ContractMetric.LATENCY, 80.0, MetricUnit.MILLISECONDS),
                MetricObservation(ContractMetric.QUALITY, 0.95, MetricUnit.RATIO),
            ),
            disposition=CandidateDisposition.ACCEPTED,
        ),
        FeasibilityCandidateEvidence(
            "candidate_quality",
            "evidence_quality",
            CandidateEvidenceStatus.MEASURED,
            SOURCE,
            observations=(
                MetricObservation(ContractMetric.LATENCY, 110.0, MetricUnit.MILLISECONDS),
                MetricObservation(ContractMetric.QUALITY, 0.98, MetricUnit.RATIO),
            ),
            disposition=CandidateDisposition.ACCEPTED,
        ),
    )
    archive = CanonicalEvidenceArchive.build(contract, candidates)
    result = build_pareto_selection_explanation(
        contract,
        archive,
        provenance=FeasibilityProvenance(
            contract.contract_id,
            SOURCE,
            archive.archive_id,
            approval_id="approval_example",
            approval_provenance={"operator": "operator_example"},
        ),
    )
    print(render_pareto_selection_explanation(result, format="chat"))


if __name__ == "__main__":
    main()

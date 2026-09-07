"""Small deterministic example for measured Pareto alternatives."""

from modelsurgeon.explain import (
    CandidateDisposition,
    CandidateEvidenceStatus,
    CanonicalEvidenceArchive,
    FeasibilityCandidateEvidence,
    FeasibilityProvenance,
    build_pareto_alternatives,
    render_pareto_explanation,
)
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    ContractMetric,
    HardConstraint,
    MetricObservation,
    MetricUnit,
    ObjectiveContract,
    ObjectiveDirection,
    ObjectiveNormalization,
    SoftObjective,
)


def main() -> None:
    contract = make_contract()
    candidates = make_measured_candidates()
    archive = CanonicalEvidenceArchive.build(contract, candidates)
    result = build_pareto_alternatives(
        contract,
        archive,
        provenance=FeasibilityProvenance(
            "spec_tiny_pareto",
            "sha256:" + "a" * 64,
            archive.archive_id,
            approval_id="approval_tiny_pareto",
        ),
    )
    print(render_pareto_explanation(result, format="chat"))


def make_contract() -> ObjectiveContract:
    return ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                ConstraintDirection.MINIMUM,
                0.95,
                MetricUnit.RATIO,
            ),
        ),
        objectives=(
            SoftObjective(
                ContractMetric.QUALITY,
                ObjectiveDirection.MAXIMIZE,
                MetricUnit.RATIO,
                baseline=1.0,
            ),
            SoftObjective(
                ContractMetric.LATENCY,
                ObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                normalization=ObjectiveNormalization.IDENTITY,
            ),
        ),
    )


def make_measured_candidates() -> tuple[FeasibilityCandidateEvidence, ...]:
    source = "sha256:" + "a" * 64

    def candidate(name: str, quality: float, latency: float) -> FeasibilityCandidateEvidence:
        observations = tuple(
            sorted(
                (
                    MetricObservation(ContractMetric.QUALITY, quality, MetricUnit.RATIO),
                    MetricObservation(ContractMetric.LATENCY, latency, MetricUnit.MILLISECONDS),
                ),
                key=lambda item: item.metric,
            )
        )
        return FeasibilityCandidateEvidence(
            f"candidate_{name}",
            f"evidence_{name}",
            CandidateEvidenceStatus.MEASURED,
            source,
            observations,
            disposition=CandidateDisposition.ACCEPTED,
            evaluation_id=f"evaluation_{name}",
            provenance={"fixture": "tiny-pareto-v1"},
        )

    return (candidate("fast", 0.95, 80.0), candidate("quality", 0.97, 110.0))


if __name__ == "__main__":
    main()

"""Measured feasibility and fail-closed infeasibility explanations."""

from __future__ import annotations

from modelsurgeon.explain import (
    CandidateDisposition,
    CandidateEvidenceStatus,
    CanonicalEvidenceArchive,
    FeasibilityCandidateEvidence,
    FeasibilityOutcome,
    FeasibilityProvenance,
    FeasibilityResourceBounds,
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

SOURCE = "sha256:" + "a" * 64


def _contract() -> ObjectiveContract:
    return ObjectiveContract(
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


def _candidate(
    name: str,
    *,
    quality: float | None = None,
    latency: float | None = None,
    status: CandidateEvidenceStatus = CandidateEvidenceStatus.MEASURED,
    disposition: CandidateDisposition = CandidateDisposition.UNKNOWN,
    quality_lower: float | None = None,
) -> FeasibilityCandidateEvidence:
    observations = []
    if quality is not None:
        observations.append(
            MetricObservation(
                ContractMetric.QUALITY,
                quality,
                MetricUnit.RATIO,
                lower=quality_lower,
            )
        )
    if latency is not None:
        observations.append(
            MetricObservation(ContractMetric.LATENCY, latency, MetricUnit.MILLISECONDS)
        )
    if status is CandidateEvidenceStatus.PREDICTED:
        predicted = tuple(sorted(observations, key=lambda item: item.metric))
        measured = ()
    else:
        measured = tuple(sorted(observations, key=lambda item: item.metric))
        predicted = ()
    return FeasibilityCandidateEvidence(
        f"candidate_{name}",
        f"evidence_{name}",
        status,
        SOURCE,
        measured,
        predicted,
        disposition,
        f"evaluation_{name}",
        {"record_type": "fixture_candidate", "source": "test"},
    )


def _explain(candidates: tuple[FeasibilityCandidateEvidence, ...]):
    contract = _contract()
    archive = CanonicalEvidenceArchive.build(contract, candidates)
    provenance = FeasibilityProvenance(
        "spec_fixture",
        SOURCE,
        archive.archive_id,
        approval_id="approval_fixture",
        approval_provenance={"record_type": "fixture_approval", "operator": "test"},
    )
    return build_feasibility_explanation(contract, archive, provenance=provenance)


def test_no_candidate_evidence_is_missing_not_infeasible() -> None:
    result = _explain(())

    assert result.outcome is FeasibilityOutcome.MISSING_EVIDENCE
    assert not result.infeasible
    assert result.measured_candidate_ids == ()
    assert result.missing_metrics == ("latency", "quality")
    assert result.provenance.approval_id == "approval_fixture"
    assert result.retained_evidence == ()
    assert result.next_actions[0].kind.value == "collect_measurements"


def test_missing_hard_constraint_evidence_is_distinct_from_infeasible() -> None:
    result = _explain((_candidate("partial", quality=0.90),))

    assert result.outcome is FeasibilityOutcome.MISSING_EVIDENCE
    assert result.missing_metrics == ("latency",)
    assert result.unmet_constraints == ()
    assert result.closest_candidates == ()


def test_predictions_never_become_measured_feasibility() -> None:
    result = _explain(
        (
            _candidate(
                "predicted",
                quality=0.99,
                latency=10.0,
                status=CandidateEvidenceStatus.PREDICTED,
            ),
        )
    )

    assert result.outcome is FeasibilityOutcome.PREDICTED_ONLY
    assert result.feasible_candidate_ids == ()
    assert result.measured_candidate_ids == ()
    assert result.next_actions[0].kind.value == "measure_predicted_candidates"


def test_measured_near_misses_report_unmet_constraints_and_conservative_distance() -> None:
    result = _explain(
        (
            _candidate("far", quality=0.80, latency=130.0),
            _candidate("near", quality=0.94, latency=101.0),
        )
    )

    assert result.outcome is FeasibilityOutcome.INFEASIBLE
    assert result.hard_constraints == _contract().constraints
    assert tuple(item.candidate_id for item in result.closest_candidates) == (
        "candidate_near",
        "candidate_far",
    )
    assert {item.constraint.metric for item in result.unmet_constraints} == {
        ContractMetric.QUALITY,
        ContractMetric.LATENCY,
    }
    assert result.closest_candidates[0].distance < result.closest_candidates[1].distance
    assert any(
        item.kind.value == "propose_explicit_objective_amendment"
        and item.requires_approval
        for item in result.next_actions
    )


def test_ordering_and_replay_are_independent_of_archive_arrival_order() -> None:
    candidates = (
        _candidate("z", quality=0.94, latency=101.0),
        _candidate("a", quality=0.94, latency=101.0),
    )

    first = _explain(candidates)
    second = _explain(tuple(reversed(candidates)))

    assert first.canonical_json() == second.canonical_json()
    assert tuple(item.candidate_id for item in first.closest_candidates) == (
        "candidate_a",
        "candidate_z",
    )


def test_rejected_and_rolled_back_measured_evidence_remains_visible() -> None:
    result = _explain(
        (
            _candidate(
                "rejected",
                quality=0.94,
                latency=101.0,
                disposition=CandidateDisposition.REJECTED,
            ),
            _candidate(
                "rollback",
                quality=0.93,
                latency=105.0,
                disposition=CandidateDisposition.ROLLED_BACK,
            ),
        )
    )

    assert result.outcome is FeasibilityOutcome.INFEASIBLE
    assert {item.candidate_id for item in result.closest_candidates} == {
        "candidate_rejected",
        "candidate_rollback",
    }
    assert {item.disposition for item in result.closest_candidates} == {
        CandidateDisposition.REJECTED,
        CandidateDisposition.ROLLED_BACK,
    }


def test_terminal_negative_statuses_are_retained_without_becoming_infeasible() -> None:
    result = _explain(
        (
            _candidate("failed", status=CandidateEvidenceStatus.FAILED),
            _candidate("inconclusive", status=CandidateEvidenceStatus.INCONCLUSIVE),
            _candidate("unknown", status=CandidateEvidenceStatus.UNKNOWN),
            _candidate("unsupported", status=CandidateEvidenceStatus.UNSUPPORTED),
        )
    )

    assert result.outcome is FeasibilityOutcome.UNSUPPORTED
    assert {item.status for item in result.retained_evidence} == {
        CandidateEvidenceStatus.FAILED,
        CandidateEvidenceStatus.INCONCLUSIVE,
        CandidateEvidenceStatus.UNKNOWN,
        CandidateEvidenceStatus.UNSUPPORTED,
    }


def test_uncertainty_is_retained_and_can_make_a_candidate_infeasible() -> None:
    result = _explain(
        (_candidate("uncertain", quality=0.96, quality_lower=0.94, latency=90.0),)
    )

    assert result.outcome is FeasibilityOutcome.INFEASIBLE
    assert result.uncertainty[0].metric == ContractMetric.QUALITY
    assert result.closest_candidates[0].uncertain_metrics == (ContractMetric.QUALITY,)


def test_resource_bounds_are_reported_and_enforced() -> None:
    result = _explain((_candidate("one", quality=0.90, latency=101.0),))
    assert result.resource_usage.evidence_records == 1
    assert result.resource_usage.output_bytes <= result.resource_bounds.max_output_bytes

    contract = _contract()
    archive = CanonicalEvidenceArchive.build(
        contract,
        (_candidate("one", quality=0.90, latency=101.0),),
    )
    provenance = FeasibilityProvenance("spec_fixture", SOURCE, archive.archive_id)
    too_small = FeasibilityResourceBounds(max_output_bytes=100)
    try:
        build_feasibility_explanation(
            contract, archive, provenance=provenance, resource_bounds=too_small
        )
    except ValueError as error:
        assert "output bound" in str(error)
    else:
        raise AssertionError("small output bound must fail closed")

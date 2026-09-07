import pytest

from modelsurgeon.config import ObjectiveDirection
from modelsurgeon.search.hardware_objectives import (
    ConstraintDirection,
    DeploymentObjective,
    EvidenceStatus,
    HardwareCandidate,
    HardwareConstraint,
    HardwareDecisionStatus,
    HardwareEvidence,
    HardwareLineageRecord,
    HardwareMetric,
    HardwareObjectiveError,
    HardwareSearchReport,
    HardwareSearchResume,
    OffloadBoundary,
    assess_hardware_candidate,
    rank_hardware_candidates,
)


def _placement(*, gpu: bool = True, fraction: float = 0.5) -> OffloadBoundary:
    return OffloadBoundary("profile_test", "runtime-test", 4, gpu, 8 if gpu else 0, fraction)


def _candidate(
    name: str,
    *,
    placement: OffloadBoundary | None = None,
    predicted: tuple[HardwareEvidence, ...] = (),
    measured: tuple[HardwareEvidence, ...] = (),
    unknown_evidence: tuple[HardwareEvidence, ...] = (),
) -> HardwareCandidate:
    return HardwareCandidate(
        name,
        "checkpoint_root",
        placement or _placement(),
        predicted,
        measured,
        unknown_evidence,
    )


def test_placement_is_part_of_candidate_identity_and_persisted_records() -> None:
    cpu = _candidate(
        "same-structure",
        placement=OffloadBoundary("profile_test", "runtime-test", 4, False),
    )
    gpu = _candidate("same-structure")
    assert cpu.candidate_id != gpu.candidate_id
    assert cpu.to_record()["placement"]["boundary_id"] == cpu.placement.boundary_id
    resume = HardwareSearchResume("search_test", 2, gpu.placement, (gpu.candidate_id,), ())
    assert resume.to_record()["profile"]["profile_id"] == "profile_test"


def test_predicted_feasibility_never_promotes_without_measured_hard_evidence() -> None:
    constraint = HardwareConstraint(
        HardwareMetric.PEAK_VRAM_BYTES,
        ConstraintDirection.MAXIMUM,
        10,
    )
    candidate = _candidate(
        "predicted",
        predicted=(
            HardwareEvidence(
                HardwareMetric.PEAK_VRAM_BYTES,
                8,
                EvidenceStatus.PREDICTED,
                1,
                "predictor_v1",
            ),
        ),
    )
    assessment = assess_hardware_candidate(candidate, (constraint,))
    assert assessment.status is HardwareDecisionStatus.PREDICTED_ONLY
    assert assessment.promotable is False
    assert assessment.constraint_results[0].reason == "measured_hard_constraint_required"


def test_conservative_bounds_can_reject_a_nominally_feasible_measurement() -> None:
    constraint = HardwareConstraint(
        HardwareMetric.LATENCY_SECONDS,
        ConstraintDirection.MAXIMUM,
        10,
    )
    candidate = _candidate(
        "measured",
        measured=(
            HardwareEvidence(
                HardwareMetric.LATENCY_SECONDS,
                9,
                EvidenceStatus.MEASURED,
                2,
                "bench-1",
            ),
        ),
    )
    assessment = assess_hardware_candidate(candidate, (constraint,))
    assert assessment.status is HardwareDecisionStatus.INFEASIBLE
    assert assessment.constraint_results[0].conservative_value == 11


def test_requested_deployment_objective_beats_parameter_reduction() -> None:
    objective = DeploymentObjective(
        HardwareMetric.LATENCY_SECONDS,
        ObjectiveDirection.MINIMIZE,
        reference=1,
    )
    constraint = HardwareConstraint(
        HardwareMetric.LATENCY_SECONDS,
        ConstraintDirection.MAXIMUM,
        20,
    )
    compact = _candidate(
        "compact",
        measured=(
            HardwareEvidence(
                HardwareMetric.LATENCY_SECONDS, 12, EvidenceStatus.MEASURED, 0, "bench-c"
            ),
            HardwareEvidence(
                HardwareMetric.PARAMETER_COUNT, 10, EvidenceStatus.MEASURED, 0, "bench-c"
            ),
        ),
    )
    fast = _candidate(
        "fast",
        measured=(
            HardwareEvidence(
                HardwareMetric.LATENCY_SECONDS, 6, EvidenceStatus.MEASURED, 0, "bench-f"
            ),
            HardwareEvidence(
                HardwareMetric.PARAMETER_COUNT, 20, EvidenceStatus.MEASURED, 0, "bench-f"
            ),
        ),
    )
    ranked = rank_hardware_candidates((compact, fast), (constraint,), (objective,))
    assert ranked[0][0] is fast
    assert all(item[1].promotable for item in ranked)


def test_report_and_lineage_retain_hardware_context() -> None:
    measured = HardwareEvidence(
        HardwareMetric.LATENCY_SECONDS, 4, EvidenceStatus.MEASURED, 0, "bench"
    )
    candidate = _candidate("accepted", measured=(measured,))
    constraint = HardwareConstraint(HardwareMetric.LATENCY_SECONDS, ConstraintDirection.MAXIMUM, 5)
    assessment = assess_hardware_candidate(candidate, (constraint,))
    report = HardwareSearchReport("search_test", candidate.placement, (candidate,), (assessment,))
    record = report.to_record()
    assert record["candidates"][0]["placement"]["profile_id"] == "profile_test"
    lineage = HardwareLineageRecord(candidate, "checkpoint_root", "checkpoint_child", assessment)
    assert lineage.to_record()["candidate"]["candidate_id"] == candidate.candidate_id


def test_invalid_acceptance_and_profile_combinations_fail_closed() -> None:
    candidate = _candidate("unknown")
    assessment = assess_hardware_candidate(
        candidate,
        (HardwareConstraint(HardwareMetric.LATENCY_SECONDS, ConstraintDirection.MAXIMUM, 5),),
    )
    with pytest.raises(HardwareObjectiveError, match="non-promotable"):
        HardwareLineageRecord(candidate, "checkpoint_root", "checkpoint_child", assessment)
    with pytest.raises(HardwareObjectiveError, match="CPU-only"):
        OffloadBoundary("cpu", "runtime-test", 2, False, 1, 0.1)


def test_unknown_evidence_is_retained_as_unknown() -> None:
    candidate = _candidate(
        "unknown",
        unknown_evidence=(
            HardwareEvidence(HardwareMetric.LATENCY_SECONDS, 0, EvidenceStatus.UNKNOWN),
        ),
    )
    assessment = assess_hardware_candidate(
        candidate,
        (HardwareConstraint(HardwareMetric.LATENCY_SECONDS, ConstraintDirection.MAXIMUM, 5),),
    )
    assert assessment.status is HardwareDecisionStatus.UNKNOWN

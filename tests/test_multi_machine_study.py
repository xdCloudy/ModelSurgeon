import pytest

from modelsurgeon.config import ObjectiveDirection
from modelsurgeon.evaluation.multi_machine_study import (
    MachineClass,
    MachineProfile,
    MultiMachineStudyError,
    ObservationStatus,
    StudyArm,
    StudyClaim,
    StudyObjective,
    StudyObservation,
    build_multi_machine_study,
)


def _profiles() -> tuple[MachineProfile, ...]:
    return (
        MachineProfile("cpu", MachineClass.CPU_ONLY, "runtime-cpu", 8, 0, "cpu-a", "env-a"),
        MachineProfile("low-vram", MachineClass.LOW_VRAM, "runtime-gpu", 8, 4, "gpu-a", "env-b"),
        MachineProfile("full-gpu", MachineClass.FULL_GPU, "runtime-gpu", 16, 24, "gpu-b", "env-c"),
    )


def _observations(
    *,
    unsupported: bool = False,
) -> tuple[StudyObservation, ...]:
    rows: list[StudyObservation] = []
    for profile in _profiles():
        for candidate in ("candidate_a", "candidate_b"):
            for arm in StudyArm:
                for repetition in range(3):
                    if (
                        unsupported
                        and profile.profile_id == "full-gpu"
                        and candidate == "candidate_b"
                    ):
                        rows.append(
                            StudyObservation(
                                profile.profile_id,
                                candidate,
                                arm,
                                "latency",
                                repetition,
                                ObservationStatus.UNSUPPORTED,
                                reason="runtime codec unavailable",
                            )
                        )
                        continue
                    if arm is StudyArm.HARDWARE_BLIND:
                        value = 5.0 if candidate == "candidate_a" else 5.5
                    elif profile.profile_id == "cpu":
                        value = 5.0 if candidate == "candidate_a" else 4.0
                    elif profile.profile_id == "low-vram":
                        value = 4.0 if candidate == "candidate_a" else 6.0
                    else:
                        value = 5.0
                    rows.append(
                        StudyObservation(
                            profile.profile_id,
                            candidate,
                            arm,
                            "latency",
                            repetition,
                            ObservationStatus.MEASURED,
                            value=value,
                        )
                    )
    return tuple(rows)


def _build(*, unsupported: bool = False):
    return build_multi_machine_study(
        profiles=_profiles(),
        candidate_ids=("candidate_a", "candidate_b"),
        objectives=(StudyObjective("latency", ObjectiveDirection.MINIMIZE),),
        observations=_observations(unsupported=unsupported),
        source_artifact_digest="source-model-v1",
        corpus_identity="corpus-v1",
        optimization_budget_id="budget-v1",
    )


def test_three_profiles_compare_hardware_aware_and_blind_selection() -> None:
    study = _build()
    assert study.claim is StudyClaim.POSITIVE
    assert study.hardware_specific_choices is True
    assert {item.hardware_aware_candidate_id for item in study.comparisons} == {
        "candidate_a",
        "candidate_b",
    }
    assert all(item.aware_beats_or_ties is True for item in study.comparisons)
    record = study.to_record()
    assert record["computed_study_id"] == study.study_id
    assert len(record["profiles"]) == 3


def test_unsupported_profile_is_retained_and_does_not_make_a_positive_claim() -> None:
    study = _build(unsupported=True)
    assert study.claim is StudyClaim.INCONCLUSIVE
    unsupported = study.comparisons[-1]
    assert unsupported.status.value == "unknown"
    assert unsupported.reason == "insufficient_measured_evidence"


def test_profiles_and_repetitions_are_required() -> None:
    with pytest.raises(MultiMachineStudyError, match="three profiles"):
        build_multi_machine_study(
            profiles=_profiles()[:2],
            candidate_ids=("candidate_a", "candidate_b"),
            objectives=(StudyObjective("latency", ObjectiveDirection.MINIMIZE),),
            observations=_observations(),
            source_artifact_digest="source-model-v1",
            corpus_identity="corpus-v1",
            optimization_budget_id="budget-v1",
        )
    with pytest.raises(MultiMachineStudyError, match="three repetitions"):
        build_multi_machine_study(
            profiles=_profiles(),
            candidate_ids=("candidate_a", "candidate_b"),
            objectives=(StudyObjective("latency", ObjectiveDirection.MINIMIZE),),
            observations=_observations()[:-1],
            source_artifact_digest="source-model-v1",
            corpus_identity="corpus-v1",
            optimization_budget_id="budget-v1",
        )

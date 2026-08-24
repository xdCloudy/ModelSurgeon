import pytest

from modelsurgeon.evaluation.iterative_search_study import (
    ArmOutcome,
    BackendStudy,
    IterativeSearchStudy,
    IterativeSearchStudyError,
    SearchGeneration,
    SearchGoals,
    StudyArm,
    StudyBackend,
    StudyMeasurement,
    compare_arm,
)


def _measurement(
    name: str,
    backend: StudyBackend,
    quality: float,
    latency: float,
    size: int,
) -> StudyMeasurement:
    return StudyMeasurement(
        f"measurement_{name}",
        backend,
        () if name == "baseline" else (1,),
        quality,
        latency,
        size,
        100,
        1.0,
        1.0 if backend is StudyBackend.HUGGING_FACE else 0.0,
        1.0 if backend is StudyBackend.NATIVE_GGUF else 0.0,
        1000,
        200 if backend is StudyBackend.HUGGING_FACE else 0,
        None,
    )


def _backend(backend: StudyBackend) -> BackendStudy:
    baseline = _measurement("baseline", backend, 2.0, 1.0, 1000)
    no_repair = _measurement("no_repair", backend, 2.1, 0.9, 900)
    repair = _measurement("repair", backend, 2.05, 0.92, 900)
    one_shot = _measurement("one_shot", backend, 2.2, 0.88, 900)
    goals = SearchGoals(0.15, 0.05, 0.05)
    return BackendStudy(
        backend,
        "loss" if backend is StudyBackend.HUGGING_FACE else "perplexity",
        goals,
        baseline.measurement_id,
        (
            SearchGeneration(
                1,
                baseline.measurement_id,
                (no_repair.measurement_id,),
                no_repair.measurement_id,
            ),
        ),
        (
            compare_arm(StudyArm.NO_REPAIR, baseline, no_repair, goals),
            compare_arm(StudyArm.REPAIR, baseline, repair, goals),
            compare_arm(StudyArm.ONE_SHOT, baseline, one_shot, goals),
        ),
        (baseline, no_repair, repair, one_shot),
    )


def test_two_backend_study_compares_goals_and_sums_unique_measurement_cost() -> None:
    study = IterativeSearchStudy(
        42,
        (_backend(StudyBackend.HUGGING_FACE), _backend(StudyBackend.NATIVE_GGUF)),
    )
    record = study.to_record()
    assert record["total_experiment_cost"] == {
        "wall_seconds": 8.0,
        "gpu_seconds": 4.0,
        "cpu_seconds": 4.0,
    }
    hf = study.backends[0]
    outcomes = {item.arm: item for item in hf.outcomes}
    assert outcomes[StudyArm.NO_REPAIR].all_goals_met is True
    assert outcomes[StudyArm.REPAIR].all_goals_met is True
    assert outcomes[StudyArm.ONE_SHOT].quality_goal_met is False


def test_missing_arm_and_cross_backend_comparison_fail_closed() -> None:
    hf = _measurement("baseline", StudyBackend.HUGGING_FACE, 2, 1, 1000)
    native = _measurement("native", StudyBackend.NATIVE_GGUF, 2, 1, 1000)
    with pytest.raises(IterativeSearchStudyError, match="backends"):
        compare_arm(StudyArm.NO_REPAIR, hf, native, SearchGoals(1, 0, 0))
    valid = _backend(StudyBackend.HUGGING_FACE)
    with pytest.raises(IterativeSearchStudyError, match="three study arms"):
        BackendStudy(
            valid.backend,
            valid.quality_metric,
            valid.goals,
            valid.baseline_measurement_id,
            valid.generations,
            (ArmOutcome(StudyArm.NO_REPAIR, "measurement_no_repair", 0, 0, 0, True, True, True),),
            valid.measurements,
        )


def test_invalid_lineage_and_derived_outcome_fail_closed() -> None:
    valid = _backend(StudyBackend.HUGGING_FACE)
    generation = valid.generations[0]
    broken_lineage = SearchGeneration(
        generation.generation,
        generation.selected_measurement_id,
        generation.candidate_measurement_ids,
        generation.selected_measurement_id,
    )
    with pytest.raises(IterativeSearchStudyError, match="lineage"):
        BackendStudy(
            valid.backend,
            valid.quality_metric,
            valid.goals,
            valid.baseline_measurement_id,
            (broken_lineage,),
            valid.outcomes,
            valid.measurements,
        )
    outcomes = list(valid.outcomes)
    original = outcomes[0]
    outcomes[0] = ArmOutcome(
        original.arm,
        original.measurement_id,
        original.quality_increase + 1,
        original.latency_gain_ratio,
        original.size_gain_ratio,
        original.quality_goal_met,
        original.latency_goal_met,
        original.size_goal_met,
    )
    with pytest.raises(IterativeSearchStudyError, match="disagrees"):
        BackendStudy(
            valid.backend,
            valid.quality_metric,
            valid.goals,
            valid.baseline_measurement_id,
            valid.generations,
            tuple(outcomes),
            valid.measurements,
        )

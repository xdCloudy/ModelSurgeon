"""Tests for hardware-conditioned surgeon predictions and held-out ablations."""

from __future__ import annotations

from modelsurgeon.surgeon.hardware_conditioning import (
    HardwareConditionedSample,
    HardwarePredictionStatus,
    HardwareProfileContext,
    HardwareStudyStatus,
    evaluate_hardware_ablation,
    fit_hardware_conditioned_model,
)


def _profile(name: str, vram: float, gpu: bool) -> HardwareProfileContext:
    return HardwareProfileContext(
        name, "runtime-v1", 8 if gpu else 4, vram, gpu, 1.0 if gpu else 0.0, "Q4_K"
    )


def _samples() -> tuple[HardwareConditionedSample, ...]:
    cpu = _profile("cpu", 0.0, False)
    gpu = _profile("gpu", 24.0, True)
    return (
        HardwareConditionedSample("a-cpu", "a", (1.0, 0.0), cpu, 0.8, 0.9, 0.2, 1.0),
        HardwareConditionedSample("b-cpu", "b", (2.0, 0.0), cpu, 0.7, 0.8, 0.3, 2.0),
        HardwareConditionedSample("a-gpu", "a", (1.0, 0.0), gpu, 0.8, 0.9, 0.9, 100.0),
        HardwareConditionedSample("b-gpu", "b", (2.0, 0.0), gpu, 0.7, 0.8, 1.0, 200.0),
    )


def test_same_candidate_gets_profile_conditioned_utility() -> None:
    model = fit_hardware_conditioned_model(_samples())
    state = (1.0, 0.0)
    cpu = model.predict(state, _profile("cpu", 0.0, False))
    gpu = model.predict(state, _profile("gpu", 24.0, True))
    assert cpu.status is HardwarePredictionStatus.SUPPORTED
    assert gpu.status is HardwarePredictionStatus.SUPPORTED
    assert cpu.utility != gpu.utility


def test_quality_path_ignores_post_mutation_runtime_target() -> None:
    samples = _samples()
    first = fit_hardware_conditioned_model(samples)
    changed = tuple(
        HardwareConditionedSample(
            item.candidate_id,
            item.lineage_group_id,
            item.pre_mutation_features,
            item.profile,
            item.quality_target,
            item.safety_target,
            item.utility_target,
            999999.0,
        )
        for item in samples
    )
    second = fit_hardware_conditioned_model(changed)
    profile = _profile("gpu", 24.0, True)
    assert first.predict((1.0, 0.0), profile).quality == second.predict((1.0, 0.0), profile).quality


def test_heldout_ablation_retains_measured_or_null_result() -> None:
    samples = _samples()
    conditional = fit_hardware_conditioned_model(samples, excluded_profile_ids=("gpu",))
    baseline = fit_hardware_conditioned_model(
        samples, include_hardware_context=False, excluded_profile_ids=("gpu",)
    )
    report = evaluate_hardware_ablation(conditional, baseline, samples, ("gpu",))
    assert report.status in {HardwareStudyStatus.MEASURED, HardwareStudyStatus.NULL_RESULT}
    assert report.conditional_mae is not None
    assert report.baseline_mae is not None

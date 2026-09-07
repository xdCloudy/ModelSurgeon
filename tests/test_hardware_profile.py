from __future__ import annotations

import json

import pytest

from modelsurgeon.experiments.hardware import (
    CPUInventory,
    CUDAInventory,
    DiskInventory,
    HardwareInventory,
    MemoryInventory,
    SoftwareInventory,
)
from modelsurgeon.experiments.hardware_profile import (
    DEFAULT_HARDWARE_PROBE_PROTOCOL,
    HardwareProbeResult,
    HardwareProfileError,
    HardwareProfileSettings,
    HardwareTargetClass,
    ProbeKind,
    ProbeOutcome,
    ProbeSample,
    RuntimeRevision,
    build_hardware_profile,
    render_hardware_profile_protocol,
    summarize_probe_samples,
)
from modelsurgeon.experiments.runtime_telemetry import HardwareNormalizationContext


def _inventory() -> HardwareInventory:
    return HardwareInventory(
        "Windows",
        "11",
        "build",
        CPUInventory("AMD64", "Test CPU", 8),
        MemoryInventory(16_000, 8_000),
        DiskInventory("C:/models", 100_000, 50_000),
        CUDAInventory(False, None, (), ()),
        SoftwareInventory("3.12.9", "CPython", "1.0.0", None),
    )


def _runtimes() -> tuple[RuntimeRevision, ...]:
    return (
        RuntimeRevision("modelsurgeon", "1.0.0", None, "package"),
        RuntimeRevision("llama.cpp", None, None, "unavailable"),
    )


def _unknown_probes(inventory: HardwareInventory) -> tuple[HardwareProbeResult, ...]:
    context_id = HardwareNormalizationContext.from_inventory(inventory).context_id
    return tuple(
        HardwareProbeResult(
            spec,
            context_id,
            ProbeOutcome.UNSUPPORTED,
            reason="capability is unavailable on this target",
        )
        for spec in DEFAULT_HARDWARE_PROBE_PROTOCOL.specs
    )


def test_cpu_only_profile_has_explicit_optional_capabilities_and_stable_identity() -> None:
    inventory = _inventory()
    profile = build_hardware_profile(
        profile_name="reference-cpu",
        target_class=HardwareTargetClass.CPU_ONLY,
        inventory=inventory,
        runtimes=_runtimes(),
        settings=HardwareProfileSettings(4, 4),
        probes=_unknown_probes(inventory),
    )

    record = profile.to_record()
    assert profile.profile_id == profile.profile_id
    assert record["profile_id"] == profile.profile_id
    assert record["inventory"]["cuda"]["available"] is False  # type: ignore[index]
    assert record["probes"][0]["outcome"] == ProbeOutcome.UNSUPPORTED.value  # type: ignore[index]
    assert json.loads(json.dumps(record)) == record
    assert render_hardware_profile_protocol() == render_hardware_profile_protocol()


@pytest.mark.parametrize("target", list(HardwareTargetClass))
def test_declared_target_classes_do_not_require_fabricated_gpu_fields(
    target: HardwareTargetClass,
) -> None:
    inventory = _inventory()
    profile = build_hardware_profile(
        profile_name=f"declared-{target.value}",
        target_class=target,
        inventory=inventory,
        runtimes=_runtimes(),
        settings=HardwareProfileSettings(4, 4),
        probes=_unknown_probes(inventory),
    )

    assert profile.inventory.cuda.devices == ()
    assert profile.settings.gpu_device_index is None


def test_probe_summary_uses_rank_p95_and_population_dispersion() -> None:
    samples = tuple(
        ProbeSample(value, "hwctx-fixture", "stable-environment") for value in range(1, 8)
    )

    summary = summarize_probe_samples(samples)

    assert summary.median == 4
    assert summary.p95 == 7
    assert summary.dispersion == pytest.approx(2.0)
    assert summary.repetitions == 7


def test_measured_probe_rejects_materially_drifting_environment() -> None:
    samples = tuple(
        ProbeSample(value, "hwctx-fixture", "stable" if value < 4 else "drifted")
        for value in range(1, 8)
    )

    with pytest.raises(HardwareProfileError, match="drifting"):
        HardwareProbeResult(
            DEFAULT_HARDWARE_PROBE_PROTOCOL.specs[0],
            "hwctx-fixture",
            ProbeOutcome.MEASURED,
            samples,
        )


def test_profile_rejects_probe_context_mismatch() -> None:
    inventory = _inventory()
    probes = tuple(
        HardwareProbeResult(
            spec,
            "wrong-context",
            ProbeOutcome.UNKNOWN,
            reason="fixture helper",
        )
        for spec in DEFAULT_HARDWARE_PROBE_PROTOCOL.specs
    )

    with pytest.raises(HardwareProfileError, match="context"):
        build_hardware_profile(
            profile_name="bad-context",
            target_class=HardwareTargetClass.CPU_ONLY,
            inventory=inventory,
            runtimes=_runtimes(),
            settings=HardwareProfileSettings(4, 4),
            probes=probes,
        )


def test_protocol_retains_all_required_probe_kinds() -> None:
    assert {spec.kind for spec in DEFAULT_HARDWARE_PROBE_PROTOCOL.specs} == set(ProbeKind)

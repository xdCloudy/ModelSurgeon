"""Tests for explicit device-capability mixed-precision selection."""

from __future__ import annotations

import pytest

from modelsurgeon.experiments import (
    CPUInventory,
    CUDAInventory,
    DiskInventory,
    GPUDeviceInventory,
    HardwareInventory,
    MemoryInventory,
    PrecisionDevice,
    PrecisionDType,
    PrecisionExecutionContext,
    PrecisionExecutionMode,
    PrecisionPolicyError,
    PrecisionRequest,
    PrecisionRequestKind,
    SoftwareInventory,
    bind_metric_precision,
    expected_execution_context,
    precision_capabilities_from_hardware,
    select_precision_policy,
)


def _hardware(
    *capabilities: str | None,
    cuda_available: bool | None = None,
) -> HardwareInventory:
    available = bool(capabilities) if cuda_available is None else cuda_available
    devices = tuple(
        GPUDeviceInventory(index, f"gpu-{index}", 12_000, capability)
        for index, capability in enumerate(capabilities)
    )
    cuda = (
        CUDAInventory(True, "12.8", ("590.1",), devices)
        if available
        else CUDAInventory(False, None, (), ())
    )
    return HardwareInventory(
        "Linux",
        "test",
        "test-version",
        CPUInventory("x86_64", "test-cpu", 8),
        MemoryInventory(32_000, 16_000),
        DiskInventory("/tmp", 100_000, 50_000),
        cuda,
        SoftwareInventory("3.12", "CPython", "0.0.1", None),
    )


def test_cpu_fp16_fails_closed_or_records_explicit_float32_fallback() -> None:
    capabilities = precision_capabilities_from_hardware(_hardware(), PrecisionDevice.CPU)
    request = PrecisionRequest(PrecisionRequestKind.FLOAT16)

    with pytest.raises(PrecisionPolicyError, match="float16 is unsupported"):
        select_precision_policy(request, capabilities)

    decision = select_precision_policy(
        PrecisionRequest(PrecisionRequestKind.FLOAT16, allow_fallback=True),
        capabilities,
    )
    assert decision.mode is PrecisionExecutionMode.DIRECT
    assert decision.compute_dtype is PrecisionDType.FLOAT32
    assert decision.accumulation_dtype is PrecisionDType.FLOAT32
    assert decision.fallback_applied
    assert decision.fallback_reason is not None
    assert "fell back to float32" in decision.fallback_reason
    assert decision.to_record()["fallback_reason"] == decision.fallback_reason


def test_ampere_autocast_prefers_bfloat16_and_records_float32_accumulation() -> None:
    capabilities = precision_capabilities_from_hardware(
        _hardware("8.6"),
        PrecisionDevice.CUDA,
    )
    assert capabilities.direct_dtypes == (
        PrecisionDType.FLOAT32,
        PrecisionDType.FLOAT16,
        PrecisionDType.BFLOAT16,
    )
    assert capabilities.autocast_dtypes == (
        PrecisionDType.FLOAT16,
        PrecisionDType.BFLOAT16,
    )

    decision = select_precision_policy(
        PrecisionRequest(PrecisionRequestKind.AUTOCAST),
        capabilities,
    )
    assert decision.mode is PrecisionExecutionMode.AUTOCAST
    assert decision.compute_dtype is PrecisionDType.BFLOAT16
    assert decision.accumulation_dtype is PrecisionDType.FLOAT32
    assert not decision.fallback_applied


def test_turing_and_mixed_gpu_contexts_use_common_denominator_float16() -> None:
    for hardware in (_hardware("7.5"), _hardware("8.0", "7.5")):
        capabilities = precision_capabilities_from_hardware(hardware, PrecisionDevice.CUDA)
        assert PrecisionDType.FLOAT16 in capabilities.autocast_dtypes
        assert PrecisionDType.BFLOAT16 not in capabilities.autocast_dtypes
        decision = select_precision_policy(
            PrecisionRequest(PrecisionRequestKind.AUTOCAST),
            capabilities,
        )
        assert decision.compute_dtype is PrecisionDType.FLOAT16
        assert decision.mode is PrecisionExecutionMode.AUTOCAST


def test_unknown_cuda_capability_never_guesses_low_precision_support() -> None:
    capabilities = precision_capabilities_from_hardware(
        _hardware(None),
        PrecisionDevice.CUDA,
    )
    assert capabilities.direct_dtypes == (PrecisionDType.FLOAT32,)
    assert capabilities.autocast_dtypes == ()

    with pytest.raises(PrecisionPolicyError, match="autocast is unsupported"):
        select_precision_policy(
            PrecisionRequest(PrecisionRequestKind.AUTOCAST),
            capabilities,
        )
    decision = select_precision_policy(
        PrecisionRequest(PrecisionRequestKind.AUTOCAST, allow_fallback=True),
        capabilities,
    )
    assert decision.compute_dtype is PrecisionDType.FLOAT32
    assert decision.mode is PrecisionExecutionMode.DIRECT
    assert decision.fallback_reason is not None


def test_cuda_request_on_cpu_only_hardware_fails_explicitly() -> None:
    with pytest.raises(PrecisionPolicyError, match="CPU-only"):
        precision_capabilities_from_hardware(_hardware(), PrecisionDevice.CUDA)


def test_unsupported_accumulation_dtype_fails_or_falls_back_explicitly() -> None:
    capabilities = precision_capabilities_from_hardware(
        _hardware("8.6"),
        PrecisionDevice.CUDA,
    )
    request = PrecisionRequest(
        PrecisionRequestKind.FLOAT16,
        accumulation_dtype=PrecisionDType.FLOAT16,
    )
    with pytest.raises(PrecisionPolicyError, match="accumulation dtype float16"):
        select_precision_policy(request, capabilities)

    decision = select_precision_policy(
        PrecisionRequest(
            PrecisionRequestKind.FLOAT16,
            accumulation_dtype=PrecisionDType.FLOAT16,
            allow_fallback=True,
        ),
        capabilities,
    )
    assert decision.compute_dtype is PrecisionDType.FLOAT16
    assert decision.accumulation_dtype is PrecisionDType.FLOAT32
    assert decision.fallback_reason is not None
    assert "float32 accumulation" in decision.fallback_reason


def test_metric_precision_binding_rejects_compute_accumulation_or_autocast_drift() -> None:
    capabilities = precision_capabilities_from_hardware(
        _hardware("8.6"),
        PrecisionDevice.CUDA,
    )
    decision = select_precision_policy(
        PrecisionRequest(PrecisionRequestKind.AUTOCAST),
        capabilities,
    )
    expected = expected_execution_context(decision)
    record = bind_metric_precision("loss", decision, expected)
    assert record.metric_name == "loss"
    assert record.decision_id == decision.decision_id
    assert record.execution == expected

    drifted = (
        PrecisionExecutionContext(
            PrecisionDType.FLOAT16,
            expected.accumulation_dtype,
            expected.autocast_enabled,
        ),
        PrecisionExecutionContext(
            expected.compute_dtype,
            PrecisionDType.FLOAT16,
            expected.autocast_enabled,
        ),
        PrecisionExecutionContext(
            expected.compute_dtype,
            expected.accumulation_dtype,
            False,
        ),
    )
    for observed in drifted:
        with pytest.raises(PrecisionPolicyError, match="precision changed"):
            bind_metric_precision("loss", decision, observed)


def test_precision_decision_identity_is_deterministic_and_policy_sensitive() -> None:
    capabilities = precision_capabilities_from_hardware(
        _hardware("8.6"),
        PrecisionDevice.CUDA,
    )
    first = select_precision_policy(
        PrecisionRequest(PrecisionRequestKind.AUTOCAST),
        capabilities,
    )
    second = select_precision_policy(
        PrecisionRequest(PrecisionRequestKind.AUTOCAST),
        capabilities,
    )
    direct = select_precision_policy(
        PrecisionRequest(PrecisionRequestKind.FLOAT16),
        capabilities,
    )

    assert first.decision_id == second.decision_id
    assert first.decision_id.startswith("precision_")
    assert first.decision_id != direct.decision_id

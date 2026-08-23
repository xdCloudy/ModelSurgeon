"""Tests for hard-ceiling full, tensor, and streaming memory planning."""

from __future__ import annotations

import pytest

from modelsurgeon.config import MemoryMode
from modelsurgeon.experiments import (
    MemoryPlanningError,
    OperationMemoryEstimates,
    ResourceCapacity,
    ResourceCeilings,
    ResourceEstimate,
    plan_memory_mode,
)


def _estimates() -> OperationMemoryEstimates:
    return OperationMemoryEstimates(
        full=ResourceEstimate(800, 400, 100),
        tensor=ResourceEstimate(500, 100, 300),
        streaming=ResourceEstimate(200, 0, 700),
    )


def test_auto_chooses_first_mode_fitting_all_resources() -> None:
    plan = plan_memory_mode(
        MemoryMode.AUTO,
        _estimates(),
        ResourceCapacity(ram_bytes=600, vram_bytes=200, scratch_bytes=500),
    )

    assert plan.mode is MemoryMode.TENSOR
    assert plan.peak == ResourceEstimate(500, 100, 300)
    assert plan.effective_capacity == ResourceCapacity(600, 200, 500)
    assert tuple(item.mode for item in plan.rejected_modes) == (MemoryMode.FULL,)
    assert plan.rejected_modes[0].exceeded_resources == ("ram", "vram")


def test_auto_falls_back_to_streaming_when_tensor_exceeds_hard_ram_ceiling() -> None:
    plan = plan_memory_mode(
        MemoryMode.AUTO,
        _estimates(),
        ResourceCapacity(1000, 1000, 1000),
        ResourceCeilings(max_ram_bytes=250, max_vram_bytes=0, max_scratch_bytes=800),
    )

    assert plan.mode is MemoryMode.STREAMING
    assert plan.peak.peak_ram_bytes <= plan.effective_capacity.ram_bytes
    assert plan.peak.peak_vram_bytes <= plan.effective_capacity.vram_bytes
    assert plan.peak.scratch_bytes <= plan.effective_capacity.scratch_bytes


def test_explicit_mode_is_honored_without_silent_fallback() -> None:
    with pytest.raises(MemoryPlanningError, match="full exceeds ram, vram"):
        plan_memory_mode(
            MemoryMode.FULL,
            _estimates(),
            ResourceCapacity(600, 200, 1000),
        )


def test_no_automatic_mode_fits_reports_each_rejected_mode() -> None:
    with pytest.raises(MemoryPlanningError) as captured:
        plan_memory_mode(
            MemoryMode.AUTO,
            _estimates(),
            ResourceCapacity(100, 0, 50),
        )

    message = str(captured.value)
    assert "full exceeds ram, vram, scratch" in message
    assert "tensor exceeds ram, vram, scratch" in message
    assert "streaming exceeds ram, scratch" in message


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: ResourceEstimate(-1, 0, 0), "peak RAM"),
        (lambda: ResourceCapacity(1, -1, 1), "available VRAM"),
        (lambda: ResourceCeilings(max_scratch_bytes=-1), "maximum scratch"),
    ],
)
def test_negative_or_boolean_resource_values_fail(constructor: object, message: str) -> None:
    with pytest.raises(MemoryPlanningError, match=message):
        constructor()  # type: ignore[operator]

"""Tests for memory-bounded calibration batching and exact sample resume."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.datasets import CalibrationSample, TokenizedCalibrationSample
from modelsurgeon.experiments import (
    CPUInventory,
    CUDAInventory,
    CalibrationBatchCursor,
    CalibrationBatchMemoryModel,
    CalibrationBatchObservation,
    CalibrationBatchPlanner,
    CalibrationBatchPlannerConfig,
    CalibrationBatchPlanningError,
    DiskInventory,
    HardwareInventory,
    MemoryInventory,
    SoftwareInventory,
    calibration_manifest_digest,
)
from modelsurgeon.experiments.resource_budget import (
    ResourceBudgetExceeded,
    ResourceKind,
    StageResourceBudget,
)


def _hardware(*, available_ram: int = 10_000) -> HardwareInventory:
    return HardwareInventory(
        "Linux",
        "test",
        "test-version",
        CPUInventory("x86_64", "test-cpu", 8),
        MemoryInventory(20_000, available_ram),
        DiskInventory("/tmp", 100_000, 50_000),
        CUDAInventory(False, None, (), ()),
        SoftwareInventory("3.12", "CPython", "0.0.1", None),
    )


def _samples(lengths: tuple[int, ...]) -> tuple[TokenizedCalibrationSample, ...]:
    return tuple(
        TokenizedCalibrationSample(
            CalibrationSample(f"sample-{index}", f"{index + 1:064x}"),
            tuple(range(length)),
        )
        for index, length in enumerate(lengths)
    )


def _planner(
    *,
    max_batch_size: int = 4,
    max_batch_tokens: int = 100,
    max_ram_bytes: int | None = None,
    memory_model: CalibrationBatchMemoryModel | None = None,
    min_batch_size: int = 1,
) -> CalibrationBatchPlanner:
    return CalibrationBatchPlanner(
        CalibrationBatchPlannerConfig(
            max_batch_size,
            max_batch_tokens,
            StageResourceBudget(max_ram_bytes=max_ram_bytes),
            memory_model or CalibrationBatchMemoryModel(),
            min_batch_size,
        ),
        _hardware(),
    )


def test_initial_batch_respects_token_limit_without_reordering_or_splitting() -> None:
    samples = _samples((2, 2, 3, 1))
    plan = _planner(max_batch_tokens=6).plan_next(samples)

    assert plan.batch is not None
    assert plan.batch.start_index == 0
    assert plan.batch.end_index == 2
    assert plan.batch.sample_ids == ("sample-0", "sample-1")
    assert plan.batch.token_count == 4
    assert plan.next_cursor.next_sample_index == 2


def test_memory_model_stops_before_configured_ram_ceiling() -> None:
    samples = _samples((2, 2, 2, 2))
    planner = _planner(
        max_ram_bytes=100,
        memory_model=CalibrationBatchMemoryModel(
            fixed_ram_bytes=10,
            ram_bytes_per_token=20,
        ),
    )

    plan = planner.plan_next(samples)

    assert plan.batch is not None
    assert plan.batch.sample_ids == ("sample-0", "sample-1")
    assert plan.batch.estimated_ram_bytes == 90
    assert plan.batch.estimated_ram_bytes <= 100


def test_measured_memory_raises_model_and_shrinks_next_batch() -> None:
    samples = _samples((2,) * 8)
    planner = _planner(
        max_ram_bytes=100,
        memory_model=CalibrationBatchMemoryModel(ram_bytes_per_token=10),
    )
    first = planner.plan_next(samples)
    assert first.batch is not None
    assert len(first.batch.sample_ids) == 4

    observation = CalibrationBatchObservation(
        first.manifest_digest,
        first.batch.start_index,
        first.batch.end_index,
        first.batch.token_count,
        ram_baseline_bytes=100,
        ram_peak_bytes=260,
    )
    second = planner.plan_next(
        samples,
        cursor=first.next_cursor,
        observations=(observation,),
    )

    assert second.memory_model.ram_bytes_per_token == 20
    assert second.batch is not None
    assert second.batch.sample_ids == ("sample-4", "sample-5")
    assert second.batch.estimated_ram_bytes == 80


def test_memory_exhaustion_halves_effective_batch_cap() -> None:
    samples = _samples((1,) * 16)
    planner = _planner(max_batch_size=8)
    first = planner.plan_next(samples)
    assert first.batch is not None
    assert len(first.batch.sample_ids) == 8
    observation = CalibrationBatchObservation(
        first.manifest_digest,
        first.batch.start_index,
        first.batch.end_index,
        first.batch.token_count,
        exhausted_resource=ResourceKind.RAM,
    )

    resumed = planner.plan_next(
        samples,
        cursor=first.next_cursor,
        observations=(observation,),
    )

    assert resumed.effective_max_batch_size == 4
    assert resumed.batch is not None
    assert len(resumed.batch.sample_ids) == 4


def test_resume_cursor_preserves_exact_sample_sequence_and_completion() -> None:
    samples = _samples((1, 2, 1, 2, 1))
    planner = _planner(max_batch_size=2)
    plans = []
    cursor = None
    while True:
        plan = planner.plan_next(samples, cursor=cursor)
        plans.append(plan)
        if plan.complete:
            break
        cursor = plan.next_cursor

    flattened = tuple(
        sample_id
        for plan in plans
        if plan.batch is not None
        for sample_id in plan.batch.sample_ids
    )
    assert flattened == tuple(sample.identity.sample_id for sample in samples)
    assert plans[-1].cursor.next_sample_index == len(samples)
    assert plans[-1].next_cursor == plans[-1].cursor

    wrong = CalibrationBatchCursor("f" * 64, 2)
    with pytest.raises(CalibrationBatchPlanningError, match="different calibration manifest"):
        planner.plan_next(samples, cursor=wrong)


def test_oversized_single_sample_is_rejected_instead_of_token_splitting() -> None:
    samples = _samples((7, 1))
    with pytest.raises(CalibrationBatchPlanningError, match="without token-splitting"):
        _planner(max_batch_tokens=6).plan_next(samples)


def test_single_sample_over_memory_limit_fails_closed() -> None:
    samples = _samples((2,))
    planner = _planner(
        max_ram_bytes=50,
        memory_model=CalibrationBatchMemoryModel(ram_bytes_per_token=30),
    )
    with pytest.raises(ResourceBudgetExceeded):
        planner.plan_next(samples)


def test_manifest_digest_binds_order_identity_content_and_tokens() -> None:
    samples = _samples((1, 2))
    first = calibration_manifest_digest(samples)
    assert first == calibration_manifest_digest(samples)
    assert first != calibration_manifest_digest(tuple(reversed(samples)))


def test_stale_telemetry_manifest_is_rejected() -> None:
    samples = _samples((1, 1, 1))
    digest = calibration_manifest_digest(samples)
    observation = CalibrationBatchObservation("f" * 64, 0, 1, 1)
    with pytest.raises(CalibrationBatchPlanningError, match="different calibration manifest"):
        _planner().plan_next(
            samples,
            cursor=CalibrationBatchCursor(digest, 1),
            observations=(observation,),
        )

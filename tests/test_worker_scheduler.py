"""Typed worker registration, placement, recovery, and publication tests."""

from __future__ import annotations

import pytest

from modelsurgeon.experiments import (
    CPUInventory,
    CUDAInventory,
    DiskInventory,
    HardwareInventory,
    MemoryInventory,
    SoftwareInventory,
)
from modelsurgeon.experiments.worker_scheduler import (
    CampaignBudget,
    ContentAddressedInput,
    WorkerCapabilityProfile,
    WorkerOperation,
    WorkerResourceCapacity,
    WorkerScheduler,
    WorkerSchedulerError,
    WorkerTaskOutcome,
    WorkerTaskResult,
    WorkerTaskSpec,
    WorkerTaskState,
    WorkerTransport,
)


def _hardware(*, gpu: bool = False) -> HardwareInventory:
    return HardwareInventory(
        "Linux",
        "test",
        "test-version",
        CPUInventory("x86_64", "test-cpu", 8),
        MemoryInventory(8_000, 4_000),
        DiskInventory("/tmp", 20_000, 10_000),
        CUDAInventory(
            gpu,
            "12.4" if gpu else None,
            ("550.1",) if gpu else (),
            (),
        ),
        SoftwareInventory("3.12", "CPython", "0.0.1", None),
    )


def _profile(
    worker_id: str,
    *,
    gpu: bool = False,
    transport: WorkerTransport = WorkerTransport.LOCAL,
    operations: tuple[WorkerOperation, ...] = (
        WorkerOperation.EVALUATE,
        WorkerOperation.MUTATE,
        WorkerOperation.PACKAGE,
        WorkerOperation.PROFILE,
    ),
    max_concurrency: int = 1,
) -> WorkerCapabilityProfile:
    return WorkerCapabilityProfile(
        worker_id,
        transport,
        operations,
        WorkerResourceCapacity(8_000, 8_000 if gpu else 0, 20_000, 8, 2 if gpu else 0),
        _hardware(gpu=gpu),
        "worker-revision-1",
        max_concurrency,
    )


def _task(
    campaign_id: str = "campaign-1",
    *,
    operation: WorkerOperation = WorkerOperation.EVALUATE,
    vram_bytes: int = 0,
    priority: int = 0,
) -> WorkerTaskSpec:
    return WorkerTaskSpec(
        campaign_id,
        operation,
        (ContentAddressedInput("sha256:" + "a" * 64, 128, "model"),),
        1_000,
        vram_bytes,
        500,
        1,
        priority,
    )


def test_authenticated_registration_and_profile_identity_are_deterministic() -> None:
    scheduler = WorkerScheduler(lease_duration_ns=10)
    profile = _profile("remote-1", transport=WorkerTransport.EXPLICIT_REMOTE)
    registration = scheduler.register(profile, credential="secret", now_ns=1)
    assert registration.profile.profile_id == profile.profile_id
    assert str(registration.to_record()["credential_digest"]).startswith("sha256:")
    with pytest.raises(WorkerSchedulerError, match="credential mismatch"):
        scheduler.register(profile, credential="wrong", now_ns=2)


def test_tasks_are_typed_allowlisted_and_content_addressed() -> None:
    with pytest.raises(WorkerSchedulerError, match="allowlisted"):
        WorkerTaskSpec(
            "campaign-1",
            "shell",  # type: ignore[arg-type]
            (ContentAddressedInput("sha256:" + "a" * 64, 1, "input"),),
            1,
            0,
            1,
            1,
        )
    with pytest.raises(WorkerSchedulerError, match="content digest"):
        ContentAddressedInput("not-a-digest", 1, "input")


def test_scheduler_places_cpu_and_gpu_tasks_without_overcommitting_resources() -> None:
    scheduler = WorkerScheduler(lease_duration_ns=100, token_factory=lambda: "token")
    scheduler.set_campaign_budget("campaign-1", CampaignBudget(3, 3))
    scheduler.register(_profile("cpu-1"), credential="cpu-secret", now_ns=0)
    scheduler.register(
        _profile("gpu-1", gpu=True, max_concurrency=2), credential="gpu-secret", now_ns=0
    )
    cpu_task = _task(priority=2)
    gpu_task = _task(vram_bytes=5_000, priority=1)
    second_gpu_task = _task(vram_bytes=5_000, operation=WorkerOperation.MUTATE)
    scheduler.submit(cpu_task)
    scheduler.submit(gpu_task)
    scheduler.submit(second_gpu_task)

    assignments = scheduler.schedule(now_ns=1)
    assert {(item.task_id, item.worker_id) for item in assignments} == {
        (cpu_task.task_id, "cpu-1"),
        (gpu_task.task_id, "gpu-1"),
    }
    assert scheduler.snapshot(second_gpu_task.task_id).state is WorkerTaskState.QUEUED


def test_expired_lease_retries_and_publication_is_idempotent() -> None:
    tokens = iter(("first-token", "second-token"))
    scheduler = WorkerScheduler(lease_duration_ns=10, token_factory=lambda: next(tokens))
    scheduler.set_campaign_budget("campaign-1", CampaignBudget(1, 1))
    scheduler.register(_profile("worker-1"), credential="secret", now_ns=0)
    task = _task()
    scheduler.submit(task)
    first = scheduler.schedule(now_ns=1)[0]
    second = scheduler.schedule(now_ns=11)[0]
    assert second.lease.attempt == 2
    with pytest.raises(WorkerSchedulerError, match="not current"):
        scheduler.complete(
            WorkerTaskResult(
                task.task_id, WorkerTaskOutcome.SUPPORTED, "worker-1", "sha256:" + "b" * 64
            ),
            lease_token=first.lease.lease_token,
            now_ns=12,
        )
    result = WorkerTaskResult(
        task.task_id,
        WorkerTaskOutcome.SUPPORTED,
        "worker-1",
        "sha256:" + "b" * 64,
        ("evidence-1",),
    )
    assert scheduler.complete(result, lease_token=second.lease.lease_token, now_ns=12) == result
    assert scheduler.complete(result, lease_token="stale", now_ns=13) == result
    assert scheduler.snapshot(task.task_id).state is WorkerTaskState.COMPLETED


def test_campaign_budget_and_capability_mismatch_leave_work_queued() -> None:
    scheduler = WorkerScheduler(lease_duration_ns=10)
    scheduler.set_campaign_budget("campaign-1", CampaignBudget(1, 1))
    scheduler.register(
        _profile("cpu-1", operations=(WorkerOperation.PROFILE,)),
        credential="secret",
        now_ns=0,
    )
    task = _task(operation=WorkerOperation.EVALUATE)
    scheduler.submit(task)
    with pytest.raises(WorkerSchedulerError, match="budget exceeded"):
        scheduler.submit(_task(operation=WorkerOperation.PROFILE))
    assert scheduler.schedule(now_ns=1) == ()
    assert scheduler.snapshot(task.task_id).state is WorkerTaskState.QUEUED

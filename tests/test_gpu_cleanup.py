"""Tests for deterministic worker-boundary GPU resource cleanup."""

from __future__ import annotations

import weakref

import pytest

from modelsurgeon.experiments.gpu_cleanup import (
    ExperimentGPUCleanup,
    GPUCleanupError,
)


class FakeCuda:
    def __init__(self) -> None:
        self.allocated = 0
        self.reserved = 0
        self.synchronize_calls = 0
        self.empty_cache_calls = 0

    def allocate(self, size: int) -> None:
        self.allocated += size
        self.reserved = max(self.reserved, self.allocated)

    def release(self, size: int) -> None:
        self.allocated -= size

    def allocated_bytes(self) -> int:
        return self.allocated

    def reserved_bytes(self) -> int:
        return self.reserved

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1
        self.reserved = self.allocated

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class Closeable:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def close(self) -> None:
        self.events.append(self.name)


class Clearable:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def clear(self) -> None:
        self.events.append(self.name)


class WeakResource:
    pass


def test_cleanup_releases_worker_resources_in_reverse_order_and_reports_vram() -> None:
    cuda = FakeCuda()
    cuda.allocate(256)
    events: list[str] = []
    boundary = ExperimentGPUCleanup(cuda=cuda, rss_probe=lambda: 100, collect_garbage=lambda: 3)
    boundary.own_model(WeakResource(), cleanup=lambda: (events.append("model"), cuda.release(256)))
    boundary.own_hooks(Closeable("hooks", events))
    boundary.own_gradients(Clearable("gradients", events))
    boundary.own_cache(Clearable("cache", events))

    report = boundary.cleanup()

    assert events == ["cache", "gradients", "hooks", "model"]
    assert report.released_resources == ("cache", "gradients", "hooks", "model")
    assert report.before.cuda_allocated_bytes == 256
    assert report.after.cuda_allocated_bytes == 0
    assert report.after.cuda_reserved_bytes == 0
    assert report.retained_cuda_reserved_bytes == 0
    assert report.gc_collected == 3
    assert report.cuda_cache_cleared is True
    assert cuda.empty_cache_calls == 1
    assert cuda.synchronize_calls == 2


def test_repeated_tiny_experiments_have_bounded_retained_vram() -> None:
    cuda = FakeCuda()
    retained: list[int | None] = []
    for _ in range(12):
        cuda.allocate(64)
        boundary = ExperimentGPUCleanup(cuda=cuda, rss_probe=lambda: 100)
        boundary.own_model(WeakResource(), cleanup=lambda: cuda.release(64))
        retained.append(boundary.cleanup().retained_cuda_reserved_bytes)

    assert retained == [0] * 12
    assert cuda.allocated_bytes() == 0
    assert cuda.reserved_bytes() == 0


def test_owned_reference_is_released_before_gc() -> None:
    boundary = ExperimentGPUCleanup(rss_probe=lambda: 100)
    reference = weakref.ref(boundary.own_model(WeakResource()))
    assert reference() is not None

    boundary.cleanup()

    assert reference() is None


def test_cleanup_runs_after_operation_exception_without_masking_original() -> None:
    events: list[str] = []
    boundary = ExperimentGPUCleanup(rss_probe=lambda: 100)

    with pytest.raises(RuntimeError, match="operation failed"), boundary:
        boundary.own_hooks(Closeable("hooks", events))
        boundary.own_gradients(Clearable("gradients", events))
        raise RuntimeError("operation failed")

    assert events == ["gradients", "hooks"]
    assert boundary.last_report is not None
    assert boundary.last_report.released_resources == ("gradients", "hooks")


def test_cleanup_failure_does_not_mask_active_exception_but_is_reported() -> None:
    def fail_cleanup() -> None:
        raise ValueError("cleanup boom")

    boundary = ExperimentGPUCleanup(rss_probe=lambda: 100)
    with pytest.raises(RuntimeError, match="operation failed") as captured, boundary:
        boundary.own_model(WeakResource(), cleanup=fail_cleanup)
        raise RuntimeError("operation failed")

    assert boundary.last_report is not None
    assert boundary.last_report.failures[0].resource == "model"
    assert any("ModelSurgeon cleanup failures" in note for note in captured.value.__notes__)


def test_cleanup_failure_without_active_exception_raises_after_full_cleanup() -> None:
    events: list[str] = []

    def fail_cleanup() -> None:
        events.append("failing")
        raise ValueError("cleanup boom")

    boundary = ExperimentGPUCleanup(rss_probe=lambda: 100)
    with pytest.raises(GPUCleanupError, match="failed cleanup"), boundary:
        boundary.own_model(WeakResource(), cleanup=fail_cleanup)
        boundary.own_cache(Clearable("cache", events))

    assert events == ["cache", "failing"]
    assert boundary.last_report is not None
    assert boundary.last_report.failures[0].exception_type == "ValueError"


def test_cpu_only_cleanup_reports_no_fabricated_cuda_values_and_is_idempotent() -> None:
    boundary = ExperimentGPUCleanup(rss_probe=lambda: 123, collect_garbage=lambda: 0)
    boundary.own_cache([])
    first = boundary.cleanup()
    second = boundary.cleanup()

    assert first is second
    assert first.before.cuda_allocated_bytes is None
    assert first.after.cuda_reserved_bytes is None
    assert first.retained_rss_bytes == 123
    assert first.cuda_cache_cleared is False


def test_duplicate_resource_names_and_post_cleanup_registration_fail_closed() -> None:
    boundary = ExperimentGPUCleanup(rss_probe=lambda: 0)
    boundary.own("custom", object())
    with pytest.raises(GPUCleanupError, match="duplicate"):
        boundary.own("custom", object())
    boundary.cleanup()
    with pytest.raises(GPUCleanupError, match="after cleanup"):
        boundary.own("new", object())

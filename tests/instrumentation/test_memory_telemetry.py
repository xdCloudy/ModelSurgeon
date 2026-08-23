import threading
import time
from unittest.mock import patch

import pytest

from modelsurgeon.instrumentation.memory_telemetry import (
    MemoryTelemetryConfig,
    collect_memory_telemetry,
)


class FakeCudaMemory:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.allocated = 10
        self.reserved = 20

    def reset_peak_stats(self) -> None:
        self.reset_calls += 1

    def allocated_bytes(self) -> int:
        return self.allocated

    def reserved_bytes(self) -> int:
        return self.reserved

    def max_allocated_bytes(self) -> int:
        return 30

    def max_reserved_bytes(self) -> int:
        return 40


def test_cpu_only_collection_works_without_sampler_thread() -> None:
    before = {thread.ident for thread in threading.enumerate()}
    with patch(
        "modelsurgeon.instrumentation.memory_telemetry.process_rss_bytes",
        side_effect=[100, 150],
    ):
        report = collect_memory_telemetry(
            "cpu-op",
            lambda: None,
            MemoryTelemetryConfig(sampling_enabled=False),
        )
    after = {thread.ident for thread in threading.enumerate()}

    assert after == before
    assert report.sampling_enabled is False
    assert report.sample_interval_seconds is None
    assert len(report.samples) == 2
    assert report.peak_rss_bytes == 150
    assert report.cuda_available is False
    assert report.peak_cuda_allocated_bytes is None
    assert report.peak_cuda_reserved_bytes is None


def test_sampling_thread_is_joined_after_success() -> None:
    prefix = "modelsurgeon-memory-sampled"
    with patch(
        "modelsurgeon.instrumentation.memory_telemetry.process_rss_bytes",
        return_value=200,
    ):
        report = collect_memory_telemetry(
            "sampled",
            lambda: time.sleep(0.02),
            MemoryTelemetryConfig(
                sampling_enabled=True,
                sample_interval_seconds=0.002,
                max_samples=32,
            ),
        )

    assert len(report.samples) >= 3
    assert not any(thread.name == prefix for thread in threading.enumerate())


def test_sampling_thread_is_joined_when_operation_raises() -> None:
    def fail() -> None:
        time.sleep(0.01)
        raise RuntimeError("boom")

    with (
        patch(
            "modelsurgeon.instrumentation.memory_telemetry.process_rss_bytes",
            return_value=300,
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        collect_memory_telemetry(
            "failing",
            fail,
            MemoryTelemetryConfig(
                sampling_enabled=True,
                sample_interval_seconds=0.001,
                max_samples=32,
            ),
        )

    assert not any(
        thread.name == "modelsurgeon-memory-failing" for thread in threading.enumerate()
    )


def test_cuda_peaks_and_samples_are_reported_from_provider() -> None:
    cuda = FakeCudaMemory()
    with patch(
        "modelsurgeon.instrumentation.memory_telemetry.process_rss_bytes",
        side_effect=[100, 120],
    ):
        report = collect_memory_telemetry(
            "cuda-op",
            lambda: None,
            MemoryTelemetryConfig(sampling_enabled=False),
            cuda=cuda,
        )

    assert cuda.reset_calls == 1
    assert report.cuda_available is True
    assert report.peak_cuda_allocated_bytes == 30
    assert report.peak_cuda_reserved_bytes == 40
    assert [sample.cuda_allocated_bytes for sample in report.samples] == [10, 10]
    assert [sample.cuda_reserved_bytes for sample in report.samples] == [20, 20]

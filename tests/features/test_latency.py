import pytest

from modelsurgeon.features.latency import (
    LatencyBackend,
    LatencyProfileConfig,
    LatencyProfileError,
    profile_component_latency,
)


class SteppingClock:
    def __init__(self, step: int = 10) -> None:
        self.value = 0
        self.step = step

    def __call__(self) -> int:
        current = self.value
        self.value += self.step
        return current


class FakeCudaTimer:
    def __init__(self, durations: list[int]) -> None:
        self.durations = iter(durations)
        self.calls = 0

    def measure_ns(self, operation: object) -> int:
        self.calls += 1
        if not callable(operation):
            raise AssertionError("operation must be callable")
        operation()
        return next(self.durations)


def test_cpu_profile_reports_warmup_samples_overhead_and_no_cuda_timing() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1

    profile = profile_component_latency(
        operation,
        LatencyProfileConfig(warmup_runs=2, sample_runs=3),
        clock_ns=SteppingClock(10),
    )

    assert calls == 5
    assert profile.sample_count == 3
    assert profile.warmup_count == 2
    assert profile.median_ns == 10.0
    assert profile.median_absolute_deviation_ns == 0.0
    assert profile.profiler_overhead_ns == 10.0
    assert profile.cpu_median_ns == 10.0
    assert profile.cuda_median_ns is None
    assert profile.environment.backend is LatencyBackend.CPU


def test_explicit_cpu_synchronizer_wraps_each_measurement() -> None:
    synchronizations = 0

    def synchronize() -> None:
        nonlocal synchronizations
        synchronizations += 1

    profile_component_latency(
        lambda: None,
        LatencyProfileConfig(warmup_runs=1, sample_runs=3),
        synchronize=synchronize,
        clock_ns=SteppingClock(5),
    )

    assert synchronizations == 13


def test_cuda_profile_uses_event_timer_and_does_not_fabricate_cpu_timing() -> None:
    timer = FakeCudaTimer([100, 120, 110, 4, 5, 3])
    profile = profile_component_latency(
        lambda: None,
        LatencyProfileConfig(warmup_runs=0, sample_runs=3),
        backend=LatencyBackend.CUDA,
        cuda_timer=timer,
        device="cuda:0",
    )

    assert timer.calls == 6
    assert profile.median_ns == 110.0
    assert profile.median_absolute_deviation_ns == 10.0
    assert profile.profiler_overhead_ns == 4.0
    assert profile.cpu_median_ns is None
    assert profile.cuda_median_ns == 110.0
    assert profile.environment.timer == "cuda_event"
    assert profile.environment.device == "cuda:0"


def test_cuda_requires_event_timer_and_cpu_rejects_one() -> None:
    timer = FakeCudaTimer([1] * 6)
    config = LatencyProfileConfig(sample_runs=3)

    with pytest.raises(LatencyProfileError, match="requires a CUDA event timer"):
        profile_component_latency(lambda: None, config, backend=LatencyBackend.CUDA)

    with pytest.raises(LatencyProfileError, match="cannot accept a CUDA timer"):
        profile_component_latency(lambda: None, config, cuda_timer=timer)

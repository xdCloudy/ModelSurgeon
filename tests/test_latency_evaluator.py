"""Tests for robust end-to-end prefill/decode latency evaluation."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from modelsurgeon.evaluation.latency import (
    LatencyEvaluationContext,
    LatencyEvaluationError,
    compare_end_to_end_latency,
    evaluate_end_to_end_latency,
)
from modelsurgeon.features.latency import LatencyBackend, LatencyEnvironment, LatencyProfile


def _profile(
    median: float,
    mad: float = 10.0,
    *,
    device: str = "cpu",
) -> LatencyProfile:
    return LatencyProfile(
        version="1",
        sample_count=20,
        warmup_count=3,
        median_ns=median,
        median_absolute_deviation_ns=mad,
        minimum_ns=int(median - mad),
        maximum_ns=int(median + mad),
        profiler_overhead_ns=1.0,
        cpu_median_ns=median,
        cuda_median_ns=None,
        environment=LatencyEnvironment(
            LatencyBackend.CPU, "perf_counter_ns", device, "explicit_callback"
        ),
    )


class FakeProfiler:
    def __init__(self, profiles: list[LatencyProfile]) -> None:
        self.profiles = profiles
        self.operations = 0

    def profile(self, operation: Callable[[], object]) -> LatencyProfile:
        operation()
        profile = self.profiles[self.operations]
        self.operations += 1
        return profile


def _context(**overrides: object) -> LatencyEvaluationContext:
    values: dict[str, object] = {
        "batch_size": 1,
        "prompt_tokens": 128,
        "decode_tokens": 16,
        "backend": LatencyBackend.CPU,
        "device": "cpu",
        "dtype": "float32",
        "power_mode": "balanced",
    }
    values.update(overrides)
    return LatencyEvaluationContext(**values)  # type: ignore[arg-type]


def test_prefill_decode_profiles_record_robust_confidence_and_context() -> None:
    calls: list[str] = []
    result = evaluate_end_to_end_latency(
        lambda: calls.append("prefill"),
        lambda: calls.append("decode"),
        _context(),
        FakeProfiler([_profile(1000.0, 100.0), _profile(200.0, 20.0)]),
    )
    assert calls == ["prefill", "decode"]
    assert result.prefill.median_ns == 1000.0
    assert result.prefill.robust_low_ns == pytest.approx(851.74)
    assert result.prefill.robust_high_ns == pytest.approx(1148.26)
    assert result.decode.median_ns == 200.0
    assert result.context.prompt_tokens == 128
    assert result.to_record()["context"] == _context().to_record()


def test_comparable_runs_report_prefill_and_decode_speedup_ratios() -> None:
    baseline = evaluate_end_to_end_latency(
        lambda: None,
        lambda: None,
        _context(),
        FakeProfiler([_profile(1000.0), _profile(400.0)]),
    )
    candidate = evaluate_end_to_end_latency(
        lambda: None,
        lambda: None,
        _context(),
        FakeProfiler([_profile(800.0), _profile(200.0)]),
    )
    comparison = compare_end_to_end_latency(baseline, candidate)
    assert comparison.comparable
    assert comparison.prefill_speedup_ratio == pytest.approx(1.25)
    assert comparison.decode_speedup_ratio == pytest.approx(2.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 2},
        {"prompt_tokens": 64},
        {"decode_tokens": 8},
        {"device": "other"},
        {"dtype": "float16"},
        {"power_mode": "performance"},
    ],
)
def test_invalid_comparison_context_is_flagged(overrides: dict[str, object]) -> None:
    baseline = evaluate_end_to_end_latency(
        lambda: None,
        lambda: None,
        _context(),
        FakeProfiler([_profile(1000.0), _profile(400.0)]),
    )
    candidate_context = _context(**overrides)
    candidate = evaluate_end_to_end_latency(
        lambda: None,
        lambda: None,
        candidate_context,
        FakeProfiler(
            [
                _profile(800.0, device=candidate_context.device),
                _profile(200.0, device=candidate_context.device),
            ]
        ),
    )
    comparison = compare_end_to_end_latency(baseline, candidate)
    assert not comparison.comparable
    assert comparison.reason == "latency evaluation contexts do not match"
    assert comparison.prefill_speedup_ratio is None
    assert comparison.decode_speedup_ratio is None


def test_profile_device_or_backend_mismatch_fails() -> None:
    bad = LatencyProfile(
        version="1",
        sample_count=3,
        warmup_count=0,
        median_ns=10.0,
        median_absolute_deviation_ns=1.0,
        minimum_ns=9,
        maximum_ns=11,
        profiler_overhead_ns=0.0,
        cpu_median_ns=10.0,
        cuda_median_ns=None,
        environment=LatencyEnvironment(LatencyBackend.CPU, "clock", "wrong", "none"),
    )
    with pytest.raises(LatencyEvaluationError, match="device"):
        evaluate_end_to_end_latency(
            lambda: None, lambda: None, _context(), FakeProfiler([bad, bad])
        )


def test_latency_context_requires_positive_geometry() -> None:
    with pytest.raises(LatencyEvaluationError, match="positive"):
        _context(prompt_tokens=0)

"""End-to-end prefill/decode latency evaluation with comparable run context."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from modelsurgeon.features.latency import LatencyBackend, LatencyProfile

LATENCY_EVALUATOR_VERSION = "1"


class LatencyEvaluationError(ValueError):
    """Raised when an end-to-end latency run has an invalid measurement contract."""


class LatencyProfiler(Protocol):
    def profile(self, operation: Callable[[], object]) -> LatencyProfile: ...


@dataclass(frozen=True, slots=True)
class LatencyEvaluationContext:
    batch_size: int
    prompt_tokens: int
    decode_tokens: int
    backend: LatencyBackend
    device: str
    dtype: str
    power_mode: str

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.prompt_tokens <= 0 or self.decode_tokens <= 0:
            raise LatencyEvaluationError("latency batch and token geometry must be positive")
        if not self.device or not self.dtype or not self.power_mode:
            raise LatencyEvaluationError("latency device, dtype, and power context are required")

    def to_record(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "prompt_tokens": self.prompt_tokens,
            "decode_tokens": self.decode_tokens,
            "backend": self.backend.value,
            "device": self.device,
            "dtype": self.dtype,
            "power_mode": self.power_mode,
        }


@dataclass(frozen=True, slots=True)
class LatencyConfidenceSummary:
    median_ns: float
    median_absolute_deviation_ns: float
    robust_low_ns: float
    robust_high_ns: float
    minimum_ns: int
    maximum_ns: int
    sample_count: int

    @classmethod
    def from_profile(cls, profile: LatencyProfile) -> "LatencyConfidenceSummary":
        scaled_mad = 1.4826 * profile.median_absolute_deviation_ns
        return cls(
            profile.median_ns,
            profile.median_absolute_deviation_ns,
            max(0.0, profile.median_ns - scaled_mad),
            profile.median_ns + scaled_mad,
            profile.minimum_ns,
            profile.maximum_ns,
            profile.sample_count,
        )


@dataclass(frozen=True, slots=True)
class EndToEndLatencyResult:
    context: LatencyEvaluationContext
    prefill: LatencyConfidenceSummary
    decode: LatencyConfidenceSummary
    prefill_profile: LatencyProfile
    decode_profile: LatencyProfile
    version: str = LATENCY_EVALUATOR_VERSION

    def __post_init__(self) -> None:
        for profile in (self.prefill_profile, self.decode_profile):
            if profile.environment.backend is not self.context.backend:
                raise LatencyEvaluationError("latency profile backend does not match run context")
            if profile.environment.device != self.context.device:
                raise LatencyEvaluationError("latency profile device does not match run context")

    def to_record(self) -> dict[str, object]:
        def summary(value: LatencyConfidenceSummary) -> dict[str, object]:
            return {
                "median_ns": value.median_ns,
                "median_absolute_deviation_ns": value.median_absolute_deviation_ns,
                "robust_low_ns": value.robust_low_ns,
                "robust_high_ns": value.robust_high_ns,
                "minimum_ns": value.minimum_ns,
                "maximum_ns": value.maximum_ns,
                "sample_count": value.sample_count,
            }

        return {
            "version": self.version,
            "context": self.context.to_record(),
            "prefill": summary(self.prefill),
            "decode": summary(self.decode),
        }


@dataclass(frozen=True, slots=True)
class LatencyComparison:
    comparable: bool
    reason: str | None
    prefill_speedup_ratio: float | None
    decode_speedup_ratio: float | None

    def __post_init__(self) -> None:
        ratios = (self.prefill_speedup_ratio, self.decode_speedup_ratio)
        if self.comparable:
            if self.reason is not None or any(value is None for value in ratios):
                raise LatencyEvaluationError("comparable latency results require both speedup ratios")
            if any(value is not None and (not math.isfinite(value) or value <= 0) for value in ratios):
                raise LatencyEvaluationError("latency speedup ratios must be finite and positive")
        elif not self.reason or any(value is not None for value in ratios):
            raise LatencyEvaluationError("invalid latency comparisons need a reason and no ratios")


def evaluate_end_to_end_latency(
    prefill_operation: Callable[[], object],
    decode_operation: Callable[[], object],
    context: LatencyEvaluationContext,
    profiler: LatencyProfiler,
) -> EndToEndLatencyResult:
    """Measure prefill and decode through the warmed/synchronized component profiler contract."""

    prefill_profile = profiler.profile(prefill_operation)
    decode_profile = profiler.profile(decode_operation)
    return EndToEndLatencyResult(
        context,
        LatencyConfidenceSummary.from_profile(prefill_profile),
        LatencyConfidenceSummary.from_profile(decode_profile),
        prefill_profile,
        decode_profile,
    )


def compare_end_to_end_latency(
    baseline: EndToEndLatencyResult,
    candidate: EndToEndLatencyResult,
) -> LatencyComparison:
    """Compare medians only when batch/token/device/dtype/power context is identical."""

    if baseline.context != candidate.context:
        return LatencyComparison(False, "latency evaluation contexts do not match", None, None)
    if candidate.prefill.median_ns <= 0 or candidate.decode.median_ns <= 0:
        return LatencyComparison(False, "candidate latency median is zero", None, None)
    return LatencyComparison(
        True,
        None,
        baseline.prefill.median_ns / candidate.prefill.median_ns,
        baseline.decode.median_ns / candidate.decode.median_ns,
    )

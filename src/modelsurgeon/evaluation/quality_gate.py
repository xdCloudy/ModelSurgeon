"""Shared lower-is-better quality gates for first-party optimize runtimes."""

from __future__ import annotations

import math


class QualityGateError(ValueError):
    """Raised when a quality gate cannot be evaluated safely."""


def evaluate_perplexity_quality_gate(
    baseline_perplexity: float,
    candidate_perplexity: float,
    *,
    min_quality_retention_ratio: float,
    max_perplexity_delta: float | None,
    profile_max_perplexity_delta: float | None,
) -> dict[str, object]:
    """Evaluate all declared perplexity bounds against one source baseline.

    Perplexity is a lower-is-better metric, so quality retention is represented
    as ``baseline / candidate``. The user constraint and selected profile are
    retained independently so evidence distinguishes the hard contract from a
    conservative profile guard.
    """

    values = (
        ("baseline_perplexity", baseline_perplexity),
        ("candidate_perplexity", candidate_perplexity),
        ("min_quality_retention_ratio", min_quality_retention_ratio),
    )
    for label, value in values:
        if not math.isfinite(value):
            raise QualityGateError(f"{label} must be finite")
    if baseline_perplexity <= 0 or candidate_perplexity <= 0:
        raise QualityGateError(
            "perplexity must be positive for quality-retention evaluation"
        )
    if not 0.0 <= min_quality_retention_ratio <= 1.0:
        raise QualityGateError(
            "min_quality_retention_ratio must be between zero and one"
        )
    optional_values: tuple[tuple[str, float | None], ...] = (
        ("max_perplexity_delta", max_perplexity_delta),
        ("profile_max_perplexity_delta", profile_max_perplexity_delta),
    )
    for optional_label, optional_value in optional_values:
        if optional_value is not None and (
            not math.isfinite(optional_value) or optional_value < 0
        ):
            raise QualityGateError(
                f"{optional_label} must be finite and non-negative when present"
            )

    delta = candidate_perplexity - baseline_perplexity
    retention_ratio = baseline_perplexity / candidate_perplexity
    retention_passed = retention_ratio >= min_quality_retention_ratio
    explicit_delta_passed = (
        max_perplexity_delta is None or delta <= max_perplexity_delta
    )
    profile_delta_passed = (
        profile_max_perplexity_delta is None or delta <= profile_max_perplexity_delta
    )
    failures: list[str] = []
    if not retention_passed:
        failures.append("minimum quality-retention ratio violated")
    if not explicit_delta_passed:
        failures.append("maximum perplexity delta violated")
    if not profile_delta_passed:
        failures.append("quality-profile perplexity guard violated")
    return {
        "metric": "perplexity",
        "measurement_authority": "physical_evaluation",
        "baseline_perplexity": baseline_perplexity,
        "candidate_perplexity": candidate_perplexity,
        "perplexity_delta": delta,
        "quality_retention_ratio": retention_ratio,
        "min_quality_retention_ratio": min_quality_retention_ratio,
        "max_perplexity_delta": max_perplexity_delta,
        "profile_max_perplexity_delta": profile_max_perplexity_delta,
        "retention_constraint_passed": retention_passed,
        "max_delta_constraint_passed": explicit_delta_passed,
        "quality_profile_guard_passed": profile_delta_passed,
        "accepted": retention_passed and explicit_delta_passed and profile_delta_passed,
        "rejection_reasons": failures,
    }


__all__ = ["QualityGateError", "evaluate_perplexity_quality_gate"]

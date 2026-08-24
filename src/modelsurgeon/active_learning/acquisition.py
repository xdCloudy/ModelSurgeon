"""Deterministic explore/exploit acquisition across value, uncertainty, and diversity."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

ACQUISITION_POLICY_SCHEMA_VERSION: Final[int] = 1


class AcquisitionPolicyError(ValueError):
    """Raised when acquisition inputs or fractions are invalid."""


class AcquisitionReason(StrEnum):
    HIGH_VALUE = "high-value"
    UNCERTAIN = "uncertain"
    DIVERSE = "diverse"
    BUDGET_FILL = "budget-fill"


@dataclass(frozen=True, slots=True)
class AcquisitionCandidate:
    candidate_id: str
    utility: float
    safe_probability: float
    uncertainty: float
    diversity: float

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("cand_"):
            raise AcquisitionPolicyError("acquisition candidates require canonical IDs")
        if any(
            not math.isfinite(value)
            for value in (self.utility, self.safe_probability, self.uncertainty, self.diversity)
        ):
            raise AcquisitionPolicyError("acquisition values must be finite")
        if not 0.0 <= self.safe_probability <= 1.0:
            raise AcquisitionPolicyError("safe probability must be within [0, 1]")
        if self.uncertainty < 0.0 or self.diversity < 0.0:
            raise AcquisitionPolicyError("uncertainty and diversity cannot be negative")


@dataclass(frozen=True, slots=True)
class AcquisitionPolicyConfig:
    high_value_fraction: float = 0.5
    uncertain_fraction: float = 0.3
    diverse_fraction: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        fractions = (
            self.high_value_fraction,
            self.uncertain_fraction,
            self.diverse_fraction,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in fractions):
            raise AcquisitionPolicyError("acquisition fractions must be finite and non-negative")
        if not math.isclose(math.fsum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise AcquisitionPolicyError("acquisition fractions must sum to one")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 64:
            raise AcquisitionPolicyError("acquisition seed must be unsigned 64-bit")


DEFAULT_ACQUISITION_POLICY_CONFIG: Final[AcquisitionPolicyConfig] = AcquisitionPolicyConfig()


@dataclass(frozen=True, slots=True)
class AcquiredCandidate:
    candidate_id: str
    rank: int
    reason: AcquisitionReason
    propensity: float
    policy_score: float

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "reason": self.reason.value,
            "propensity": self.propensity,
            "policy_score": self.policy_score,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionReport:
    selections: tuple[AcquiredCandidate, ...]
    requested_budget: int
    effective_budget: int
    candidate_count: int
    quotas: tuple[tuple[AcquisitionReason, int], ...]
    config: AcquisitionPolicyConfig
    schema_version: int = ACQUISITION_POLICY_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "requested_budget": self.requested_budget,
            "effective_budget": self.effective_budget,
            "candidate_count": self.candidate_count,
            "oversubscribed": self.requested_budget > self.candidate_count,
            "quotas": {reason.value: count for reason, count in self.quotas},
            "fractions": {
                "high_value": self.config.high_value_fraction,
                "uncertain": self.config.uncertain_fraction,
                "diverse": self.config.diverse_fraction,
            },
            "seed": self.config.seed,
            "propensity_semantics": "deterministic-conditional-inclusion",
            "selections": [item.to_record() for item in self.selections],
        }


def acquire_candidates(
    candidates: tuple[AcquisitionCandidate, ...],
    budget: int,
    *,
    config: AcquisitionPolicyConfig = DEFAULT_ACQUISITION_POLICY_CONFIG,
) -> AcquisitionReport:
    """Allocate exact quotas, deduplicate phase overlap, and fill deterministically."""

    if budget < 0:
        raise AcquisitionPolicyError("acquisition budget cannot be negative")
    ids = tuple(item.candidate_id for item in candidates)
    if len(ids) != len(set(ids)):
        raise AcquisitionPolicyError("acquisition candidate IDs must be unique")
    effective = min(budget, len(candidates))
    quotas = _quotas(effective, config)
    if effective == 0:
        return AcquisitionReport((), budget, 0, len(candidates), quotas, config)
    by_reason = {
        AcquisitionReason.HIGH_VALUE: sorted(
            candidates,
            key=lambda item: (
                -(item.utility * item.safe_probability),
                _seed_rank(config.seed, item.candidate_id),
            ),
        ),
        AcquisitionReason.UNCERTAIN: sorted(
            candidates,
            key=lambda item: (-item.uncertainty, _seed_rank(config.seed, item.candidate_id)),
        ),
        AcquisitionReason.DIVERSE: sorted(
            candidates,
            key=lambda item: (-item.diversity, _seed_rank(config.seed, item.candidate_id)),
        ),
    }
    selected: set[str] = set()
    output: list[AcquiredCandidate] = []
    quota_map = dict(quotas)
    for reason in (
        AcquisitionReason.HIGH_VALUE,
        AcquisitionReason.UNCERTAIN,
        AcquisitionReason.DIVERSE,
    ):
        taken = 0
        for candidate in by_reason[reason]:
            if candidate.candidate_id in selected:
                continue
            output.append(
                AcquiredCandidate(
                    candidate.candidate_id,
                    len(output) + 1,
                    reason,
                    1.0,
                    _score(candidate, reason),
                )
            )
            selected.add(candidate.candidate_id)
            taken += 1
            if taken == quota_map[reason]:
                break
    if len(output) < effective:
        remaining = sorted(
            (item for item in candidates if item.candidate_id not in selected),
            key=lambda item: (
                -_fill_score(item),
                _seed_rank(config.seed, item.candidate_id),
            ),
        )
        for candidate in remaining[: effective - len(output)]:
            output.append(
                AcquiredCandidate(
                    candidate.candidate_id,
                    len(output) + 1,
                    AcquisitionReason.BUDGET_FILL,
                    1.0,
                    _fill_score(candidate),
                )
            )
    return AcquisitionReport(tuple(output), budget, effective, len(candidates), quotas, config)


def _quotas(
    budget: int, config: AcquisitionPolicyConfig
) -> tuple[tuple[AcquisitionReason, int], ...]:
    fractions = (
        (AcquisitionReason.HIGH_VALUE, config.high_value_fraction),
        (AcquisitionReason.UNCERTAIN, config.uncertain_fraction),
        (AcquisitionReason.DIVERSE, config.diverse_fraction),
    )
    floors = {reason: math.floor(budget * fraction) for reason, fraction in fractions}
    remaining = budget - sum(floors.values())
    remainders = sorted(
        fractions,
        key=lambda item: (-(budget * item[1] - floors[item[0]]), item[0].value),
    )
    for reason, _ in remainders[:remaining]:
        floors[reason] += 1
    return tuple((reason, floors[reason]) for reason, _ in fractions)


def _score(candidate: AcquisitionCandidate, reason: AcquisitionReason) -> float:
    if reason is AcquisitionReason.HIGH_VALUE:
        return candidate.utility * candidate.safe_probability
    if reason is AcquisitionReason.UNCERTAIN:
        return candidate.uncertainty
    return candidate.diversity


def _fill_score(candidate: AcquisitionCandidate) -> float:
    return (
        candidate.utility * candidate.safe_probability + candidate.uncertainty + candidate.diversity
    )


def _seed_rank(seed: int, candidate_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{candidate_id}".encode("ascii")).digest(), "big")

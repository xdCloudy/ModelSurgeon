"""Evidence contract for long interaction-aware pruning sequences."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

LONG_SEQUENCE_STUDY_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class LongSequenceStudyError(ValueError):
    """Raised when a long-sequence study would overclaim incomplete evidence."""


class LongSequencePolicy(StrEnum):
    STATELESS = "stateless"
    ADDITIVE = "additive"
    STATE_AWARE = "state_aware"


class HorizonOutcome(StrEnum):
    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class AdversarialKind(StrEnum):
    SAME_LAYER = "same_layer"
    CROSS_LAYER = "cross_layer"
    DISTRIBUTION_SHIFT = "distribution_shift"


class AdversarialOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class StudyClaim(StrEnum):
    POSITIVE = "positive"
    NEGATIVE_RESULT = "negative_result"
    INCONCLUSIVE = "inconclusive"
    UNSUPPORTED = "unsupported"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LongSequenceStudyError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise LongSequenceStudyError(f"{label} must be lowercase SHA-256")
    return result


def _finite(value: float, label: str) -> None:
    if not math.isfinite(value):
        raise LongSequenceStudyError(f"{label} must be finite")


@dataclass(frozen=True, slots=True)
class AdversarialInteraction:
    kind: AdversarialKind
    outcome: AdversarialOutcome
    interaction_residual: float
    violation_count: int
    reason: str

    def __post_init__(self) -> None:
        _finite(self.interaction_residual, "interaction residual")
        if self.violation_count < 0:
            raise LongSequenceStudyError("adversarial violation count cannot be negative")
        _text(self.reason, "adversarial reason")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "outcome": self.outcome.value,
            "interaction_residual": self.interaction_residual,
            "violation_count": self.violation_count,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class LongSequenceKey:
    family: str
    policy: LongSequencePolicy
    horizon: int
    seed: int

    def __post_init__(self) -> None:
        _text(self.family, "model family")
        if self.horizon not in {10, 20, 50}:
            raise LongSequenceStudyError("supported horizons are exactly 10, 20, and 50")
        if self.seed < 0:
            raise LongSequenceStudyError("sequence seed cannot be negative")

    def to_record(self) -> dict[str, object]:
        return {
            "family": self.family,
            "policy": self.policy.value,
            "horizon": self.horizon,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class LongSequenceEvidence:
    key: LongSequenceKey
    outcome: HorizonOutcome
    state_id: str | None
    artifact_digest: str | None
    quality: float | None
    cumulative_regret: float | None
    violation_count: int | None
    evaluations: int | None
    rollback_count: int | None
    optimization_cost_seconds: float | None
    parent_state_id: str | None
    adversarial: tuple[AdversarialInteraction, ...]
    provenance: tuple[tuple[str, str], ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.outcome in {HorizonOutcome.MEASURED, HorizonOutcome.NEGATIVE_RESULT}:
            if (
                self.state_id is None
                or not self.state_id.startswith("state_")
                or self.artifact_digest is None
                or self.quality is None
                or self.cumulative_regret is None
                or self.violation_count is None
                or self.evaluations is None
                or self.rollback_count is None
                or self.optimization_cost_seconds is None
            ):
                raise LongSequenceStudyError(
                    "measured horizons require checkpoint and cost evidence"
                )
            _digest(self.artifact_digest, "checkpoint artifact digest")
            for value, label in (
                (self.quality, "quality"),
                (self.cumulative_regret, "cumulative regret"),
                (self.optimization_cost_seconds, "optimization cost"),
            ):
                assert value is not None
                _finite(value, label)
            if self.violation_count < 0 or self.evaluations <= 0 or self.rollback_count < 0:
                raise LongSequenceStudyError("horizon counts are invalid")
            if {item.kind for item in self.adversarial} != set(AdversarialKind):
                raise LongSequenceStudyError("measured horizons require all adversarial cases")
        else:
            if not self.reason or any(
                value is not None
                for value in (
                    self.state_id,
                    self.artifact_digest,
                    self.quality,
                    self.cumulative_regret,
                    self.violation_count,
                    self.evaluations,
                    self.rollback_count,
                    self.optimization_cost_seconds,
                )
            ):
                raise LongSequenceStudyError("non-measured horizons require only a retained reason")
        keys = tuple(key for key, _ in self.provenance)
        if not keys or keys != tuple(sorted(set(keys))):
            raise LongSequenceStudyError("horizon provenance must be non-empty and canonical")

    @property
    def evidence_id(self) -> str:
        return (
            "long_sequence_" + hashlib.sha256(_canonical(self.key.to_record()).encode()).hexdigest()
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": LONG_SEQUENCE_STUDY_SCHEMA_VERSION,
            "evidence_id": self.evidence_id,
            "key": self.key.to_record(),
            "outcome": self.outcome.value,
            "state_id": self.state_id,
            "artifact_digest": self.artifact_digest,
            "quality": self.quality,
            "cumulative_regret": self.cumulative_regret,
            "violation_count": self.violation_count,
            "evaluations": self.evaluations,
            "rollback_count": self.rollback_count,
            "optimization_cost_seconds": self.optimization_cost_seconds,
            "parent_state_id": self.parent_state_id,
            "adversarial": [item.to_record() for item in self.adversarial],
            "provenance": [{"key": key, "value": value} for key, value in self.provenance],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class LongSequenceSummary:
    policy: LongSequencePolicy
    horizon: int
    mean_regret: float | None
    regret_low: float | None
    regret_high: float | None
    mean_violations: float | None
    mean_evaluations: float | None
    claim: StudyClaim
    reason: str

    def to_record(self) -> dict[str, object]:
        return {
            "policy": self.policy.value,
            "horizon": self.horizon,
            "mean_regret": self.mean_regret,
            "regret_low": self.regret_low,
            "regret_high": self.regret_high,
            "mean_violations": self.mean_violations,
            "mean_evaluations": self.mean_evaluations,
            "claim": self.claim.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class LongSequenceStudy:
    families: tuple[str, ...]
    seeds: tuple[int, ...]
    horizons: tuple[int, ...]
    evidence: tuple[LongSequenceEvidence, ...]
    summaries: tuple[LongSequenceSummary, ...]
    bootstrap_repetitions: int
    confidence: float
    claim: StudyClaim
    reason: str

    def __post_init__(self) -> None:
        if len(self.families) < 2 or self.families != tuple(sorted(set(self.families))):
            raise LongSequenceStudyError("study requires two sorted unique families")
        if len(self.seeds) < 3 or self.seeds != tuple(sorted(set(self.seeds))):
            raise LongSequenceStudyError("study requires three sorted unique seeds")
        if self.horizons != (10, 20, 50):
            raise LongSequenceStudyError("study requires horizons 10, 20, and 50")
        if self.bootstrap_repetitions < 100 or not 0 < self.confidence < 1:
            raise LongSequenceStudyError("bootstrap configuration is invalid")
        expected = {
            (family, policy, horizon, seed)
            for family, policy, horizon, seed in product(
                self.families, LongSequencePolicy, self.horizons, self.seeds
            )
        }
        actual = {
            (item.key.family, item.key.policy, item.key.horizon, item.key.seed)
            for item in self.evidence
        }
        if actual != expected or len(actual) != len(self.evidence):
            raise LongSequenceStudyError("study must retain every family/policy/horizon/seed cell")
        _text(self.reason, "study claim reason")

    @property
    def study_id(self) -> str:
        return "long_sequence_study_" + hashlib.sha256(self.to_json().encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": LONG_SEQUENCE_STUDY_SCHEMA_VERSION,
            "study_id": self.study_id,
            "families": list(self.families),
            "seeds": list(self.seeds),
            "horizons": list(self.horizons),
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "confidence": self.confidence,
            "claim": self.claim.value,
            "reason": self.reason,
            "evidence": [item.to_record() for item in self.evidence],
            "summaries": [item.to_record() for item in self.summaries],
        }

    def to_json(self) -> str:
        record = self.to_record()
        record.pop("study_id", None)
        return _canonical(record)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _bootstrap(
    values: tuple[float, ...], seed: int, repetitions: int, confidence: float
) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(repetitions)
    )
    alpha = (1 - confidence) / 2
    return samples[max(0, math.floor(alpha * repetitions))], samples[
        min(repetitions - 1, math.ceil((1 - alpha) * repetitions) - 1)
    ]


def summarize_long_sequence_study(
    evidence: tuple[LongSequenceEvidence, ...],
    *,
    families: tuple[str, ...],
    seeds: tuple[int, ...],
    bootstrap_repetitions: int = 1_000,
    confidence: float = 0.95,
) -> LongSequenceStudy:
    summaries: list[LongSequenceSummary] = []
    for policy in LongSequencePolicy:
        for horizon in (10, 20, 50):
            rows = tuple(
                item
                for item in evidence
                if item.key.policy is policy and item.key.horizon == horizon
            )
            measured = tuple(
                item
                for item in rows
                if item.outcome in {HorizonOutcome.MEASURED, HorizonOutcome.NEGATIVE_RESULT}
            )
            if len(measured) != len(rows) or not measured:
                summaries.append(
                    LongSequenceSummary(
                        policy,
                        horizon,
                        None,
                        None,
                        None,
                        None,
                        None,
                        StudyClaim.UNSUPPORTED,
                        "one or more family/seed horizons lack physical evidence",
                    )
                )
                continue
            regrets = tuple(
                item.cumulative_regret for item in measured if item.cumulative_regret is not None
            )
            violations = tuple(
                item.violation_count for item in measured if item.violation_count is not None
            )
            evaluations = tuple(
                item.evaluations for item in measured if item.evaluations is not None
            )
            low, high = _bootstrap(regrets, horizon, bootstrap_repetitions, confidence)
            summaries.append(
                LongSequenceSummary(
                    policy,
                    horizon,
                    sum(regrets) / len(regrets),
                    low,
                    high,
                    sum(violations) / len(violations),
                    sum(evaluations) / len(evaluations),
                    StudyClaim.INCONCLUSIVE,
                    "measured horizon retained; paired policy comparison is required",
                )
            )
    claim = (
        StudyClaim.INCONCLUSIVE
        if any(item.claim is StudyClaim.INCONCLUSIVE for item in summaries)
        else StudyClaim.UNSUPPORTED
    )
    reason = "physical horizons are incomplete for a directional state-aware claim"
    return LongSequenceStudy(
        tuple(sorted(set(families))),
        tuple(sorted(set(seeds))),
        (10, 20, 50),
        tuple(sorted(evidence, key=lambda item: item.evidence_id)),
        tuple(summaries),
        bootstrap_repetitions,
        confidence,
        claim,
        reason,
    )


def build_default_long_sequence_study() -> LongSequenceStudy:
    families = ("llama", "qwen")
    seeds = (11, 23, 47)
    evidence = tuple(
        LongSequenceEvidence(
            LongSequenceKey(family, policy, horizon, seed),
            HorizonOutcome.UNSUPPORTED,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (),
            (("corpus", "heldout-corpus-v1"), ("protocol", "long-sequence-v1")),
            "no licensed physical multi-family checkpoint bundle is available",
        )
        for family, policy, horizon, seed in product(
            families, LongSequencePolicy, (10, 20, 50), seeds
        )
    )
    return summarize_long_sequence_study(
        evidence,
        families=families,
        seeds=seeds,
    )


DEFAULT_LONG_SEQUENCE_STUDY = build_default_long_sequence_study()

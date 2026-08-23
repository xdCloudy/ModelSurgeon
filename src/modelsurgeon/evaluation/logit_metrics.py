"""Streaming teacher-to-candidate KL, cosine, and top-k logit similarity metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

LOGIT_METRICS_VERSION = "1"


class LogitMetricError(ValueError):
    """Raised when logit similarity inputs are not comparable."""


@dataclass(frozen=True, slots=True)
class LogitMetricConfig:
    temperature: float = 1.0
    top_k: int = 5
    reduction: str = "mean_over_token_rows"

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise LogitMetricError("logit metric temperature must be finite and positive")
        if self.top_k <= 0:
            raise LogitMetricError("top-k must be positive")
        if self.reduction != "mean_over_token_rows":
            raise LogitMetricError("unsupported logit metric reduction")


@dataclass(frozen=True, slots=True)
class LogitPairBatch:
    teacher: tuple[tuple[float, ...], ...]
    candidate: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.teacher or len(self.teacher) != len(self.candidate):
            raise LogitMetricError("teacher and candidate token rows must align")
        widths = {len(row) for row in (*self.teacher, *self.candidate)}
        if len(widths) != 1 or 0 in widths:
            raise LogitMetricError("teacher/candidate vocabulary width must match and be positive")
        if any(
            not math.isfinite(value)
            for row in (*self.teacher, *self.candidate)
            for value in row
        ):
            raise LogitMetricError("logit metric inputs must be finite")


@dataclass(frozen=True, slots=True)
class LogitMetricResult:
    mean_teacher_to_candidate_kl: float
    mean_cosine_similarity: float
    mean_top_k_agreement: float
    token_rows: int
    temperature: float
    top_k: int
    reduction: str
    version: str = LOGIT_METRICS_VERSION

    def __post_init__(self) -> None:
        values = (
            self.mean_teacher_to_candidate_kl,
            self.mean_cosine_similarity,
            self.mean_top_k_agreement,
        )
        if self.token_rows <= 0 or any(not math.isfinite(value) for value in values):
            raise LogitMetricError("logit metric result must be finite with token rows")
        if self.mean_teacher_to_candidate_kl < -1e-12:
            raise LogitMetricError("KL divergence cannot be materially negative")
        if not -1.0 <= self.mean_cosine_similarity <= 1.0:
            raise LogitMetricError("cosine similarity must be within [-1, 1]")
        if not 0.0 <= self.mean_top_k_agreement <= 1.0:
            raise LogitMetricError("top-k agreement must be within [0, 1]")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "teacher_to_candidate_kl": self.mean_teacher_to_candidate_kl,
            "cosine_similarity": self.mean_cosine_similarity,
            "top_k_agreement": self.mean_top_k_agreement,
            "token_rows": self.token_rows,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "reduction": self.reduction,
        }


def _log_softmax(row: tuple[float, ...], temperature: float) -> tuple[float, ...]:
    scaled = tuple(value / temperature for value in row)
    maximum = max(scaled)
    log_partition = maximum + math.log(
        math.fsum(math.exp(value - maximum) for value in scaled)
    )
    return tuple(value - log_partition for value in scaled)


def _teacher_kl(
    teacher: tuple[float, ...],
    candidate: tuple[float, ...],
    temperature: float,
) -> float:
    teacher_log = _log_softmax(teacher, temperature)
    candidate_log = _log_softmax(candidate, temperature)
    value = math.fsum(
        math.exp(log_p) * (log_p - log_q)
        for log_p, log_q in zip(teacher_log, candidate_log, strict=True)
    )
    return max(0.0, value)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm_sq = math.fsum(value * value for value in left)
    right_norm_sq = math.fsum(value * value for value in right)
    if left_norm_sq == 0.0 and right_norm_sq == 0.0:
        return 1.0
    if left_norm_sq == 0.0 or right_norm_sq == 0.0:
        return 0.0
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    value = dot / math.sqrt(left_norm_sq * right_norm_sq)
    return max(-1.0, min(1.0, value))


def _top_k(row: tuple[float, ...], k: int) -> frozenset[int]:
    resolved = min(k, len(row))
    ranked = sorted(range(len(row)), key=lambda index: (-row[index], index))
    return frozenset(ranked[:resolved])


def evaluate_logit_similarity(
    batches: tuple[LogitPairBatch, ...],
    config: LogitMetricConfig | None = None,
) -> LogitMetricResult:
    """Stream batches into row-mean KL, cosine, and deterministic top-k overlap."""

    if not batches:
        raise LogitMetricError("logit similarity requires at least one batch")
    resolved = config or LogitMetricConfig()
    kl_sum = 0.0
    cosine_sum = 0.0
    agreement_sum = 0.0
    rows = 0
    for batch in batches:
        width = len(batch.teacher[0])
        k = min(resolved.top_k, width)
        for teacher, candidate in zip(batch.teacher, batch.candidate, strict=True):
            kl_sum += _teacher_kl(teacher, candidate, resolved.temperature)
            cosine_sum += _cosine(teacher, candidate)
            teacher_top = _top_k(teacher, k)
            candidate_top = _top_k(candidate, k)
            agreement_sum += len(teacher_top & candidate_top) / k
            rows += 1
    return LogitMetricResult(
        kl_sum / rows,
        cosine_sum / rows,
        agreement_sum / rows,
        rows,
        resolved.temperature,
        resolved.top_k,
        resolved.reduction,
    )

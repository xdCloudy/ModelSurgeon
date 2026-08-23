"""Masked next-token cross entropy and token-weighted perplexity evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from modelsurgeon.evaluation.baseline_cache import BaselineArtifact

PERPLEXITY_EVALUATOR_VERSION = "1"


class PerplexityEvaluationError(ValueError):
    """Raised when causal-LM loss inputs or aggregation are invalid."""


@dataclass(frozen=True, slots=True)
class CausalLMBatch:
    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[bool, ...], ...]
    logits: tuple[tuple[tuple[float, ...], ...], ...]

    def __post_init__(self) -> None:
        if not self.input_ids or len(self.input_ids) != len(self.attention_mask):
            raise PerplexityEvaluationError("batch token IDs and attention masks must align")
        if len(self.input_ids) != len(self.logits):
            raise PerplexityEvaluationError("batch token IDs and logits must align")
        vocab_widths: set[int] = set()
        for ids, mask, rows in zip(
            self.input_ids, self.attention_mask, self.logits, strict=True
        ):
            if len(ids) < 2 or len(mask) != len(ids) or len(rows) != len(ids):
                raise PerplexityEvaluationError(
                    "each sequence needs aligned IDs, mask, and per-position logits"
                )
            vocab_widths.update(len(row) for row in rows)
        if len(vocab_widths) != 1 or 0 in vocab_widths:
            raise PerplexityEvaluationError(
                "logit vocabulary width must be positive and consistent"
            )


@dataclass(frozen=True, slots=True)
class PerplexityResult:
    mean_loss: float
    perplexity: float
    token_count: int
    total_negative_log_likelihood: float
    baseline_mean_loss: float | None
    baseline_perplexity: float | None
    loss_delta: float | None
    perplexity_delta: float | None
    reduction: str = "sum_token_nll_div_valid_shifted_tokens"
    version: str = PERPLEXITY_EVALUATOR_VERSION

    def __post_init__(self) -> None:
        numeric = (
            self.mean_loss,
            self.perplexity,
            self.total_negative_log_likelihood,
        )
        if self.token_count <= 0 or any(not math.isfinite(value) for value in numeric):
            raise PerplexityEvaluationError("perplexity result must be finite with valid tokens")
        if self.mean_loss < 0 or self.perplexity < 1 or self.total_negative_log_likelihood < 0:
            raise PerplexityEvaluationError("perplexity result values are outside valid ranges")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "mean_loss": self.mean_loss,
            "perplexity": self.perplexity,
            "token_count": self.token_count,
            "total_negative_log_likelihood": self.total_negative_log_likelihood,
            "baseline_mean_loss": self.baseline_mean_loss,
            "baseline_perplexity": self.baseline_perplexity,
            "loss_delta": self.loss_delta,
            "perplexity_delta": self.perplexity_delta,
            "reduction": self.reduction,
        }


def _negative_log_probability(row: tuple[float, ...], target: int) -> float:
    if target < 0 or target >= len(row):
        raise PerplexityEvaluationError(
            f"target token {target} is outside vocabulary width {len(row)}"
        )
    if any(not math.isfinite(value) for value in row):
        raise PerplexityEvaluationError("perplexity logits must be finite")
    maximum = max(row)
    log_partition = maximum + math.log(math.fsum(math.exp(value - maximum) for value in row))
    return log_partition - row[target]


def evaluate_perplexity(
    batches: tuple[CausalLMBatch, ...],
    *,
    baseline: BaselineArtifact | None = None,
) -> PerplexityResult:
    """Aggregate shifted causal-LM loss by valid target-token count, never by batch mean."""

    if not batches:
        raise PerplexityEvaluationError("perplexity evaluation requires at least one batch")
    total_nll = 0.0
    token_count = 0
    for batch in batches:
        for ids, mask, rows in zip(
            batch.input_ids, batch.attention_mask, batch.logits, strict=True
        ):
            for position in range(len(ids) - 1):
                if not mask[position + 1]:
                    continue
                total_nll += _negative_log_probability(rows[position], ids[position + 1])
                token_count += 1
    if token_count == 0:
        raise PerplexityEvaluationError("no valid shifted target tokens remain after masking")
    mean_loss = total_nll / token_count
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError as error:
        raise PerplexityEvaluationError("perplexity overflowed finite range") from error
    if not math.isfinite(perplexity):
        raise PerplexityEvaluationError("perplexity is non-finite")

    baseline_loss: float | None = None
    baseline_perplexity: float | None = None
    loss_delta: float | None = None
    perplexity_delta: float | None = None
    if baseline is not None:
        baseline_loss = baseline.mean_loss
        try:
            baseline_perplexity = math.exp(baseline_loss)
        except OverflowError as error:
            raise PerplexityEvaluationError(
                "baseline perplexity overflowed finite range"
            ) from error
        if not math.isfinite(baseline_perplexity):
            raise PerplexityEvaluationError("baseline perplexity is non-finite")
        loss_delta = mean_loss - baseline_loss
        perplexity_delta = perplexity - baseline_perplexity

    return PerplexityResult(
        mean_loss,
        perplexity,
        token_count,
        total_nll,
        baseline_loss,
        baseline_perplexity,
        loss_delta,
        perplexity_delta,
    )

"""Bounded MLP-channel duplicate screening and activation confirmation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from modelsurgeon.features.cosine_similarity import (
    CosineCandidate,
    CosineVector,
    CosineVectorKind,
    cosine_similarity_for_candidate,
)
from modelsurgeon.graph import ComponentId

NEURON_CORRELATION_EXTRACTOR_VERSION = "1"


class NeuronCorrelationError(ValueError):
    """Raised when duplicate-channel ranking cannot preserve its bounded contract."""


@dataclass(frozen=True, slots=True)
class NeuronCorrelationConfig:
    candidate_budget: int = 256
    confirmation_budget: int = 32
    block_size: int = 4096

    def __post_init__(self) -> None:
        if self.candidate_budget <= 0 or self.confirmation_budget <= 0:
            raise NeuronCorrelationError("candidate and confirmation budgets must be positive")
        if self.confirmation_budget > self.candidate_budget:
            raise NeuronCorrelationError(
                "confirmation budget cannot exceed candidate budget"
            )
        if self.block_size <= 0:
            raise NeuronCorrelationError("correlation block size must be positive")


@dataclass(frozen=True, slots=True)
class ChannelRedundancyInput:
    component_id: ComponentId
    weight_values: Sequence[float]
    activation_values: Sequence[float]
    weight_dtype: str = "float64"

    def __post_init__(self) -> None:
        if not self.weight_values or not self.activation_values:
            raise NeuronCorrelationError("channel weights and activations cannot be empty")
        if not self.weight_dtype:
            raise NeuronCorrelationError("channel weight dtype cannot be empty")


@dataclass(frozen=True, slots=True)
class DuplicateChannelFeature:
    left: ComponentId
    right: ComponentId
    weight_cosine: float
    activation_correlation: float
    duplicate_score: float
    screening_rank: int
    confirmation_rank: int
    candidate_budget: int
    confirmation_budget: int
    block_size: int

    def __post_init__(self) -> None:
        if self.left == self.right:
            raise NeuronCorrelationError("duplicate-channel endpoints must be distinct")
        if self.screening_rank <= 0 or self.confirmation_rank <= 0:
            raise NeuronCorrelationError("duplicate-channel ranks must be positive")
        for value in (
            self.weight_cosine,
            self.activation_correlation,
            self.duplicate_score,
        ):
            if not math.isfinite(value):
                raise NeuronCorrelationError("duplicate-channel scores must be finite")
        if not 0.0 <= self.duplicate_score <= 1.0:
            raise NeuronCorrelationError("duplicate-channel score must be within [0, 1]")

    def to_record(self) -> dict[str, object]:
        return {
            "left": str(self.left),
            "right": str(self.right),
            "weight_cosine": self.weight_cosine,
            "activation_correlation": self.activation_correlation,
            "duplicate_score": self.duplicate_score,
            "screening_rank": self.screening_rank,
            "confirmation_rank": self.confirmation_rank,
            "candidate_budget": self.candidate_budget,
            "confirmation_budget": self.confirmation_budget,
            "block_size": self.block_size,
            "activation_zero_variance_policy": "return_zero",
        }


def _block(values: Sequence[float], start: int, end: int) -> tuple[float, ...]:
    try:
        raw = values[start:end]
        block = tuple(float(value) for value in raw)
    except (IndexError, TypeError, ValueError, OverflowError) as error:
        raise NeuronCorrelationError("activation vector could not be read in blocks") from error
    if len(block) != end - start:
        raise NeuronCorrelationError("activation block returned an unexpected length")
    if any(not math.isfinite(value) for value in block):
        raise NeuronCorrelationError("activation vectors must contain finite values")
    return block


def _activation_correlation(
    left: Sequence[float],
    right: Sequence[float],
    block_size: int,
) -> float:
    count = len(left)
    if count <= 0 or len(right) != count:
        raise NeuronCorrelationError("activation vectors must have equal non-zero lengths")
    sum_left = 0.0
    sum_right = 0.0
    sum_left_sq = 0.0
    sum_right_sq = 0.0
    sum_cross = 0.0
    for start in range(0, count, block_size):
        end = min(count, start + block_size)
        left_block = _block(left, start, end)
        right_block = _block(right, start, end)
        sum_left += math.fsum(left_block)
        sum_right += math.fsum(right_block)
        sum_left_sq += math.fsum(value * value for value in left_block)
        sum_right_sq += math.fsum(value * value for value in right_block)
        sum_cross += math.fsum(
            a * b for a, b in zip(left_block, right_block, strict=True)
        )
    covariance_numerator = sum_cross - (sum_left * sum_right / count)
    left_variance = max(0.0, sum_left_sq - sum_left * sum_left / count)
    right_variance = max(0.0, sum_right_sq - sum_right * sum_right / count)
    if left_variance == 0.0 or right_variance == 0.0:
        return 0.0
    value = covariance_numerator / math.sqrt(left_variance * right_variance)
    return max(-1.0, min(1.0, value))


def rank_duplicate_channels(
    channels: Mapping[ComponentId, ChannelRedundancyInput],
    candidates: Iterable[CosineCandidate],
    config: NeuronCorrelationConfig | None = None,
) -> tuple[DuplicateChannelFeature, ...]:
    """Screen a bounded candidate set by weights, then confirm only the best activations."""

    resolved = config or NeuronCorrelationConfig()
    if any(component_id != channel.component_id for component_id, channel in channels.items()):
        raise NeuronCorrelationError("channel mapping key does not match component identity")

    screened: list[tuple[float, CosineCandidate]] = []
    seen: set[CosineCandidate] = set()
    for index, candidate in enumerate(candidates, start=1):
        if index > resolved.candidate_budget:
            raise NeuronCorrelationError("duplicate-channel candidate budget exceeded")
        if candidate in seen:
            continue
        if candidate.left not in channels or candidate.right not in channels:
            raise NeuronCorrelationError("duplicate-channel candidate references unknown channel")
        seen.add(candidate)
        left = channels[candidate.left]
        right = channels[candidate.right]
        similarity = cosine_similarity_for_candidate(
            CosineVector(
                left.component_id,
                left.weight_values,
                CosineVectorKind.TENSOR,
                left.weight_dtype,
                "mlp_channel_weight",
            ),
            CosineVector(
                right.component_id,
                right.weight_values,
                CosineVectorKind.TENSOR,
                right.weight_dtype,
                "mlp_channel_weight",
            ),
            resolved.block_size,
        )
        screened.append((abs(similarity.value), candidate))

    ordered_screen = sorted(
        screened,
        key=lambda item: (-item[0], item[1].left, item[1].right),
    )
    screening_rank = {
        candidate: rank
        for rank, (_, candidate) in enumerate(ordered_screen, start=1)
    }
    confirmation = ordered_screen[: resolved.confirmation_budget]
    confirmed: list[tuple[float, float, float, CosineCandidate]] = []
    for weight_similarity, candidate in confirmation:
        left = channels[candidate.left]
        right = channels[candidate.right]
        activation_correlation = _activation_correlation(
            left.activation_values,
            right.activation_values,
            resolved.block_size,
        )
        duplicate_score = weight_similarity * abs(activation_correlation)
        confirmed.append(
            (
                duplicate_score,
                weight_similarity,
                activation_correlation,
                candidate,
            )
        )

    confirmed.sort(key=lambda item: (-item[0], item[3].left, item[3].right))
    return tuple(
        DuplicateChannelFeature(
            candidate.left,
            candidate.right,
            weight_similarity,
            activation_correlation,
            duplicate_score,
            screening_rank[candidate],
            confirmation_rank,
            resolved.candidate_budget,
            resolved.confirmation_budget,
            resolved.block_size,
        )
        for confirmation_rank, (
            duplicate_score,
            weight_similarity,
            activation_correlation,
            candidate,
        ) in enumerate(confirmed, start=1)
    )

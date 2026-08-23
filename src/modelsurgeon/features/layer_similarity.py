"""Bounded residual-stream and output similarity for comparable transformer layers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from modelsurgeon.features.cosine_similarity import (
    CosineVector,
    CosineVectorKind,
    cosine_similarity_for_candidate,
)
from modelsurgeon.graph import ComponentId

LAYER_SIMILARITY_EXTRACTOR_VERSION = "1"


class LayerSimilarityError(ValueError):
    """Raised when layer similarity inputs violate the bounded comparison contract."""


@dataclass(frozen=True, slots=True)
class LayerSimilarityConfig:
    block_size: int = 4096
    max_pairs: int = 256

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.max_pairs <= 0:
            raise LayerSimilarityError(
                "layer similarity block size and pair limit must be positive"
            )


@dataclass(frozen=True, slots=True)
class LayerSummary:
    layer_id: ComponentId
    residual_shape: tuple[int, ...]
    residual_summary: Sequence[float]
    output_shape: tuple[int, ...]
    output_summary: Sequence[float]
    summary_kind: str = "calibration_mean"

    def __post_init__(self) -> None:
        if not self.residual_shape or not self.output_shape:
            raise LayerSimilarityError("layer residual/output shapes cannot be empty")
        if any(dimension <= 0 for dimension in (*self.residual_shape, *self.output_shape)):
            raise LayerSimilarityError("layer residual/output shapes must be positive")
        if len(self.residual_summary) <= 0 or len(self.output_summary) <= 0:
            raise LayerSimilarityError("layer summaries cannot be empty")
        if not self.summary_kind:
            raise LayerSimilarityError("layer summary provenance cannot be empty")


@dataclass(frozen=True, slots=True, order=True)
class LayerPair:
    left: ComponentId
    right: ComponentId

    def __post_init__(self) -> None:
        if self.left == self.right:
            raise LayerSimilarityError("layer pair endpoints must be distinct")
        if self.right < self.left:
            left = self.right
            right = self.left
            object.__setattr__(self, "left", left)
            object.__setattr__(self, "right", right)


@dataclass(frozen=True, slots=True)
class LayerSimilarity:
    left: ComponentId
    right: ComponentId
    residual_similarity: float | None
    output_similarity: float | None
    skipped: tuple[str, ...]
    block_size: int
    summary_kind: str

    @property
    def comparable(self) -> bool:
        return self.residual_similarity is not None or self.output_similarity is not None

    def to_record(self) -> dict[str, object]:
        return {
            "left": str(self.left),
            "right": str(self.right),
            "residual_similarity": self.residual_similarity,
            "output_similarity": self.output_similarity,
            "skipped": list(self.skipped),
            "comparable": self.comparable,
            "block_size": self.block_size,
            "summary_kind": self.summary_kind,
        }


def _cosine(
    left_id: ComponentId,
    right_id: ComponentId,
    left: Sequence[float],
    right: Sequence[float],
    block_size: int,
) -> float:
    return cosine_similarity_for_candidate(
        CosineVector(left_id, left, CosineVectorKind.OUTPUT, source="layer_summary"),
        CosineVector(right_id, right, CosineVectorKind.OUTPUT, source="layer_summary"),
        block_size,
    ).value


def compare_layers(
    left: LayerSummary,
    right: LayerSummary,
    config: LayerSimilarityConfig | None = None,
) -> LayerSimilarity:
    """Compare only shape-compatible bounded summaries and preserve skip provenance."""

    resolved = config or LayerSimilarityConfig()
    if left.layer_id == right.layer_id:
        raise LayerSimilarityError("cannot compare a layer with itself")
    skipped: list[str] = []
    residual_similarity: float | None = None
    output_similarity: float | None = None

    if left.residual_shape != right.residual_shape:
        skipped.append(
            f"residual_shape_mismatch:{left.residual_shape}!={right.residual_shape}"
        )
    elif len(left.residual_summary) != len(right.residual_summary):
        skipped.append(
            "residual_summary_length_mismatch:"
            f"{len(left.residual_summary)}!={len(right.residual_summary)}"
        )
    else:
        residual_similarity = _cosine(
            left.layer_id,
            right.layer_id,
            left.residual_summary,
            right.residual_summary,
            resolved.block_size,
        )

    if left.output_shape != right.output_shape:
        skipped.append(f"output_shape_mismatch:{left.output_shape}!={right.output_shape}")
    elif len(left.output_summary) != len(right.output_summary):
        skipped.append(
            "output_summary_length_mismatch:"
            f"{len(left.output_summary)}!={len(right.output_summary)}"
        )
    else:
        output_similarity = _cosine(
            left.layer_id,
            right.layer_id,
            left.output_summary,
            right.output_summary,
            resolved.block_size,
        )

    summary_kind = (
        left.summary_kind
        if left.summary_kind == right.summary_kind
        else f"mixed:{left.summary_kind}|{right.summary_kind}"
    )
    pair = LayerPair(left.layer_id, right.layer_id)
    return LayerSimilarity(
        pair.left,
        pair.right,
        residual_similarity,
        output_similarity,
        tuple(skipped),
        resolved.block_size,
        summary_kind,
    )


def extract_layer_similarities(
    layers: Mapping[ComponentId, LayerSummary],
    pairs: Iterable[LayerPair],
    config: LayerSimilarityConfig | None = None,
) -> tuple[LayerSimilarity, ...]:
    """Evaluate only the supplied layer pairs under a hard pair ceiling."""

    resolved = config or LayerSimilarityConfig()
    if any(layer_id != summary.layer_id for layer_id, summary in layers.items()):
        raise LayerSimilarityError("layer mapping key does not match layer identity")
    output: list[LayerSimilarity] = []
    seen: set[LayerPair] = set()
    for index, pair in enumerate(pairs, start=1):
        if index > resolved.max_pairs:
            raise LayerSimilarityError("layer similarity pair limit exceeded")
        if pair in seen:
            continue
        if pair.left not in layers or pair.right not in layers:
            raise LayerSimilarityError("layer pair references an unknown layer")
        seen.add(pair)
        output.append(compare_layers(layers[pair.left], layers[pair.right], resolved))
    return tuple(output)

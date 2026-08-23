"""Candidate-restricted tensor/output cosine similarities without dense all-pairs state."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentId

COSINE_SIMILARITY_EXTRACTOR_VERSION = "1"


class CosineSimilarityError(ValueError):
    """Raised when pairwise cosine extraction cannot preserve its bounded contract."""


class CosineVectorKind(StrEnum):
    TENSOR = "tensor"
    OUTPUT = "output"


class CosineCandidateMode(StrEnum):
    ADJACENT = "adjacent"
    EXPLICIT = "explicit"
    ALL = "all"


@dataclass(frozen=True, slots=True)
class CosineSimilarityConfig:
    candidate_mode: CosineCandidateMode = CosineCandidateMode.ADJACENT
    block_size: int = 4096
    max_candidates: int = 10_000

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise CosineSimilarityError("cosine block size must be positive")
        if self.max_candidates <= 0:
            raise CosineSimilarityError("cosine max_candidates must be positive")


@dataclass(frozen=True, slots=True)
class CosineVector:
    component_id: ComponentId
    values: Sequence[float]
    kind: CosineVectorKind
    storage_dtype: str = "float64"
    source: str = "derived"

    def __post_init__(self) -> None:
        if len(self.values) <= 0:
            raise CosineSimilarityError("cosine vectors cannot be empty")
        if not self.storage_dtype or not self.source:
            raise CosineSimilarityError("cosine vector provenance cannot be empty")


@dataclass(frozen=True, slots=True, order=True)
class CosineCandidate:
    left: ComponentId
    right: ComponentId

    def __post_init__(self) -> None:
        if self.left == self.right:
            raise CosineSimilarityError("cosine candidate endpoints must be distinct")
        if self.right < self.left:
            left = self.right
            right = self.left
            object.__setattr__(self, "left", left)
            object.__setattr__(self, "right", right)


@dataclass(frozen=True, slots=True)
class CosineSimilarity:
    left: ComponentId
    right: ComponentId
    left_kind: CosineVectorKind
    right_kind: CosineVectorKind
    value: float
    element_count: int
    zero_vector: bool
    block_size: int
    left_storage_dtype: str
    right_storage_dtype: str

    def __post_init__(self) -> None:
        if self.left == self.right or self.element_count <= 0 or self.block_size <= 0:
            raise CosineSimilarityError("cosine similarity identity or size is invalid")
        if not math.isfinite(self.value) or not -1.0 <= self.value <= 1.0:
            raise CosineSimilarityError("cosine similarity must be finite within [-1, 1]")

    def to_record(self) -> dict[str, object]:
        return {
            "left": str(self.left),
            "right": str(self.right),
            "left_kind": self.left_kind.value,
            "right_kind": self.right_kind.value,
            "value": self.value,
            "element_count": self.element_count,
            "zero_vector": self.zero_vector,
            "block_size": self.block_size,
            "left_storage_dtype": self.left_storage_dtype,
            "right_storage_dtype": self.right_storage_dtype,
            "zero_vector_policy": "return_zero_and_flag",
        }

    def feature_record(self) -> FeatureRecord:
        precision = PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION,
            self.left_storage_dtype,
            "float64",
        )
        metadata = (
            ("right_component_id", str(self.right)),
            ("left_kind", self.left_kind.value),
            ("right_kind", self.right_kind.value),
            ("right_storage_dtype", self.right_storage_dtype),
            ("element_count", self.element_count),
            ("block_size", self.block_size),
            ("zero_vector", self.zero_vector),
            ("zero_vector_policy", "return_zero_and_flag"),
        )
        return FeatureRecord(
            self.left,
            "pairwise_cosine_similarity",
            FeatureKind.SCALAR,
            self.value,
            "float64",
            "cosine_similarity",
            COSINE_SIMILARITY_EXTRACTOR_VERSION,
            precision,
            metadata=metadata,
        )


def _canonical_candidate(left: ComponentId, right: ComponentId) -> CosineCandidate:
    return CosineCandidate(left, right)


def generate_cosine_candidates(
    component_ids: Iterable[ComponentId],
    config: CosineSimilarityConfig,
    *,
    explicit_candidates: Iterable[CosineCandidate] = (),
) -> Iterator[CosineCandidate]:
    """Yield deterministic candidates lazily and never allocate an N×N pair matrix."""

    ordered = tuple(sorted(set(component_ids)))
    if config.candidate_mode is CosineCandidateMode.EXPLICIT:
        seen: set[CosineCandidate] = set()
        count = 0
        allowed = set(ordered)
        for candidate in explicit_candidates:
            canonical = _canonical_candidate(candidate.left, candidate.right)
            if canonical.left not in allowed or canonical.right not in allowed:
                raise CosineSimilarityError("explicit cosine candidate references an unknown vector")
            if canonical in seen:
                continue
            count += 1
            if count > config.max_candidates:
                raise CosineSimilarityError("cosine candidate count exceeds configured maximum")
            seen.add(canonical)
            yield canonical
        return

    if config.candidate_mode is CosineCandidateMode.ADJACENT:
        count = 0
        for left, right in zip(ordered, ordered[1:], strict=False):
            count += 1
            if count > config.max_candidates:
                raise CosineSimilarityError("cosine candidate count exceeds configured maximum")
            yield CosineCandidate(left, right)
        return

    possible = len(ordered) * (len(ordered) - 1) // 2
    if possible > config.max_candidates:
        raise CosineSimilarityError(
            f"all-pairs candidate count {possible} exceeds configured maximum "
            f"{config.max_candidates}"
        )
    for left_index in range(len(ordered) - 1):
        for right_index in range(left_index + 1, len(ordered)):
            yield CosineCandidate(ordered[left_index], ordered[right_index])


def _block(values: Sequence[float], start: int, end: int) -> tuple[float, ...]:
    try:
        raw = values[start:end]
    except (IndexError, TypeError) as error:
        raise CosineSimilarityError("cosine vector does not support bounded slicing") from error
    try:
        block = tuple(float(value) for value in raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise CosineSimilarityError("cosine vector contains non-numeric values") from error
    if len(block) != end - start:
        raise CosineSimilarityError("cosine vector slice returned an unexpected length")
    if any(not math.isfinite(value) for value in block):
        raise CosineSimilarityError("cosine vectors must contain only finite values")
    return block


def cosine_similarity_for_candidate(
    left: CosineVector,
    right: CosineVector,
    block_size: int,
) -> CosineSimilarity:
    """Compute one cosine in bounded blocks; zero vectors return 0 and are flagged."""

    if left.component_id == right.component_id:
        raise CosineSimilarityError("cosine endpoints must be distinct")
    if block_size <= 0:
        raise CosineSimilarityError("cosine block size must be positive")
    count = len(left.values)
    if count <= 0 or len(right.values) != count:
        raise CosineSimilarityError("cosine vectors must have equal non-zero lengths")

    dot_terms: list[float] = []
    left_norm_terms: list[float] = []
    right_norm_terms: list[float] = []
    for start in range(0, count, block_size):
        end = min(count, start + block_size)
        left_block = _block(left.values, start, end)
        right_block = _block(right.values, start, end)
        dot_terms.append(
            math.fsum(
                a * b for a, b in zip(left_block, right_block, strict=True)
            )
        )
        left_norm_terms.append(math.fsum(value * value for value in left_block))
        right_norm_terms.append(math.fsum(value * value for value in right_block))

    dot = math.fsum(dot_terms)
    left_energy = math.fsum(left_norm_terms)
    right_energy = math.fsum(right_norm_terms)
    zero_vector = left_energy == 0.0 or right_energy == 0.0
    if zero_vector:
        value = 0.0
    else:
        value = dot / math.sqrt(left_energy * right_energy)
        value = max(-1.0, min(1.0, value))
    candidate = _canonical_candidate(left.component_id, right.component_id)
    if candidate.left == left.component_id:
        left_vector = left
        right_vector = right
    else:
        left_vector = right
        right_vector = left
    return CosineSimilarity(
        candidate.left,
        candidate.right,
        left_vector.kind,
        right_vector.kind,
        value,
        count,
        zero_vector,
        block_size,
        left_vector.storage_dtype,
        right_vector.storage_dtype,
    )


def extract_pairwise_cosine_similarities(
    vectors: Mapping[ComponentId, CosineVector],
    config: CosineSimilarityConfig | None = None,
    *,
    explicit_candidates: Iterable[CosineCandidate] = (),
) -> tuple[CosineSimilarity, ...]:
    """Compute only configured candidate pairs with no dense all-pairs allocation."""

    resolved = config or CosineSimilarityConfig()
    if any(component_id != vector.component_id for component_id, vector in vectors.items()):
        raise CosineSimilarityError("cosine vector mapping key does not match component identity")
    candidates = generate_cosine_candidates(
        vectors,
        resolved,
        explicit_candidates=explicit_candidates,
    )
    return tuple(
        cosine_similarity_for_candidate(
            vectors[candidate.left],
            vectors[candidate.right],
            resolved.block_size,
        )
        for candidate in candidates
    )

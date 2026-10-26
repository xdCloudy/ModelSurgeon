"""Memory-bounded seeded farthest-first selection in feature/topology space."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

DIVERSITY_SELECTION_SCHEMA_VERSION: Final[int] = 2
MAX_DIVERSITY_CANDIDATES: Final[int] = 100_000
_NUMERIC_NORMALIZATION: Final[str] = "candidate_pool_minmax_v1"


class DiversitySelectionError(ValueError):
    """Raised when diversity candidates or bounds are incompatible."""


@dataclass(frozen=True, slots=True)
class DiversityCandidate:
    candidate_id: str
    numeric_features: tuple[float, ...]
    categorical_features: tuple[str, ...]
    topology: frozenset[str]

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("cand_"):
            raise DiversitySelectionError("diversity candidates require canonical IDs")
        if any(not math.isfinite(value) for value in self.numeric_features):
            raise DiversitySelectionError("diversity numeric features must be finite")
        if any(not value for value in (*self.categorical_features, *self.topology)):
            raise DiversitySelectionError("diversity categorical/topology values cannot be blank")


@dataclass(frozen=True, slots=True)
class DiversitySelectionConfig:
    seed: int
    numeric_weight: float = 1.0
    categorical_weight: float = 1.0
    topology_weight: float = 1.0
    max_candidates: int = MAX_DIVERSITY_CANDIDATES
    max_selection: int = 4096

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 64:
            raise DiversitySelectionError("diversity seed must be unsigned 64-bit")
        weights = (self.numeric_weight, self.categorical_weight, self.topology_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in weights) or not any(weights):
            raise DiversitySelectionError(
                "diversity weights must be finite, non-negative, and nonzero"
            )
        if not 1 <= self.max_candidates <= MAX_DIVERSITY_CANDIDATES:
            raise DiversitySelectionError("diversity candidate bound must be within 1..100000")
        if not 1 <= self.max_selection <= 4096:
            raise DiversitySelectionError("diversity selection bound must be within 1..4096")


@dataclass(frozen=True, slots=True)
class DiverseSelection:
    candidate_id: str
    rank: int
    minimum_distance: float | None

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "minimum_distance": self.minimum_distance,
        }


@dataclass(frozen=True, slots=True)
class DiversitySelectionReport:
    selections: tuple[DiverseSelection, ...]
    candidate_count: int
    seed: int
    numeric_weight: float
    categorical_weight: float
    topology_weight: float
    working_distance_values: int
    schema_version: int = DIVERSITY_SELECTION_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_count": self.candidate_count,
            "selection_count": len(self.selections),
            "seed": self.seed,
            "weights": {
                "numeric": self.numeric_weight,
                "categorical": self.categorical_weight,
                "topology": self.topology_weight,
            },
            "numeric_normalization": _NUMERIC_NORMALIZATION,
            "working_distance_values": self.working_distance_values,
            "selections": [item.to_record() for item in self.selections],
        }


def select_diverse_candidates(
    candidates: Sequence[DiversityCandidate],
    count: int,
    *,
    config: DiversitySelectionConfig,
) -> DiversitySelectionReport:
    """Select seeded farthest-first candidates using O(candidate-count) working memory."""

    if len(candidates) > config.max_candidates:
        raise DiversitySelectionError("diversity candidate pool exceeds configured bound")
    if count < 0 or count > config.max_selection:
        raise DiversitySelectionError("diversity requested selection exceeds configured bound")
    if count > len(candidates):
        raise DiversitySelectionError("diversity selection cannot exceed candidate count")
    if not candidates or count == 0:
        return DiversitySelectionReport(
            (),
            len(candidates),
            config.seed,
            config.numeric_weight,
            config.categorical_weight,
            config.topology_weight,
            len(candidates),
        )
    ids = tuple(item.candidate_id for item in candidates)
    if len(ids) != len(set(ids)):
        raise DiversitySelectionError("diversity candidate IDs must be unique")
    numeric_width = len(candidates[0].numeric_features)
    categorical_width = len(candidates[0].categorical_features)
    if any(len(item.numeric_features) != numeric_width for item in candidates):
        raise DiversitySelectionError("diversity numeric feature schemas must align")
    if any(len(item.categorical_features) != categorical_width for item in candidates):
        raise DiversitySelectionError("diversity categorical feature schemas must align")
    numeric_bounds = _numeric_normalization_bounds(candidates, numeric_width)

    selected_indexes: set[int] = set()
    minimum_distances = [math.inf] * len(candidates)
    first = min(range(len(candidates)), key=lambda index: _seed_rank(config.seed, ids[index]))
    selections = [DiverseSelection(ids[first], 1, None)]
    selected_indexes.add(first)
    latest = first
    while len(selections) < count:
        best_index: int | None = None
        best_key: tuple[float, int] | None = None
        for index, candidate in enumerate(candidates):
            if index in selected_indexes:
                continue
            distance = _distance(candidate, candidates[latest], config, numeric_bounds)
            if distance < minimum_distances[index]:
                minimum_distances[index] = distance
            key = (minimum_distances[index], -_seed_rank(config.seed, ids[index]))
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            raise DiversitySelectionError("diversity selection exhausted candidates unexpectedly")
        selections.append(
            DiverseSelection(ids[best_index], len(selections) + 1, minimum_distances[best_index])
        )
        selected_indexes.add(best_index)
        latest = best_index
    return DiversitySelectionReport(
        tuple(selections),
        len(candidates),
        config.seed,
        config.numeric_weight,
        config.categorical_weight,
        config.topology_weight,
        len(minimum_distances),
    )


def _distance(
    left: DiversityCandidate,
    right: DiversityCandidate,
    config: DiversitySelectionConfig,
    numeric_bounds: tuple[tuple[float, float, float], ...],
) -> float:
    numeric = (
        math.sqrt(
            math.fsum(
                (_normalize_numeric(left_value, bounds) - _normalize_numeric(right_value, bounds))
                ** 2
                for left_value, right_value, bounds in zip(
                    left.numeric_features,
                    right.numeric_features,
                    numeric_bounds,
                    strict=True,
                )
            )
            / len(left.numeric_features)
        )
        if left.numeric_features
        else 0.0
    )
    categorical = (
        math.fsum(
            left_value != right_value
            for left_value, right_value in zip(
                left.categorical_features, right.categorical_features, strict=True
            )
        )
        / len(left.categorical_features)
        if left.categorical_features
        else 0.0
    )
    union = left.topology | right.topology
    topology = 0.0 if not union else 1.0 - len(left.topology & right.topology) / len(union)
    weight_scale = max(
        config.numeric_weight,
        config.categorical_weight,
        config.topology_weight,
    )
    numeric_weight = config.numeric_weight / weight_scale
    categorical_weight = config.categorical_weight / weight_scale
    topology_weight = config.topology_weight / weight_scale
    total_weight = math.fsum((numeric_weight, categorical_weight, topology_weight))
    return (
        numeric_weight * numeric + categorical_weight * categorical + topology_weight * topology
    ) / total_weight


def _numeric_normalization_bounds(
    candidates: Sequence[DiversityCandidate],
    width: int,
) -> tuple[tuple[float, float, float], ...]:
    """Fit scale-safe min/max transforms on the unlabeled candidate pool."""

    bounds: list[tuple[float, float, float]] = []
    for column in range(width):
        low = min(candidate.numeric_features[column] for candidate in candidates)
        high = max(candidate.numeric_features[column] for candidate in candidates)
        scale = max(abs(low), abs(high))
        if scale == 0.0:
            bounds.append((1.0, 0.0, 0.0))
            continue
        low_scaled = low / scale
        high_scaled = high / scale
        bounds.append((scale, low_scaled, high_scaled - low_scaled))
    return tuple(bounds)


def _normalize_numeric(value: float, bounds: tuple[float, float, float]) -> float:
    scale, low_scaled, width_scaled = bounds
    if width_scaled == 0.0:
        return 0.0
    normalized = (value / scale - low_scaled) / width_scaled
    return min(1.0, max(0.0, normalized))


def _seed_rank(seed: int, candidate_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{candidate_id}".encode("ascii")).digest(), "big")

"""Deterministic random, magnitude, and hand-crafted surgeon ranking baselines."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.experiments.candidates import MutationCandidate
from modelsurgeon.features.schema import FeatureKind, FeatureRecord

RANKING_SCHEMA_VERSION: Final[int] = 1
HEURISTIC_BASELINE_VERSION: Final[str] = "1"


class RankingError(ValueError):
    """Raised when a candidate pool cannot be ranked under the selected policy."""


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate_id: str
    rank: int
    score: float
    selection_propensity: float
    metadata: tuple[tuple[str, str | float | int | bool], ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise RankingError("ranked candidates require an identity")
        if self.rank < 1:
            raise RankingError("candidate ranks are one-based")
        if not math.isfinite(self.score):
            raise RankingError("candidate ranking scores must be finite")
        if not 0.0 <= self.selection_propensity <= 1.0:
            raise RankingError("selection propensity must be within [0, 1]")

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "rank": self.rank,
            "score": self.score,
            "selection_propensity": self.selection_propensity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RankingExclusion:
    candidate_id: str
    reason: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.reason:
            raise RankingError("ranking exclusions require identity and reason")


@dataclass(frozen=True, slots=True)
class RankingResult:
    method: str
    entries: tuple[RankedCandidate, ...]
    exclusions: tuple[RankingExclusion, ...] = ()
    schema_version: int = RANKING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.method:
            raise RankingError("ranking method is required")
        if self.schema_version != RANKING_SCHEMA_VERSION:
            raise RankingError("unsupported ranking schema version")
        ids = tuple(item.candidate_id for item in self.entries)
        if len(ids) != len(set(ids)):
            raise RankingError("ranked candidate IDs must be unique")
        ranks = tuple(item.rank for item in self.entries)
        if ranks != tuple(range(1, len(self.entries) + 1)):
            raise RankingError("ranking entries must have contiguous one-based ranks")

    def top(self, count: int) -> tuple[RankedCandidate, ...]:
        if count < 0:
            raise RankingError("top count cannot be negative")
        return self.entries[:count]

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "entries": [item.to_record() for item in self.entries],
            "exclusions": [
                {"candidate_id": item.candidate_id, "reason": item.reason}
                for item in self.exclusions
            ],
        }


def _candidate_id(candidate: MutationCandidate | str) -> str:
    if isinstance(candidate, MutationCandidate):
        return candidate.candidate_id
    if not candidate:
        raise RankingError("candidate IDs cannot be blank")
    return candidate


def rank_random(
    candidates: Sequence[MutationCandidate | str],
    *,
    seed: int,
    select_count: int | None = None,
) -> RankingResult:
    """Rank a pool by a stable seeded hash and report uniform subset propensities."""

    if isinstance(seed, bool) or seed < 0 or seed >= 1 << 64:
        raise RankingError("random ranking seed must be an unsigned 64-bit integer")
    ids = tuple(_candidate_id(item) for item in candidates)
    if len(ids) != len(set(ids)):
        raise RankingError("random ranking candidate IDs must be unique")
    if select_count is not None and (
        isinstance(select_count, bool) or select_count <= 0 or select_count > len(ids)
    ):
        raise RankingError("select_count must be within 1..pool_size")
    selected = len(ids) if select_count is None else select_count
    propensity = 0.0 if not ids else selected / len(ids)

    keyed: list[tuple[int, str]] = []
    for candidate_id in ids:
        payload = f"random-baseline-v1:{seed}:{candidate_id}".encode()
        score = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        keyed.append((score, candidate_id))
    # A hash collision is resolved by candidate ID, making tie behavior explicit.
    keyed.sort(key=lambda item: (item[0], item[1]))
    maximum = float((1 << 256) - 1)
    entries = tuple(
        RankedCandidate(
            candidate_id,
            rank,
            score / maximum,
            propensity,
            (("seed", seed), ("subset_size", selected)),
        )
        for rank, (score, candidate_id) in enumerate(keyed, start=1)
    )
    return RankingResult("seeded_random_v1", entries)


class MagnitudeNormalization(StrEnum):
    """Cross-shape normalization applied to weight statistics."""

    MEAN_ABSOLUTE = "mean_absolute"
    ROOT_MEAN_SQUARE = "root_mean_square"
    MAX_ABSOLUTE = "max_absolute"


class ComponentAggregation(StrEnum):
    MEAN = "mean"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"


@dataclass(frozen=True, slots=True)
class MagnitudeRankingConfig:
    normalization: MagnitudeNormalization = MagnitudeNormalization.MEAN_ABSOLUTE
    component_aggregation: ComponentAggregation = ComponentAggregation.MEAN
    select_count: int | None = None


DEFAULT_MAGNITUDE_RANKING_CONFIG: Final[MagnitudeRankingConfig] = MagnitudeRankingConfig()


def _scalar_features(
    features: Iterable[FeatureRecord],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for feature in features:
        if feature.kind is not FeatureKind.SCALAR or isinstance(feature.value, tuple):
            continue
        component = str(feature.component_id)
        values = output.setdefault(component, {})
        if feature.name in values:
            raise RankingError(
                f"duplicate scalar feature {feature.name!r} for component {component!r}"
            )
        values[feature.name] = feature.value
    return output


def _component_magnitude(
    values: Mapping[str, float],
    normalization: MagnitudeNormalization,
) -> float:
    count = values.get("weight_count")
    if count is None or count <= 0:
        raise RankingError("magnitude ranking requires positive weight_count")
    if normalization is MagnitudeNormalization.MEAN_ABSOLUTE:
        total = values.get("weight_l1_norm")
        if total is None:
            raise RankingError("mean-absolute ranking requires weight_l1_norm")
        return total / count
    if normalization is MagnitudeNormalization.ROOT_MEAN_SQUARE:
        l2 = values.get("weight_l2_norm")
        if l2 is None:
            raise RankingError("RMS ranking requires weight_l2_norm")
        return l2 / math.sqrt(count)
    maximum = values.get("weight_max_magnitude")
    if maximum is None:
        raise RankingError("max-absolute ranking requires weight_max_magnitude")
    return maximum


def _aggregate(values: Sequence[float], mode: ComponentAggregation) -> float:
    if not values:
        raise RankingError("cannot aggregate an empty magnitude set")
    if mode is ComponentAggregation.MEAN:
        return math.fsum(values) / len(values)
    if mode is ComponentAggregation.MAXIMUM:
        return max(values)
    return min(values)


def rank_magnitude(
    candidates: Sequence[MutationCandidate],
    features: Iterable[FeatureRecord],
    *,
    config: MagnitudeRankingConfig = DEFAULT_MAGNITUDE_RANKING_CONFIG,
) -> RankingResult:
    """Rank graph-compatible candidates from normalized pre-mutation weight statistics."""

    if config.select_count is not None and (
        config.select_count <= 0 or config.select_count > len(candidates)
    ):
        raise RankingError("magnitude select_count must be within 1..pool_size")
    by_component = _scalar_features(features)
    scores: list[tuple[float, str]] = []
    exclusions: list[RankingExclusion] = []

    for candidate in candidates:
        magnitudes: list[float] = []
        missing: list[str] = []
        for component in candidate.affected_components:
            values = by_component.get(str(component))
            if values is None:
                missing.append(str(component))
                continue
            try:
                magnitudes.append(_component_magnitude(values, config.normalization))
            except RankingError as error:
                missing.append(f"{component} ({error})")
        if missing:
            exclusions.append(
                RankingExclusion(
                    candidate.candidate_id,
                    "missing compatible magnitude features: " + ", ".join(missing),
                )
            )
            continue
        score = _aggregate(magnitudes, config.component_aggregation)
        scores.append((score, candidate.candidate_id))

    scores.sort(key=lambda item: (item[0], item[1]))
    selected = len(scores) if config.select_count is None else min(config.select_count, len(scores))
    propensity = 0.0 if not scores else selected / len(scores)
    metadata = (
        ("normalization", config.normalization.value),
        ("component_aggregation", config.component_aggregation.value),
    )
    entries = tuple(
        RankedCandidate(candidate_id, rank, score, propensity, metadata)
        for rank, (score, candidate_id) in enumerate(scores, start=1)
    )
    return RankingResult(
        "magnitude_v1",
        entries,
        tuple(sorted(exclusions, key=lambda item: item.candidate_id)),
    )


class MissingSignalPolicy(StrEnum):
    ERROR = "error"
    NEUTRAL = "neutral"
    WORST = "worst"


class SignalDirection(StrEnum):
    """Whether a larger raw signal makes a candidate more or less pruneable."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True, slots=True)
class HeuristicSignal:
    name: str
    feature_names: tuple[str, ...]
    weight: float
    direction: SignalDirection

    def __post_init__(self) -> None:
        if not self.name or not self.feature_names:
            raise RankingError("heuristic signals require names and feature names")
        if not math.isfinite(self.weight):
            raise RankingError("heuristic signal weights must be finite")


DEFAULT_HEURISTIC_SIGNALS: Final[tuple[HeuristicSignal, ...]] = (
    HeuristicSignal(
        "activation",
        ("activation_rms", "activation_frequency", "activation_activation_frequency"),
        1.0,
        SignalDirection.LOWER_IS_BETTER,
    ),
    HeuristicSignal(
        "magnitude",
        ("weight_l1_norm", "weight_l2_norm", "weight_max_magnitude"),
        1.0,
        SignalDirection.LOWER_IS_BETTER,
    ),
    HeuristicSignal(
        "redundancy",
        ("cosine_similarity", "redundancy_score", "similarity_max"),
        1.0,
        SignalDirection.HIGHER_IS_BETTER,
    ),
    HeuristicSignal(
        "sensitivity",
        ("sensitivity", "gradient_sensitivity", "loss_sensitivity"),
        1.0,
        SignalDirection.LOWER_IS_BETTER,
    ),
)


@dataclass(frozen=True, slots=True)
class HeuristicConfig:
    signals: tuple[HeuristicSignal, ...] = DEFAULT_HEURISTIC_SIGNALS
    missing_policy: MissingSignalPolicy = MissingSignalPolicy.WORST
    version: str = HEURISTIC_BASELINE_VERSION

    def __post_init__(self) -> None:
        if self.version != HEURISTIC_BASELINE_VERSION:
            raise RankingError(f"unsupported heuristic version {self.version!r}")
        names = tuple(item.name for item in self.signals)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise RankingError("heuristic signal names must be unique and canonical")


DEFAULT_HEURISTIC_CONFIG: Final[HeuristicConfig] = HeuristicConfig()


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    candidate_id: str
    total_score: float
    contributions: tuple[tuple[str, float], ...]
    missing_signals: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "total_score": self.total_score,
            "contributions": dict(self.contributions),
            "missing_signals": list(self.missing_signals),
        }


@dataclass(frozen=True, slots=True)
class HeuristicRankingResult:
    ranking: RankingResult
    traces: tuple[DecisionTrace, ...]


def _candidate_feature_average(
    candidate: MutationCandidate,
    by_component: Mapping[str, Mapping[str, float]],
    signal: HeuristicSignal,
) -> float | None:
    values: list[float] = []
    for component in candidate.affected_components:
        component_features = by_component.get(str(component))
        if component_features is None:
            continue
        matched = [
            component_features[name]
            for name in signal.feature_names
            if name in component_features
        ]
        if matched:
            values.append(math.fsum(matched) / len(matched))
    if not values:
        return None
    return math.fsum(values) / len(values)


def _minmax(
    value: float,
    minimum: float,
    maximum: float,
    direction: SignalDirection,
) -> float:
    normalized = 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
    if direction is SignalDirection.LOWER_IS_BETTER:
        return 1.0 - normalized
    return normalized


def rank_heuristic(
    candidates: Sequence[MutationCandidate],
    features: Iterable[FeatureRecord],
    *,
    config: HeuristicConfig = DEFAULT_HEURISTIC_CONFIG,
) -> HeuristicRankingResult:
    """Combine normalized magnitude/activation/sensitivity/redundancy rules with traces."""

    by_component = _scalar_features(features)
    raw: dict[str, dict[str, float | None]] = {}
    extrema: dict[str, tuple[float, float]] = {}

    for signal in config.signals:
        measured: list[float] = []
        for candidate in candidates:
            value = _candidate_feature_average(candidate, by_component, signal)
            raw.setdefault(candidate.candidate_id, {})[signal.name] = value
            if value is not None:
                measured.append(value)
        if measured:
            extrema[signal.name] = (min(measured), max(measured))

    scored: list[tuple[float, str]] = []
    traces: list[DecisionTrace] = []
    exclusions: list[RankingExclusion] = []
    for candidate in candidates:
        contributions: list[tuple[str, float]] = []
        missing: list[str] = []
        failed = False
        for signal in config.signals:
            value = raw[candidate.candidate_id][signal.name]
            bounds = extrema.get(signal.name)
            if value is None or bounds is None:
                missing.append(signal.name)
                if config.missing_policy is MissingSignalPolicy.ERROR:
                    failed = True
                    continue
                normalized = (
                    0.0
                    if config.missing_policy is MissingSignalPolicy.WORST
                    else 0.5
                )
            else:
                normalized = _minmax(value, bounds[0], bounds[1], signal.direction)
            contributions.append((signal.name, normalized * signal.weight))
        if failed:
            exclusions.append(
                RankingExclusion(
                    candidate.candidate_id,
                    "missing required heuristic signals: " + ", ".join(sorted(missing)),
                )
            )
            continue
        total = math.fsum(value for _, value in contributions)
        scored.append((-total, candidate.candidate_id))
        traces.append(
            DecisionTrace(
                candidate.candidate_id,
                total,
                tuple(sorted(contributions)),
                tuple(sorted(missing)),
            )
        )

    # More pruneable is better, so descending heuristic utility becomes ascending negative score.
    scored.sort(key=lambda item: (item[0], item[1]))
    entries = tuple(
        RankedCandidate(
            candidate_id,
            rank,
            -negative_score,
            1.0,
            (
                ("heuristic_version", config.version),
                ("missing_policy", config.missing_policy.value),
            ),
        )
        for rank, (negative_score, candidate_id) in enumerate(scored, start=1)
    )
    return HeuristicRankingResult(
        RankingResult(
            f"heuristic_v{config.version}",
            entries,
            tuple(sorted(exclusions, key=lambda item: item.candidate_id)),
        ),
        tuple(sorted(traces, key=lambda item: item.candidate_id)),
    )
"""Bounded, versioned state representations for state-dependent prediction."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.search.deployable_state import (
    ArchitectureAxis,
    AxisStatus,
    DeployableArchitectureState,
)

STATE_EMBEDDING_SCHEMA_VERSION: Final[int] = 1
STATE_EMBEDDING_VERSION: Final[str] = "state-embedding-v1"


class StateEmbeddingError(ValueError):
    """Raised when a current-state representation cannot be bounded or canonicalized."""


class StateObservationStatus(StrEnum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class StateEvidenceOutcome(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StateEmbeddingError(f"{label} is required")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StateEmbeddingError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise StateEmbeddingError(f"{label} must be finite")
    return result


def _positive(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StateEmbeddingError(f"{label} must be positive")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _bucket(value: str, size: int) -> int:
    digest = hashlib.sha256(value.encode()).digest()
    return int.from_bytes(digest[:8], "big") % size


@dataclass(frozen=True, slots=True)
class StateEmbeddingConfig:
    """Fixed-width limits and vocabulary for one embedding protocol."""

    max_history: int = 16
    max_set_items: int = 128
    mutation_buckets: int = 32
    history_buckets: int = 32
    constraint_buckets: int = 16
    histogram_bins: int = 8
    metric_names: tuple[str, ...] = (
        "quality_loss",
        "load_time_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "latency_seconds",
        "peak_ram_bytes",
        "peak_vram_bytes",
    )
    constraint_names: tuple[str, ...] = ()
    schema_version: int = STATE_EMBEDDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value, label in (
            (self.max_history, "maximum history"),
            (self.max_set_items, "maximum set items"),
            (self.mutation_buckets, "mutation buckets"),
            (self.history_buckets, "history buckets"),
            (self.constraint_buckets, "constraint buckets"),
            (self.histogram_bins, "histogram bins"),
        ):
            _positive(value, label)
        if self.schema_version != STATE_EMBEDDING_SCHEMA_VERSION:
            raise StateEmbeddingError("unsupported state embedding schema version")
        for values, label in (
            (self.metric_names, "metric"),
            (self.constraint_names, "constraint"),
        ):
            if any(not item.strip() for item in values) or len(values) != len(set(values)):
                raise StateEmbeddingError(f"{label} names must be unique and canonical")
        object.__setattr__(self, "metric_names", tuple(sorted(self.metric_names)))
        object.__setattr__(self, "constraint_names", tuple(sorted(self.constraint_names)))

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "max_history": self.max_history,
            "max_set_items": self.max_set_items,
            "mutation_buckets": self.mutation_buckets,
            "history_buckets": self.history_buckets,
            "constraint_buckets": self.constraint_buckets,
            "histogram_bins": self.histogram_bins,
            "metric_names": list(self.metric_names),
            "constraint_names": list(self.constraint_names),
        }


@dataclass(frozen=True, slots=True)
class StateMetricObservation:
    name: str
    status: StateObservationStatus
    value: float | None = None

    def __post_init__(self) -> None:
        _text(self.name, "state metric name")
        if self.status is StateObservationStatus.KNOWN:
            if self.value is None:
                raise StateEmbeddingError("known state metrics require a value")
            _finite(self.value, f"state metric {self.name}")
        elif self.value is not None:
            raise StateEmbeddingError(
                "unknown, unsupported, and failed metrics cannot carry values"
            )

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "status": self.status.value, "value": self.value}


@dataclass(frozen=True, slots=True)
class StateConstraintObservation:
    name: str
    status: StateObservationStatus
    normalized_value: float | None = None

    def __post_init__(self) -> None:
        _text(self.name, "state constraint name")
        if self.status is StateObservationStatus.KNOWN:
            if (
                self.normalized_value is None
                or not 0.0 <= _finite(self.normalized_value, f"state constraint {self.name}") <= 1.0
            ):
                raise StateEmbeddingError("known constraints require a normalized value")
        elif self.normalized_value is not None:
            raise StateEmbeddingError("non-known constraints cannot carry values")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "normalized_value": self.normalized_value,
        }


@dataclass(frozen=True, slots=True)
class StateEvidenceObservation:
    evidence_id: str
    outcome: StateEvidenceOutcome

    def __post_init__(self) -> None:
        _text(self.evidence_id, "state evidence ID")

    def to_record(self) -> dict[str, object]:
        return {"evidence_id": self.evidence_id, "outcome": self.outcome.value}


@dataclass(frozen=True, slots=True)
class CurrentStateInput:
    """Current state plus already accepted mutations and available evidence only."""

    state: DeployableArchitectureState
    accepted_mutations: tuple[str, ...] = ()
    cumulative_metrics: tuple[StateMetricObservation, ...] = ()
    remaining_constraints: tuple[StateConstraintObservation, ...] = ()
    evidence_outcomes: tuple[StateEvidenceObservation, ...] = ()

    def __post_init__(self) -> None:
        if len(self.accepted_mutations) > 128:
            raise StateEmbeddingError("accepted mutation set exceeds the default input bound")
        if any(not item.strip() for item in self.accepted_mutations):
            raise StateEmbeddingError("accepted mutations must be non-empty")
        if len(self.accepted_mutations) != len(set(self.accepted_mutations)):
            raise StateEmbeddingError("accepted mutations must be unique")
        if len(self.cumulative_metrics) != len({item.name for item in self.cumulative_metrics}):
            raise StateEmbeddingError("state metrics must be unique and canonical")
        if len(self.remaining_constraints) != len(
            {item.name for item in self.remaining_constraints}
        ):
            raise StateEmbeddingError("state constraints must be unique and canonical")
        if len(self.evidence_outcomes) != len(
            {item.evidence_id for item in self.evidence_outcomes}
        ):
            raise StateEmbeddingError("state evidence must be unique and canonical")
        object.__setattr__(
            self,
            "cumulative_metrics",
            tuple(sorted(self.cumulative_metrics, key=lambda item: item.name)),
        )
        object.__setattr__(
            self,
            "remaining_constraints",
            tuple(sorted(self.remaining_constraints, key=lambda item: item.name)),
        )
        object.__setattr__(
            self,
            "evidence_outcomes",
            tuple(sorted(self.evidence_outcomes, key=lambda item: item.evidence_id)),
        )
        object.__setattr__(self, "accepted_mutations", tuple(sorted(self.accepted_mutations)))

    def to_record(self) -> dict[str, object]:
        return {
            "state_id": self.state.state_id,
            "accepted_mutations": list(self.accepted_mutations),
            "cumulative_metrics": [item.to_record() for item in self.cumulative_metrics],
            "remaining_constraints": [item.to_record() for item in self.remaining_constraints],
            "evidence_outcomes": [item.to_record() for item in self.evidence_outcomes],
        }


@dataclass(frozen=True, slots=True)
class StateEmbedding:
    state_id: str
    values: tuple[float, ...]
    masks: tuple[float, ...]
    feature_names: tuple[str, ...]
    config: StateEmbeddingConfig
    history_length: int
    history_truncated_count: int
    history_prefix_digest: str | None
    embedding_version: str = STATE_EMBEDDING_VERSION

    def __post_init__(self) -> None:
        if not self.state_id.startswith("state_"):
            raise StateEmbeddingError("state embedding requires a state ID")
        if len(self.values) != len(self.masks) or len(self.values) != len(self.feature_names):
            raise StateEmbeddingError("state embedding vectors and names must align")
        if not self.values or any(
            not math.isfinite(value) for value in (*self.values, *self.masks)
        ):
            raise StateEmbeddingError("state embedding values must be finite")
        if any(mask not in (0.0, 1.0) for mask in self.masks):
            raise StateEmbeddingError("state embedding masks must be binary")
        if self.feature_names != tuple(sorted(set(self.feature_names))):
            raise StateEmbeddingError("state embedding feature names must be unique and canonical")
        if self.history_length < 0 or self.history_truncated_count < 0:
            raise StateEmbeddingError("state embedding history counts must be non-negative")
        if self.history_truncated_count and self.history_prefix_digest is None:
            raise StateEmbeddingError("truncated history requires a prefix digest")

    @property
    def embedding_id(self) -> str:
        payload = {
            "embedding_version": self.embedding_version,
            "state_id": self.state_id,
            "values": self.values,
            "masks": self.masks,
            "feature_names": self.feature_names,
            "config": self.config.to_record(),
            "history_length": self.history_length,
            "history_truncated_count": self.history_truncated_count,
            "history_prefix_digest": self.history_prefix_digest,
        }
        return f"state_embedding_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"

    def to_record(self) -> dict[str, object]:
        return {
            "embedding_id": self.embedding_id,
            "embedding_version": self.embedding_version,
            "state_id": self.state_id,
            "values": list(self.values),
            "masks": list(self.masks),
            "feature_names": list(self.feature_names),
            "config": self.config.to_record(),
            "history_length": self.history_length,
            "history_truncated_count": self.history_truncated_count,
            "history_prefix_digest": self.history_prefix_digest,
        }


def _status_value(status: StateObservationStatus | AxisStatus) -> float:
    return {
        "known": 1.0,
        "unknown": 0.25,
        "unsupported": 0.0,
        "failed": 0.5,
        "predicted": 0.75,
    }[status.value]


def _normalized(value: int | float | None, reference: float = 1_000_000.0) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    raw = float(value)
    return math.copysign(math.log1p(abs(raw)) / math.log1p(reference), raw), 1.0


def _append(
    values: list[float], masks: list[float], names: list[str], name: str, value: float, mask: float
) -> None:
    names.append(name)
    values.append(value)
    masks.append(mask)


def encode_current_state(
    current: CurrentStateInput,
    config: StateEmbeddingConfig | None = None,
) -> StateEmbedding:
    """Encode only current-state and already-observed evidence into fixed width."""

    resolved = config or StateEmbeddingConfig()
    values: list[float] = []
    masks: list[float] = []
    names: list[str] = []
    state = current.state
    axis_values: Mapping[ArchitectureAxis, int | None] = {
        ArchitectureAxis.DEPTH: state.depth,
        ArchitectureAxis.QUERY_HEADS: state.query_heads,
        ArchitectureAxis.KV_HEADS: state.kv_heads,
        ArchitectureAxis.HIDDEN_SIZE: state.hidden_size,
        ArchitectureAxis.EMBEDDING_SIZE: state.embedding_size,
        ArchitectureAxis.LAYER_WIDTHS: None
        if state.layer_widths is None
        else len(state.layer_widths),
    }
    for axis in ArchitectureAxis:
        status = state.axis_status(axis)
        _append(values, masks, names, f"axis_{axis.value}_status", _status_value(status), 1.0)
        if axis in axis_values:
            value, mask = _normalized(axis_values[axis])
        elif axis is ArchitectureAxis.LOW_RANK_FACTORS:
            value, mask = _normalized(
                None if state.low_rank_factors is None else len(state.low_rank_factors)
            )
        elif axis is ArchitectureAxis.SPARSITY:
            value, mask = (
                (sum(item.fraction for item in state.sparsity) / len(state.sparsity), 1.0)
                if state.sparsity
                else (0.0, 0.0)
            )
        else:
            value, mask = (1.0 if getattr(state, axis.value) is not None else 0.0, 1.0)
        _append(values, masks, names, f"axis_{axis.value}_value", value, mask)
    for name, raw in (
        ("parameter_count", state.parameter_count),
        ("storage_bytes", state.storage_bytes),
    ):
        normalized_value, normalized_mask = _normalized(raw)
        _append(values, masks, names, name, normalized_value, normalized_mask)

    widths = state.layer_widths or ()
    for label, sequence in (
        ("attention", tuple(item.attention_width for item in widths)),
        ("mlp", tuple(item.mlp_width for item in widths)),
    ):
        if sequence:
            maximum = float(max(sequence))
            mean = sum(sequence) / len(sequence)
            variance = sum((item - mean) ** 2 for item in sequence) / len(sequence)
            for stat, width_raw in (
                ("mean", mean),
                ("max", maximum),
                ("std", math.sqrt(variance)),
            ):
                normalized_value, normalized_mask = _normalized(width_raw)
                _append(
                    values,
                    masks,
                    names,
                    f"layer_width_{label}_{stat}",
                    normalized_value,
                    normalized_mask,
                )
            histogram = [0.0] * resolved.histogram_bins
            for item in sequence:
                histogram[
                    min(resolved.histogram_bins - 1, int(item / maximum * resolved.histogram_bins))
                ] += 1.0
            for index, count in enumerate(histogram):
                _append(
                    values,
                    masks,
                    names,
                    f"layer_width_{label}_histogram_{index}",
                    count / len(sequence),
                    1.0,
                )
        else:
            for stat in ("mean", "max", "std"):
                _append(values, masks, names, f"layer_width_{label}_{stat}", 0.0, 0.0)
            for index in range(resolved.histogram_bins):
                _append(values, masks, names, f"layer_width_{label}_histogram_{index}", 0.0, 0.0)

    if len(current.accepted_mutations) > resolved.max_set_items:
        raise StateEmbeddingError("accepted mutation set exceeds configured bound")
    accepted_buckets = [0.0] * resolved.mutation_buckets
    for mutation in current.accepted_mutations:
        accepted_buckets[_bucket(mutation, resolved.mutation_buckets)] += 1.0
    for index, value in enumerate(accepted_buckets):
        _append(values, masks, names, f"accepted_mutation_bucket_{index}", value, 1.0)

    history = state.mutation_order
    retained = history[-resolved.max_history :]
    omitted = history[: -resolved.max_history]
    prefix_digest = hashlib.sha256(_canonical(omitted).encode()).hexdigest() if omitted else None
    history_buckets = [0.0] * resolved.history_buckets
    history_mask = [0.0] * resolved.max_history
    for position, mutation in enumerate(retained):
        history_buckets[_bucket(f"{position}:{mutation}", resolved.history_buckets)] += 1.0
        history_mask[position] = 1.0
    for index, value in enumerate(history_buckets):
        _append(values, masks, names, f"history_bucket_{index}", value, 1.0)
    for index, value in enumerate(history_mask):
        _append(values, masks, names, f"history_position_mask_{index}", value, 1.0)
    _append(values, masks, names, "history_length", float(len(history)), 1.0)
    _append(values, masks, names, "history_truncated_count", float(len(omitted)), 1.0)

    metric_by_name = {item.name: item for item in current.cumulative_metrics}
    for name in resolved.metric_names:
        observation = metric_by_name.get(name)
        if observation is None:
            _append(values, masks, names, f"metric_{name}_value", 0.0, 0.0)
            _append(values, masks, names, f"metric_{name}_status", 0.0, 0.0)
        else:
            metric_value = observation.value if observation.value is not None else 0.0
            _append(
                values,
                masks,
                names,
                f"metric_{name}_value",
                _normalized(metric_value)[0],
                float(observation.status is StateObservationStatus.KNOWN),
            )
            _append(
                values,
                masks,
                names,
                f"metric_{name}_status",
                _status_value(observation.status),
                1.0,
            )

    if len(current.remaining_constraints) > resolved.max_set_items:
        raise StateEmbeddingError("constraint set exceeds configured bound")
    constraint_buckets = [0.0] * resolved.constraint_buckets
    constraint_masks = [0.0] * resolved.constraint_buckets
    for constraint_item in current.remaining_constraints:
        constraint_index = _bucket(constraint_item.name, resolved.constraint_buckets)
        constraint_masks[constraint_index] = 1.0
        constraint_buckets[constraint_index] += (
            0.0 if constraint_item.normalized_value is None else constraint_item.normalized_value
        )
    for index, value in enumerate(constraint_buckets):
        _append(values, masks, names, f"constraint_bucket_{index}", value, constraint_masks[index])
    for index, value in enumerate(constraint_masks):
        _append(values, masks, names, f"constraint_status_bucket_{index}", value, 1.0)

    counts = {outcome: 0.0 for outcome in StateEvidenceOutcome}
    if len(current.evidence_outcomes) > resolved.max_set_items:
        raise StateEmbeddingError("evidence set exceeds configured bound")
    for evidence in current.evidence_outcomes:
        counts[evidence.outcome] += 1.0
    for outcome in StateEvidenceOutcome:
        _append(values, masks, names, f"evidence_count_{outcome.value}", counts[outcome], 1.0)

    ordered = sorted(zip(names, values, masks, strict=True))
    return StateEmbedding(
        state_id=state.state_id,
        values=tuple(item[1] for item in ordered),
        masks=tuple(item[2] for item in ordered),
        feature_names=tuple(item[0] for item in ordered),
        config=resolved,
        history_length=len(history),
        history_truncated_count=len(omitted),
        history_prefix_digest=prefix_digest,
    )

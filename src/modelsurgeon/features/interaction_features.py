"""Bounded, pre-mutation features for pairwise and cumulative interactions."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.datasets.interaction import InteractionExample
from modelsurgeon.features.schema import FeatureSampleContext
from modelsurgeon.features.topology import TopologyFeatures
from modelsurgeon.graph import ComponentId

INTERACTION_FEATURE_EXTRACTOR_VERSION: Final[str] = "1"
INTERACTION_FEATURE_PHASE: Final[str] = "pre_mutation"


class InteractionFeatureError(ValueError):
    """Raised when bounded interaction feature extraction is unsafe."""


class InteractionFeatureKind(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class InteractionFeatureBudget:
    """Hard limits for explicit interaction cells and their temporary buffers."""

    max_pairs: int = 256
    block_size: int = 4096
    max_ram_bytes: int = 64 * 1024 * 1024
    max_vram_bytes: int = 0
    max_elapsed_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_pairs <= 0 or self.block_size <= 0:
            raise InteractionFeatureError("interaction feature counts and blocks must be positive")
        if self.max_ram_bytes <= 0 or self.max_vram_bytes < 0:
            raise InteractionFeatureError("interaction feature memory budgets are invalid")
        if not math.isfinite(self.max_elapsed_seconds) or self.max_elapsed_seconds <= 0.0:
            raise InteractionFeatureError("interaction feature time budget must be positive")
        if self.planned_peak_ram_bytes > self.max_ram_bytes:
            raise InteractionFeatureError(
                f"planned interaction feature RAM {self.planned_peak_ram_bytes} exceeds budget "
                f"{self.max_ram_bytes}"
            )
        if self.planned_peak_vram_bytes > self.max_vram_bytes:
            raise InteractionFeatureError("interaction feature VRAM plan exceeds budget")

    @property
    def planned_peak_ram_bytes(self) -> int:
        # Two input blocks, one output block, and fixed per-cell metadata. No
        # pairwise matrix is allocated; the estimate is linear in the explicit cap.
        return self.max_pairs * self.block_size * 3 * 8 + self.max_pairs * 2048

    @property
    def planned_peak_vram_bytes(self) -> int:
        return 0

    def to_record(self) -> dict[str, object]:
        return {
            "max_pairs": self.max_pairs,
            "block_size": self.block_size,
            "max_ram_bytes": self.max_ram_bytes,
            "max_vram_bytes": self.max_vram_bytes,
            "max_elapsed_seconds": self.max_elapsed_seconds,
            "planned_peak_ram_bytes": self.planned_peak_ram_bytes,
            "planned_peak_vram_bytes": self.planned_peak_vram_bytes,
            "pair_generation": "explicit_bounded_cells_only",
        }


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InteractionFeatureError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InteractionFeatureError(f"{label} must be finite")
    return result


def _bounded_cosine(
    left: Sequence[float], right: Sequence[float], block_size: int, label: str
) -> float:
    if len(left) == 0 or len(left) != len(right):
        raise InteractionFeatureError(f"{label} vectors must have equal non-zero lengths")
    dot = 0.0
    left_energy = 0.0
    right_energy = 0.0
    for start in range(0, len(left), block_size):
        end = min(len(left), start + block_size)
        try:
            left_block = tuple(float(value) for value in left[start:end])
            right_block = tuple(float(value) for value in right[start:end])
        except (TypeError, ValueError, OverflowError) as error:
            raise InteractionFeatureError(f"{label} vectors contain non-numeric values") from error
        if len(left_block) != end - start or len(right_block) != end - start:
            raise InteractionFeatureError(f"{label} vector block shape changed")
        if any(not math.isfinite(value) for value in (*left_block, *right_block)):
            raise InteractionFeatureError(f"{label} vectors must be finite")
        dot += math.fsum(a * b for a, b in zip(left_block, right_block, strict=True))
        left_energy += math.fsum(value * value for value in left_block)
        right_energy += math.fsum(value * value for value in right_block)
    if left_energy == 0.0 or right_energy == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / math.sqrt(left_energy * right_energy)))


@dataclass(frozen=True, slots=True)
class InteractionFeatureCell:
    """Pre-mutation observations for one explicit component pair and target cell."""

    interaction: InteractionExample
    left_component: ComponentId
    right_component: ComponentId
    sample_context: FeatureSampleContext
    left_activation: Sequence[float] | None = None
    right_activation: Sequence[float] | None = None
    left_gradient: Sequence[float] | None = None
    right_gradient: Sequence[float] | None = None
    left_topology: TopologyFeatures | None = None
    right_topology: TopologyFeatures | None = None
    redundancy_score: float | None = None
    source_phase: str = INTERACTION_FEATURE_PHASE

    def __post_init__(self) -> None:
        if self.left_component == self.right_component:
            raise InteractionFeatureError("interaction feature endpoints must be distinct")
        if self.source_phase != INTERACTION_FEATURE_PHASE:
            raise InteractionFeatureError(
                "interaction features may only consume pre-mutation observations"
            )
        if self.redundancy_score is not None:
            score = _finite(self.redundancy_score, "redundancy score")
            if not 0.0 <= score <= 1.0:
                raise InteractionFeatureError("redundancy score must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class InteractionFeatureValue:
    name: str
    value: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InteractionFeatureError("interaction feature name is required")
        _finite(self.value, f"interaction feature {self.name}")

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class InteractionFeatureRecord:
    example_id: str
    left_component: ComponentId
    right_component: ComponentId
    parent_state_id: str
    mutation_ids: tuple[str, ...]
    interaction_kind: str
    outcome: InteractionFeatureKind
    values: tuple[InteractionFeatureValue, ...]
    missing_features: tuple[str, ...]
    sample_context: FeatureSampleContext
    budget: InteractionFeatureBudget
    elapsed_seconds: float
    source_phase: str = INTERACTION_FEATURE_PHASE
    extractor_version: str = INTERACTION_FEATURE_EXTRACTOR_VERSION

    def __post_init__(self) -> None:
        if not self.example_id or not self.parent_state_id or not self.mutation_ids:
            raise InteractionFeatureError("interaction feature identity is incomplete")
        if self.left_component == self.right_component:
            raise InteractionFeatureError("interaction feature endpoints must be distinct")
        if self.source_phase != INTERACTION_FEATURE_PHASE:
            raise InteractionFeatureError("interaction feature source phase is not pre-mutation")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise InteractionFeatureError("interaction feature elapsed time is invalid")
        names = tuple(item.name for item in self.values)
        if names != tuple(sorted(set(names))):
            raise InteractionFeatureError("interaction feature names must be unique and canonical")
        if self.missing_features != tuple(sorted(set(self.missing_features))):
            raise InteractionFeatureError("missing interaction feature names must be canonical")

    @property
    def feature_id(self) -> str:
        payload = {
            "extractor_version": self.extractor_version,
            "example_id": self.example_id,
            "left_component": str(self.left_component),
            "right_component": str(self.right_component),
            "sample_context": self.sample_context.to_record(),
            "values": [item.to_record() for item in self.values],
            "missing_features": list(self.missing_features),
            "source_phase": self.source_phase,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return f"interaction_feature_{digest}"

    def to_record(self) -> dict[str, object]:
        return {
            "feature_id": self.feature_id,
            "extractor_version": self.extractor_version,
            "example_id": self.example_id,
            "left_component": str(self.left_component),
            "right_component": str(self.right_component),
            "parent_state_id": self.parent_state_id,
            "mutation_ids": list(self.mutation_ids),
            "interaction_kind": self.interaction_kind,
            "outcome": self.outcome.value,
            "values": [item.to_record() for item in self.values],
            "missing_features": list(self.missing_features),
            "sample_context": self.sample_context.to_record(),
            "budget": self.budget.to_record(),
            "elapsed_seconds": self.elapsed_seconds,
            "source_phase": self.source_phase,
        }


def _topology_distance(left: TopologyFeatures, right: TopologyFeatures) -> float:
    return float(
        abs(left.depth - right.depth)
        + abs(left.position - right.position)
        + abs(left.sibling_position - right.sibling_position)
    )


def extract_interaction_features(
    cells: Iterable[InteractionFeatureCell],
    budget: InteractionFeatureBudget | None = None,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[InteractionFeatureRecord, ...]:
    """Extract bounded pair features from explicit cells, never from all pairs."""

    resolved = budget or InteractionFeatureBudget()
    started = clock()
    records: list[InteractionFeatureRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for index, cell in enumerate(cells, start=1):
        if index > resolved.max_pairs:
            raise InteractionFeatureError("interaction feature pair budget exceeded")
        if clock() - started > resolved.max_elapsed_seconds:
            raise InteractionFeatureError("interaction feature time budget exceeded")
        identity = (
            cell.interaction.example_id,
            str(cell.left_component),
            str(cell.right_component),
        )
        if identity in seen:
            raise InteractionFeatureError("duplicate interaction feature cell")
        seen.add(identity)
        values: list[InteractionFeatureValue] = []
        missing: list[str] = []
        if cell.left_activation is not None and cell.right_activation is not None:
            values.append(
                InteractionFeatureValue(
                    "activation_overlap",
                    _bounded_cosine(
                        cell.left_activation,
                        cell.right_activation,
                        resolved.block_size,
                        "activation",
                    ),
                )
            )
        else:
            missing.append("activation_overlap")
        if cell.left_gradient is not None and cell.right_gradient is not None:
            values.append(
                InteractionFeatureValue(
                    "gradient_overlap",
                    _bounded_cosine(
                        cell.left_gradient,
                        cell.right_gradient,
                        resolved.block_size,
                        "gradient",
                    ),
                )
            )
        else:
            missing.append("gradient_overlap")
        if cell.left_topology is not None and cell.right_topology is not None:
            values.append(
                InteractionFeatureValue(
                    "topology_distance",
                    _topology_distance(cell.left_topology, cell.right_topology),
                )
            )
        else:
            missing.append("topology_distance")
        if cell.redundancy_score is None:
            missing.append("redundancy_score")
        else:
            values.append(InteractionFeatureValue("redundancy_score", cell.redundancy_score))
        values.extend(
            (
                InteractionFeatureValue(
                    "mutation_order_length", float(len(cell.interaction.mutation_ids))
                ),
                InteractionFeatureValue(
                    "order_sensitive",
                    float(cell.interaction.kind.value in {"ordered_pair", "cumulative"}),
                ),
            )
        )
        non_additive = {
            metric.name: metric.value for metric in cell.interaction.non_additivity_metrics
        }
        if non_additive:
            values.extend(
                InteractionFeatureValue(f"cumulative_error_{name}", value)
                for name, value in sorted(non_additive.items())
            )
        else:
            missing.append("cumulative_error")
        values.sort(key=lambda item: item.name)
        missing_tuple = tuple(sorted(set(missing)))
        elapsed = max(0.0, clock() - started)
        records.append(
            InteractionFeatureRecord(
                example_id=cell.interaction.example_id,
                left_component=cell.left_component,
                right_component=cell.right_component,
                parent_state_id=cell.interaction.parent_state_id,
                mutation_ids=cell.interaction.mutation_ids,
                interaction_kind=cell.interaction.kind.value,
                outcome=InteractionFeatureKind(cell.interaction.outcome.value),
                values=tuple(values),
                missing_features=missing_tuple,
                sample_context=cell.sample_context,
                budget=resolved,
                elapsed_seconds=elapsed,
            )
        )
    return tuple(records)

"""Versioned deployable architecture states and conservative distances."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

DEPLOYABLE_STATE_SCHEMA_VERSION: Final[int] = 1
DEPLOYABLE_STATE_DISTANCE_SCHEMA_VERSION: Final[int] = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DeployableStateError(ValueError):
    """Raised when a deployable state or distance record is unsafe."""


class ArchitectureAxis(StrEnum):
    DEPTH = "depth"
    LAYER_WIDTHS = "layer_widths"
    QUERY_HEADS = "query_heads"
    KV_HEADS = "kv_heads"
    HIDDEN_SIZE = "hidden_size"
    EMBEDDING_SIZE = "embedding_size"
    LOW_RANK_FACTORS = "low_rank_factors"
    SPARSITY = "sparsity"
    QUANTIZATION = "quantization"
    PLACEMENT = "placement"


class AxisStatus(StrEnum):
    KNOWN = "known"
    PREDICTED = "predicted"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class MaterializationStatus(StrEnum):
    COMPLETE = "complete"
    PREDICTED = "predicted"
    NOT_YET_MATERIALIZED = "not_yet_materialized"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class ArtifactContainerFormat(StrEnum):
    HUGGINGFACE = "safetensors"
    GGUF = "GGUF"


class DistanceStatus(StrEnum):
    COMPUTED = "computed"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeployableStateError(f"{label} is required")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise DeployableStateError(f"{label} must be a lowercase SHA-256")
    return result


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DeployableStateError(f"{label} must be a positive integer")
    return value


def _nonnegative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DeployableStateError(f"{label} must be a non-negative integer")
    return value


def _fraction(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeployableStateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise DeployableStateError(f"{label} must be finite and within [0, 1]")
    return result


@dataclass(frozen=True, slots=True)
class LayerWidth:
    """Width facts for one logical transformer layer."""

    layer_index: int
    attention_width: int
    mlp_width: int

    def __post_init__(self) -> None:
        _nonnegative(self.layer_index, "layer index")
        _positive(self.attention_width, "attention width")
        _positive(self.mlp_width, "MLP width")

    def to_record(self) -> dict[str, int]:
        return {
            "layer_index": self.layer_index,
            "attention_width": self.attention_width,
            "mlp_width": self.mlp_width,
        }


@dataclass(frozen=True, slots=True)
class LowRankFactor:
    component: str
    rank: int

    def __post_init__(self) -> None:
        _text(self.component, "low-rank component")
        _positive(self.rank, "low-rank factor")

    def to_record(self) -> dict[str, object]:
        return {"component": self.component, "rank": self.rank}


@dataclass(frozen=True, slots=True)
class SparsityEntry:
    component: str
    fraction: float

    def __post_init__(self) -> None:
        _text(self.component, "sparsity component")
        _fraction(self.fraction, "sparsity fraction")

    def to_record(self) -> dict[str, object]:
        return {"component": self.component, "fraction": self.fraction}


@dataclass(frozen=True, slots=True)
class QuantizationState:
    codec: str
    tensor_codecs: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _text(self.codec, "quantization codec")
        components = tuple(component for component, _ in self.tensor_codecs)
        if components != tuple(sorted(set(components))):
            raise DeployableStateError("quantization components must be unique and canonical")
        if any(not component or not codec for component, codec in self.tensor_codecs):
            raise DeployableStateError("quantization component and codec names are required")

    def to_record(self) -> dict[str, object]:
        return {
            "codec": self.codec,
            "tensor_codecs": [
                {"component": component, "codec": codec} for component, codec in self.tensor_codecs
            ],
        }


@dataclass(frozen=True, slots=True)
class PlacementState:
    assignments: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        components = tuple(component for component, _ in self.assignments)
        if not components or components != tuple(sorted(set(components))):
            raise DeployableStateError("placement assignments must be non-empty and canonical")
        if any(not component or not device for component, device in self.assignments):
            raise DeployableStateError("placement components and devices are required")

    def to_record(self) -> dict[str, object]:
        return {
            "assignments": [
                {"component": component, "device": device} for component, device in self.assignments
            ]
        }


@dataclass(frozen=True, slots=True)
class ArtifactLineage:
    """Source/output identity and materialization status for one state."""

    source_artifact_digest: str
    format: ArtifactContainerFormat
    status: MaterializationStatus
    materialized_artifact_digest: str | None = None
    size_bytes: int | None = None
    physical_outcome_id: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _digest(self.source_artifact_digest, "source artifact digest")
        if self.materialized_artifact_digest is not None:
            _digest(self.materialized_artifact_digest, "materialized artifact digest")
        if self.size_bytes is not None:
            _positive(self.size_bytes, "artifact size")
        if self.physical_outcome_id is not None and not self.physical_outcome_id.startswith(
            "outcome_"
        ):
            raise DeployableStateError("physical outcome IDs must be content-addressed")
        if self.status is MaterializationStatus.COMPLETE:
            if self.materialized_artifact_digest is None or self.size_bytes is None:
                raise DeployableStateError("complete artifacts require output digest and size")
            if self.physical_outcome_id is None:
                raise DeployableStateError("complete artifacts require physical outcome evidence")
            if self.reason is not None:
                raise DeployableStateError("complete artifacts cannot carry a reason")
        elif self.status is MaterializationStatus.PREDICTED:
            if self.materialized_artifact_digest is not None or not self.reason:
                raise DeployableStateError(
                    "predicted artifacts require a reason and no output digest"
                )
        else:
            if self.materialized_artifact_digest is not None or self.size_bytes is not None:
                raise DeployableStateError("unmaterialized artifacts cannot carry output facts")
            if not self.reason:
                raise DeployableStateError("unmaterialized artifacts require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "source_artifact_digest": self.source_artifact_digest,
            "format": self.format.value,
            "status": self.status.value,
            "materialized_artifact_digest": self.materialized_artifact_digest,
            "size_bytes": self.size_bytes,
            "physical_outcome_id": self.physical_outcome_id,
            "reason": self.reason,
        }


def _axis_names() -> tuple[ArchitectureAxis, ...]:
    return tuple(ArchitectureAxis)


@dataclass(frozen=True, slots=True)
class DeployableArchitectureState:
    """Complete state identity without holding a live framework model."""

    model_family: str
    model_revision: str
    depth: int | None
    layer_widths: tuple[LayerWidth, ...] | None
    query_heads: int | None
    kv_heads: int | None
    hidden_size: int | None
    embedding_size: int | None
    low_rank_factors: tuple[LowRankFactor, ...] | None
    sparsity: tuple[SparsityEntry, ...] | None
    quantization: QuantizationState | None
    placement: PlacementState | None
    artifact: ArtifactLineage
    mutation_order: tuple[str, ...] = ()
    predicted_axes: tuple[ArchitectureAxis, ...] = ()
    unknown_axes: tuple[ArchitectureAxis, ...] = ()
    unsupported_axes: tuple[ArchitectureAxis, ...] = ()
    schema_version: int = DEPLOYABLE_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.model_family, "model family")
        _text(self.model_revision, "model revision")
        if self.schema_version != DEPLOYABLE_STATE_SCHEMA_VERSION:
            raise DeployableStateError("unsupported deployable state schema version")
        if self.depth is not None:
            _positive(self.depth, "depth")
        if self.layer_widths is not None:
            indexes = tuple(item.layer_index for item in self.layer_widths)
            if indexes != tuple(range(len(indexes))):
                raise DeployableStateError("layer widths must cover canonical contiguous indices")
        for axis_value, label in (
            (self.query_heads, "query heads"),
            (self.kv_heads, "KV heads"),
            (self.hidden_size, "hidden size"),
            (self.embedding_size, "embedding size"),
        ):
            if axis_value is not None:
                _positive(axis_value, label)
        if (
            self.query_heads is not None
            and self.kv_heads is not None
            and (self.query_heads < self.kv_heads or self.query_heads % self.kv_heads)
        ):
            raise DeployableStateError("query heads must be a positive multiple of KV heads")
        if self.low_rank_factors is not None:
            components = tuple(item.component for item in self.low_rank_factors)
            if components != tuple(sorted(set(components))):
                raise DeployableStateError("low-rank factors must be unique and canonical")
        if self.sparsity is not None:
            components = tuple(item.component for item in self.sparsity)
            if components != tuple(sorted(set(components))):
                raise DeployableStateError("sparsity entries must be unique and canonical")
        if len(self.mutation_order) != len(set(self.mutation_order)) or any(
            not item.strip() for item in self.mutation_order
        ):
            raise DeployableStateError("mutation order must contain unique non-empty IDs")
        for values, label in (
            (self.predicted_axes, "predicted"),
            (self.unknown_axes, "unknown"),
            (self.unsupported_axes, "unsupported"),
        ):
            if values != tuple(sorted(set(values))):
                raise DeployableStateError(f"{label} axes must be unique and canonical")
        status_sets = (
            set(self.predicted_axes),
            set(self.unknown_axes),
            set(self.unsupported_axes),
        )
        if (
            status_sets[0] & status_sets[1]
            or status_sets[0] & status_sets[2]
            or status_sets[1] & status_sets[2]
        ):
            raise DeployableStateError("axis statuses cannot overlap")
        values_by_axis: dict[ArchitectureAxis, object | None] = {
            ArchitectureAxis.DEPTH: self.depth,
            ArchitectureAxis.LAYER_WIDTHS: self.layer_widths,
            ArchitectureAxis.QUERY_HEADS: self.query_heads,
            ArchitectureAxis.KV_HEADS: self.kv_heads,
            ArchitectureAxis.HIDDEN_SIZE: self.hidden_size,
            ArchitectureAxis.EMBEDDING_SIZE: self.embedding_size,
            ArchitectureAxis.LOW_RANK_FACTORS: self.low_rank_factors,
            ArchitectureAxis.SPARSITY: self.sparsity,
            ArchitectureAxis.QUANTIZATION: self.quantization,
            ArchitectureAxis.PLACEMENT: self.placement,
        }
        for axis in _axis_names():
            value = values_by_axis[axis]
            status = self.axis_status(axis)
            if value is None and status is AxisStatus.KNOWN:
                raise DeployableStateError(f"missing value for known axis {axis.value}")
            if value is not None and status in {AxisStatus.UNKNOWN, AxisStatus.UNSUPPORTED}:
                raise DeployableStateError(f"status {status.value} cannot carry axis {axis.value}")
            if status is AxisStatus.PREDICTED and value is None:
                raise DeployableStateError(f"predicted axis {axis.value} requires a value")

    def axis_status(self, axis: ArchitectureAxis) -> AxisStatus:
        if axis in self.predicted_axes:
            return AxisStatus.PREDICTED
        if axis in self.unknown_axes:
            return AxisStatus.UNKNOWN
        if axis in self.unsupported_axes:
            return AxisStatus.UNSUPPORTED
        return AxisStatus.KNOWN

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_family": self.model_family,
            "model_revision": self.model_revision,
            "axes": self._axes_record(include_status=True),
            "artifact": self.artifact.to_record(),
            "mutation_order": list(self.mutation_order),
        }

    def _axes_record(self, *, include_status: bool) -> dict[str, object]:
        values: dict[ArchitectureAxis, object | None] = {
            ArchitectureAxis.DEPTH: self.depth,
            ArchitectureAxis.LAYER_WIDTHS: None
            if self.layer_widths is None
            else [item.to_record() for item in self.layer_widths],
            ArchitectureAxis.QUERY_HEADS: self.query_heads,
            ArchitectureAxis.KV_HEADS: self.kv_heads,
            ArchitectureAxis.HIDDEN_SIZE: self.hidden_size,
            ArchitectureAxis.EMBEDDING_SIZE: self.embedding_size,
            ArchitectureAxis.LOW_RANK_FACTORS: None
            if self.low_rank_factors is None
            else [item.to_record() for item in self.low_rank_factors],
            ArchitectureAxis.SPARSITY: None
            if self.sparsity is None
            else [item.to_record() for item in self.sparsity],
            ArchitectureAxis.QUANTIZATION: None
            if self.quantization is None
            else self.quantization.to_record(),
            ArchitectureAxis.PLACEMENT: None
            if self.placement is None
            else self.placement.to_record(),
        }
        return {
            axis.value: {
                "status": self.axis_status(axis).value,
                "value": values[axis],
            }
            if include_status
            else values[axis]
            for axis in _axis_names()
        }

    @property
    def state_id(self) -> str:
        digest = hashlib.sha256(_canonical(self._identity_record()).encode()).hexdigest()
        return f"state_{digest}"

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["state_id"] = self.state_id
        return record


@dataclass(frozen=True, slots=True)
class DistanceComponent:
    axis: str
    value: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.axis, "distance axis")
        if self.value is not None and (not math.isfinite(self.value) or self.value < 0):
            raise DeployableStateError("distance components must be finite and non-negative")
        if self.value is None and not self.reason:
            raise DeployableStateError("undefined distance components require a reason")

    def to_record(self) -> dict[str, object]:
        return {"axis": self.axis, "value": self.value, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ArchitectureDistance:
    left_state_id: str
    right_state_id: str
    status: DistanceStatus
    total: float | None
    components: tuple[DistanceComponent, ...]
    schema_version: int = DEPLOYABLE_STATE_DISTANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.left_state_id.startswith("state_") or not self.right_state_id.startswith(
            "state_"
        ):
            raise DeployableStateError("architecture distances require state IDs")
        if self.schema_version != DEPLOYABLE_STATE_DISTANCE_SCHEMA_VERSION:
            raise DeployableStateError("unsupported architecture distance schema version")
        axes = tuple(item.axis for item in self.components)
        if axes != tuple(sorted(set(axes))):
            raise DeployableStateError("distance components must be unique and canonical")
        if self.status is DistanceStatus.COMPUTED and (
            self.total is None or not math.isfinite(self.total) or self.total < 0
        ):
            raise DeployableStateError("computed distances require a finite total")
        if self.status is not DistanceStatus.COMPUTED and self.total is not None:
            raise DeployableStateError("undefined distances cannot carry a total")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "left_state_id": self.left_state_id,
            "right_state_id": self.right_state_id,
            "status": self.status.value,
            "total": self.total,
            "components": [item.to_record() for item in self.components],
        }


def _normalized_delta(left: float, right: float) -> float:
    return abs(left - right) / max(1.0, abs(left), abs(right))


def _mapping_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return sum(_normalized_delta(left.get(key, 0.0), right.get(key, 0.0)) for key in keys) / max(
        1, len(keys)
    )


def architecture_distance(
    left: DeployableArchitectureState,
    right: DeployableArchitectureState,
) -> ArchitectureDistance:
    """Compute a symmetric, axis-decomposed distance without collapsing context."""

    components: list[DistanceComponent] = []
    family_value = 1.0 if left.model_family != right.model_family else 0.0
    components.append(DistanceComponent("model_family", family_value))
    components.append(
        DistanceComponent(
            ArchitectureAxis.DEPTH.value,
            None
            if left.depth is None or right.depth is None
            else _normalized_delta(float(left.depth), float(right.depth)),
            "depth is unknown or unsupported"
            if left.depth is None or right.depth is None
            else None,
        )
    )
    if left.layer_widths is None or right.layer_widths is None:
        components.append(
            DistanceComponent(ArchitectureAxis.LAYER_WIDTHS.value, None, "layer widths unavailable")
        )
    else:
        left_widths = {
            str(item.layer_index): float(item.attention_width + item.mlp_width)
            for item in left.layer_widths
        }
        right_widths = {
            str(item.layer_index): float(item.attention_width + item.mlp_width)
            for item in right.layer_widths
        }
        components.append(
            DistanceComponent(
                ArchitectureAxis.LAYER_WIDTHS.value, _mapping_distance(left_widths, right_widths)
            )
        )
    for axis, left_value, right_value in (
        (ArchitectureAxis.QUERY_HEADS, left.query_heads, right.query_heads),
        (ArchitectureAxis.KV_HEADS, left.kv_heads, right.kv_heads),
        (ArchitectureAxis.HIDDEN_SIZE, left.hidden_size, right.hidden_size),
        (ArchitectureAxis.EMBEDDING_SIZE, left.embedding_size, right.embedding_size),
    ):
        components.append(
            DistanceComponent(
                axis.value,
                None
                if left_value is None or right_value is None
                else _normalized_delta(float(left_value), float(right_value)),
                f"{axis.value} is unknown or unsupported"
                if left_value is None or right_value is None
                else None,
            )
        )
    if left.low_rank_factors is None or right.low_rank_factors is None:
        components.append(
            DistanceComponent(
                ArchitectureAxis.LOW_RANK_FACTORS.value, None, "low-rank factors unavailable"
            )
        )
    else:
        components.append(
            DistanceComponent(
                ArchitectureAxis.LOW_RANK_FACTORS.value,
                _mapping_distance(
                    {item.component: float(item.rank) for item in left.low_rank_factors},
                    {item.component: float(item.rank) for item in right.low_rank_factors},
                ),
            )
        )
    if left.sparsity is None or right.sparsity is None:
        components.append(
            DistanceComponent(ArchitectureAxis.SPARSITY.value, None, "sparsity unavailable")
        )
    else:
        components.append(
            DistanceComponent(
                ArchitectureAxis.SPARSITY.value,
                _mapping_distance(
                    {item.component: item.fraction for item in left.sparsity},
                    {item.component: item.fraction for item in right.sparsity},
                ),
            )
        )
    components.append(
        DistanceComponent(
            ArchitectureAxis.QUANTIZATION.value,
            None
            if left.quantization is None or right.quantization is None
            else float(left.quantization != right.quantization),
            "quantization unavailable"
            if left.quantization is None or right.quantization is None
            else None,
        )
    )
    components.append(
        DistanceComponent(
            ArchitectureAxis.PLACEMENT.value,
            None
            if left.placement is None or right.placement is None
            else float(left.placement != right.placement),
            "placement unavailable" if left.placement is None or right.placement is None else None,
        )
    )
    components.append(
        DistanceComponent("mutation_order", float(left.mutation_order != right.mutation_order))
    )
    components = sorted(components, key=lambda item: item.axis)
    if any(item.value is None for item in components):
        return ArchitectureDistance(
            left.state_id,
            right.state_id,
            DistanceStatus.UNKNOWN,
            None,
            tuple(components),
        )
    total = math.fsum(item.value or 0.0 for item in components)
    return ArchitectureDistance(
        left.state_id,
        right.state_id,
        DistanceStatus.COMPUTED,
        total,
        tuple(components),
    )


def migrate_deployable_state_record(raw: Mapping[str, object]) -> dict[str, object]:
    """Accept only the current record or a minimal v0-to-v1 additive migration."""

    value = dict(raw)
    version = value.get("schema_version")
    if version == DEPLOYABLE_STATE_SCHEMA_VERSION:
        return value
    if version != 0:
        raise DeployableStateError("unknown deployable state schema version")
    if "model_family" not in value or "model_revision" not in value or "artifact" not in value:
        raise DeployableStateError("v0 deployable state is missing required identity fields")
    value["schema_version"] = DEPLOYABLE_STATE_SCHEMA_VERSION
    value.setdefault("mutation_order", [])
    value.setdefault("axes", {})
    return value


def _record_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise DeployableStateError(f"{label} must be an object")
    return dict(value)


def _record_optional_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    return _positive(value, label)


def _parse_axis(
    axes: Mapping[str, object],
    axis: ArchitectureAxis,
) -> tuple[AxisStatus, object | None]:
    raw = axes.get(axis.value)
    if raw is None:
        return AxisStatus.UNKNOWN, None
    item = _record_object(raw, f"axes.{axis.value}")
    try:
        status = AxisStatus(_text(item.get("status"), f"axes.{axis.value}.status"))
    except ValueError as error:
        raise DeployableStateError(f"unknown axis status for {axis.value}") from error
    return status, item.get("value")


def deployable_state_from_record(raw: Mapping[str, object]) -> DeployableArchitectureState:
    """Import a state record and recompute its content-addressed ID."""

    value = migrate_deployable_state_record(raw)
    if value.get("schema_version") != DEPLOYABLE_STATE_SCHEMA_VERSION:
        raise DeployableStateError("unsupported deployable state schema version")
    axes = _record_object(value.get("axes"), "axes")
    parsed = {axis: _parse_axis(axes, axis) for axis in _axis_names()}
    predicted = tuple(
        axis for axis, (status, _) in parsed.items() if status is AxisStatus.PREDICTED
    )
    unknown = tuple(axis for axis, (status, _) in parsed.items() if status is AxisStatus.UNKNOWN)
    unsupported = tuple(
        axis for axis, (status, _) in parsed.items() if status is AxisStatus.UNSUPPORTED
    )
    layer_raw = parsed[ArchitectureAxis.LAYER_WIDTHS][1]
    layer_widths = (
        None
        if layer_raw is None
        else tuple(
            LayerWidth(
                _nonnegative(item.get("layer_index"), "layer index"),
                _positive(item.get("attention_width"), "attention width"),
                _positive(item.get("mlp_width"), "MLP width"),
            )
            for item in (
                _record_object(item, "layer width") for item in cast(list[object], layer_raw)
            )
        )
    )
    low_rank_raw = parsed[ArchitectureAxis.LOW_RANK_FACTORS][1]
    low_rank = (
        None
        if low_rank_raw is None
        else tuple(
            LowRankFactor(
                _text(item.get("component"), "low-rank component"),
                _positive(item.get("rank"), "low-rank factor"),
            )
            for item in (
                _record_object(item, "low-rank factor") for item in cast(list[object], low_rank_raw)
            )
        )
    )
    sparsity_raw = parsed[ArchitectureAxis.SPARSITY][1]
    sparsity = (
        None
        if sparsity_raw is None
        else tuple(
            SparsityEntry(
                _text(item.get("component"), "sparsity component"),
                _fraction(item.get("fraction"), "sparsity fraction"),
            )
            for item in (
                _record_object(item, "sparsity entry") for item in cast(list[object], sparsity_raw)
            )
        )
    )
    quant_raw = parsed[ArchitectureAxis.QUANTIZATION][1]
    quantization = (
        None
        if quant_raw is None
        else QuantizationState(
            _text(_record_object(quant_raw, "quantization").get("codec"), "quantization codec"),
            tuple(
                (
                    _text(item.get("component"), "quantization component"),
                    _text(item.get("codec"), "quantization codec"),
                )
                for item in (
                    _record_object(item, "tensor quantization")
                    for item in cast(
                        list[object],
                        _record_object(quant_raw, "quantization").get("tensor_codecs", []),
                    )
                )
            ),
        )
    )
    placement_raw = parsed[ArchitectureAxis.PLACEMENT][1]
    placement = (
        None
        if placement_raw is None
        else PlacementState(
            tuple(
                (
                    _text(item.get("component"), "placement component"),
                    _text(item.get("device"), "placement device"),
                )
                for item in (
                    _record_object(item, "placement assignment")
                    for item in cast(
                        list[object],
                        _record_object(placement_raw, "placement").get("assignments", []),
                    )
                )
            )
        )
    )
    artifact_raw = _record_object(value.get("artifact"), "artifact")
    artifact = ArtifactLineage(
        _digest(artifact_raw.get("source_artifact_digest"), "source artifact digest"),
        ArtifactContainerFormat(_text(artifact_raw.get("format"), "artifact format")),
        MaterializationStatus(_text(artifact_raw.get("status"), "artifact status")),
        None
        if artifact_raw.get("materialized_artifact_digest") is None
        else _digest(
            artifact_raw.get("materialized_artifact_digest"), "materialized artifact digest"
        ),
        None
        if artifact_raw.get("size_bytes") is None
        else _positive(artifact_raw.get("size_bytes"), "artifact size"),
        None
        if artifact_raw.get("physical_outcome_id") is None
        else _text(artifact_raw.get("physical_outcome_id"), "physical outcome ID"),
        None
        if artifact_raw.get("reason") is None
        else _text(artifact_raw.get("reason"), "artifact reason"),
    )
    mutation_order = value.get("mutation_order", [])
    if not isinstance(mutation_order, list) or not all(
        isinstance(item, str) for item in mutation_order
    ):
        raise DeployableStateError("mutation_order must be a string array")
    state = DeployableArchitectureState(
        _text(value.get("model_family"), "model family"),
        _text(value.get("model_revision"), "model revision"),
        _record_optional_int(parsed[ArchitectureAxis.DEPTH][1], "depth"),
        layer_widths,
        _record_optional_int(parsed[ArchitectureAxis.QUERY_HEADS][1], "query heads"),
        _record_optional_int(parsed[ArchitectureAxis.KV_HEADS][1], "KV heads"),
        _record_optional_int(parsed[ArchitectureAxis.HIDDEN_SIZE][1], "hidden size"),
        _record_optional_int(parsed[ArchitectureAxis.EMBEDDING_SIZE][1], "embedding size"),
        low_rank,
        sparsity,
        quantization,
        placement,
        artifact,
        tuple(mutation_order),
        predicted,
        unknown,
        unsupported,
    )
    if value.get("state_id") != state.state_id:
        raise DeployableStateError("state ID does not match its canonical content")
    return state

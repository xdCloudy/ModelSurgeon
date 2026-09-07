"""Source-fitted, architecture-normalized features for cross-model transfer."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

META_FEATURE_SCHEMA_VERSION: Final[int] = 1
META_FEATURE_PROTOCOL_REVISION: Final[str] = "meta-features-v1"


class MetaFeatureError(ValueError):
    """Raised when a cross-model feature contract cannot be satisfied."""


class MetaFeatureStatus(StrEnum):
    MEASURED = "measured"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class ArchitectureKind(StrEnum):
    DENSE = "dense"
    MOE = "moe"
    UNKNOWN = "unknown"


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetaFeatureError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MetaFeatureError(f"{label} must be finite")
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetaFeatureError(f"{label} is required")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class MetaFeatureSample:
    """Raw architecture/state context before source normalization."""

    sample_id: str
    model_revision: str
    partition: str
    structural_role: str
    architecture_family: str
    architecture_kind: ArchitectureKind
    layer_index: int | None
    layer_count: int | None
    width: int | None
    hidden_size: int | None
    head_count: int | None
    kv_head_count: int | None
    expert_count: int | None
    parameter_count: int | None
    quantization: str | None
    hardware_profile: str | None
    mutation_history: tuple[str, ...] = ()
    state_values: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        for label, text_value in (
            ("sample ID", self.sample_id),
            ("model revision", self.model_revision),
            ("partition", self.partition),
            ("structural role", self.structural_role),
            ("architecture family", self.architecture_family),
        ):
            _text(text_value, label)
        for label, numeric_value in (
            ("layer index", self.layer_index),
            ("layer count", self.layer_count),
            ("width", self.width),
            ("hidden size", self.hidden_size),
            ("head count", self.head_count),
            ("KV head count", self.kv_head_count),
            ("expert count", self.expert_count),
            ("parameter count", self.parameter_count),
        ):
            if numeric_value is not None and (
                isinstance(numeric_value, bool)
                or not isinstance(numeric_value, int)
                or numeric_value <= 0
            ):
                raise MetaFeatureError(f"{label} must be a positive integer when present")
        if (
            self.layer_index is not None
            and self.layer_count is not None
            and self.layer_index >= self.layer_count
        ):
            raise MetaFeatureError("layer index must be below layer count")
        for label, optional_text in (
            ("quantization", self.quantization),
            ("hardware profile", self.hardware_profile),
        ):
            if optional_text is not None:
                _text(optional_text, label)
        if len(self.mutation_history) != len(set(self.mutation_history)):
            raise MetaFeatureError("mutation history entries must be unique")
        if any(not item.strip() for item in self.mutation_history):
            raise MetaFeatureError("mutation history entries cannot be blank")
        names = tuple(name for name, _ in self.state_values)
        if len(names) != len(set(names)) or any(not name.strip() for name in names):
            raise MetaFeatureError("state feature names must be unique and non-empty")
        for name, state_value in self.state_values:
            _text(name, "state feature name")
            _finite(state_value, f"state feature {name}")

    def to_record(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "model_revision": self.model_revision,
            "partition": self.partition,
            "structural_role": self.structural_role,
            "architecture_family": self.architecture_family,
            "architecture_kind": self.architecture_kind.value,
            "layer_index": self.layer_index,
            "layer_count": self.layer_count,
            "width": self.width,
            "hidden_size": self.hidden_size,
            "head_count": self.head_count,
            "kv_head_count": self.kv_head_count,
            "expert_count": self.expert_count,
            "parameter_count": self.parameter_count,
            "quantization": self.quantization,
            "hardware_profile": self.hardware_profile,
            "mutation_history": list(self.mutation_history),
            "state_values": {name: value for name, value in self.state_values},
        }


@dataclass(frozen=True, slots=True)
class MetaFeatureConfig:
    max_history: int = 16
    numeric_names: tuple[str, ...] = (
        "relative_layer_depth",
        "width_to_hidden_ratio",
        "heads_to_kv_heads_ratio",
        "log_parameter_count",
        "state_value_mean",
    )
    categorical_names: tuple[str, ...] = (
        "architecture_family",
        "structural_role",
        "quantization",
        "hardware_profile",
    )
    allow_moe: bool = False
    unknown_category: str = "__unknown__"

    def __post_init__(self) -> None:
        if self.max_history <= 0 or not self.numeric_names or not self.categorical_names:
            raise MetaFeatureError("meta feature bounds and names are required")
        names = (*self.numeric_names, *self.categorical_names)
        if any(not name.strip() for name in names) or len(names) != len(set(names)):
            raise MetaFeatureError("meta feature names must be unique and non-empty")
        _text(self.unknown_category, "unknown category")

    def to_record(self) -> dict[str, object]:
        return {
            "max_history": self.max_history,
            "numeric_names": list(self.numeric_names),
            "categorical_names": list(self.categorical_names),
            "allow_moe": self.allow_moe,
            "unknown_category": self.unknown_category,
        }


@dataclass(frozen=True, slots=True)
class NormalizedMetaFeatures:
    sample_id: str
    numeric_values: tuple[float, ...]
    numeric_mask: tuple[bool, ...]
    categorical_indices: tuple[int, ...]
    categorical_known: tuple[bool, ...]
    categorical_values: tuple[str, ...]
    history_indices: tuple[int, ...]
    field_status: tuple[tuple[str, MetaFeatureStatus], ...]
    schema_id: str
    schema_version: int = META_FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.sample_id or self.schema_version != META_FEATURE_SCHEMA_VERSION:
            raise MetaFeatureError("normalized feature identity/schema is invalid")
        if len(self.numeric_values) != len(self.numeric_mask):
            raise MetaFeatureError("numeric values and masks must align")
        if len(self.categorical_indices) != len(self.categorical_known) or len(
            self.categorical_indices
        ) != len(self.categorical_values):
            raise MetaFeatureError("categorical values and masks must align")
        if any(not math.isfinite(value) for value in self.numeric_values):
            raise MetaFeatureError("normalized numeric values must be finite")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_id": self.sample_id,
            "numeric_values": list(self.numeric_values),
            "numeric_mask": list(self.numeric_mask),
            "categorical_indices": list(self.categorical_indices),
            "categorical_known": list(self.categorical_known),
            "categorical_values": list(self.categorical_values),
            "history_indices": list(self.history_indices),
            "field_status": {name: status.value for name, status in self.field_status},
            "schema_id": self.schema_id,
        }


@dataclass(frozen=True, slots=True)
class MetaFeatureCoverageReport:
    total_samples: int
    measured_fields: int
    unknown_fields: int
    unsupported_fields: int
    missingness: tuple[tuple[str, float], ...]
    category_collisions: int

    def to_record(self) -> dict[str, object]:
        return {
            "total_samples": self.total_samples,
            "measured_fields": self.measured_fields,
            "unknown_fields": self.unknown_fields,
            "unsupported_fields": self.unsupported_fields,
            "missingness": dict(self.missingness),
            "category_collisions": self.category_collisions,
        }


@dataclass(frozen=True, slots=True)
class MetaFeatureCompatibility:
    source_family: str
    target_family: str
    comparable_fields: tuple[str, ...]
    unknown_fields: tuple[str, ...]
    unsupported_fields: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "source_family": self.source_family,
            "target_family": self.target_family,
            "comparable_fields": list(self.comparable_fields),
            "unknown_fields": list(self.unknown_fields),
            "unsupported_fields": list(self.unsupported_fields),
        }


@dataclass(frozen=True, slots=True)
class MetaFeatureNormalizer:
    config: MetaFeatureConfig
    numeric_means: tuple[float, ...]
    numeric_scales: tuple[float, ...]
    categorical_vocabularies: tuple[tuple[str, ...], ...]
    history_vocabulary: tuple[str, ...]
    source_sample_ids: tuple[str, ...]
    source_model_revisions: tuple[str, ...]
    schema_id: str
    schema_version: int = META_FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != META_FEATURE_SCHEMA_VERSION:
            raise MetaFeatureError("unsupported meta feature schema version")
        if len(self.numeric_means) != len(self.config.numeric_names) or len(
            self.numeric_scales
        ) != len(self.config.numeric_names):
            raise MetaFeatureError("numeric normalization does not match schema")
        if len(self.categorical_vocabularies) != len(self.config.categorical_names):
            raise MetaFeatureError("categorical vocabularies do not match schema")
        if any(scale <= 0 or not math.isfinite(scale) for scale in self.numeric_scales):
            raise MetaFeatureError("numeric normalization scales must be positive")

    def transform(self, samples: Sequence[MetaFeatureSample]) -> tuple[NormalizedMetaFeatures, ...]:
        ids = tuple(sample.sample_id for sample in samples)
        if len(ids) != len(set(ids)):
            raise MetaFeatureError("transformed sample IDs must be unique")
        return tuple(self._transform(sample) for sample in samples)

    def _transform(self, sample: MetaFeatureSample) -> NormalizedMetaFeatures:
        raw_values, numeric_statuses = _raw_numeric(sample, self.config)
        numeric_values = tuple(
            (value - mean) / scale if status is MetaFeatureStatus.MEASURED else 0.0
            for value, status, mean, scale in zip(
                raw_values,
                numeric_statuses,
                self.numeric_means,
                self.numeric_scales,
                strict=True,
            )
        )
        numeric_mask = tuple(status is MetaFeatureStatus.MEASURED for status in numeric_statuses)
        category_values = _categorical_values(sample, self.config)
        category_indices: list[int] = []
        category_known: list[bool] = []
        category_statuses: list[MetaFeatureStatus] = []
        for value, vocabulary in zip(category_values, self.categorical_vocabularies, strict=True):
            known = value in vocabulary
            category_indices.append(vocabulary.index(value) if known else 0)
            category_known.append(known)
            category_statuses.append(
                MetaFeatureStatus.MEASURED if known else MetaFeatureStatus.UNKNOWN
            )
        history_indices = tuple(
            self.history_vocabulary.index(item) if item in self.history_vocabulary else 0
            for item in sample.mutation_history[: self.config.max_history]
        )
        field_status = tuple(
            [
                *zip(self.config.numeric_names, numeric_statuses, strict=True),
                *zip(self.config.categorical_names, category_statuses, strict=True),
            ]
        )
        return NormalizedMetaFeatures(
            sample.sample_id,
            numeric_values,
            numeric_mask,
            tuple(category_indices),
            tuple(category_known),
            category_values,
            history_indices,
            field_status,
            self.schema_id,
        )

    def coverage(self, samples: Sequence[MetaFeatureSample]) -> MetaFeatureCoverageReport:
        normalized = self.transform(samples)
        statuses = [status for item in normalized for _, status in item.field_status]
        missingness = tuple(
            (
                name,
                sum(
                    dict(item.field_status).get(name) is not MetaFeatureStatus.MEASURED
                    for item in normalized
                )
                / max(1, len(normalized)),
            )
            for name in (*self.config.numeric_names, *self.config.categorical_names)
        )
        return MetaFeatureCoverageReport(
            len(normalized),
            sum(status is MetaFeatureStatus.MEASURED for status in statuses),
            sum(status is MetaFeatureStatus.UNKNOWN for status in statuses),
            sum(status is MetaFeatureStatus.UNSUPPORTED for status in statuses),
            missingness,
            sum(not known for item in normalized for known in item.categorical_known),
        )

    def compatibility(
        self, source_family: str, target: MetaFeatureSample
    ) -> MetaFeatureCompatibility:
        _text(source_family, "source family")
        statuses = dict(self._transform(target).field_status)
        return MetaFeatureCompatibility(
            source_family,
            target.architecture_family,
            tuple(
                name for name, status in statuses.items() if status is MetaFeatureStatus.MEASURED
            ),
            tuple(name for name, status in statuses.items() if status is MetaFeatureStatus.UNKNOWN),
            tuple(
                name for name, status in statuses.items() if status is MetaFeatureStatus.UNSUPPORTED
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": META_FEATURE_PROTOCOL_REVISION,
            "config": self.config.to_record(),
            "numeric_means": list(self.numeric_means),
            "numeric_scales": list(self.numeric_scales),
            "categorical_vocabularies": [list(item) for item in self.categorical_vocabularies],
            "history_vocabulary": list(self.history_vocabulary),
            "source_sample_ids": list(self.source_sample_ids),
            "source_model_revisions": list(self.source_model_revisions),
            "schema_id": self.schema_id,
        }


def _raw_numeric(
    sample: MetaFeatureSample, config: MetaFeatureConfig
) -> tuple[tuple[float, ...], tuple[MetaFeatureStatus, ...]]:
    state_mean = (
        math.fsum(value for _, value in sample.state_values) / len(sample.state_values)
        if sample.state_values
        else None
    )
    layer_depth = (
        sample.layer_index / max(1, sample.layer_count - 1)
        if sample.layer_index is not None and sample.layer_count is not None
        else None
    )
    width_ratio = (
        sample.width / sample.hidden_size
        if sample.width is not None and sample.hidden_size is not None
        else None
    )
    head_ratio = (
        sample.head_count / sample.kv_head_count
        if sample.head_count is not None and sample.kv_head_count is not None
        else None
    )
    parameter_log = (
        math.log1p(sample.parameter_count) if sample.parameter_count is not None else None
    )
    raw_by_name: dict[str, float | None] = {
        "relative_layer_depth": layer_depth,
        "width_to_hidden_ratio": width_ratio,
        "heads_to_kv_heads_ratio": head_ratio,
        "log_parameter_count": parameter_log,
        "state_value_mean": state_mean,
    }
    values: list[float] = []
    statuses: list[MetaFeatureStatus] = []
    for name in config.numeric_names:
        value = raw_by_name.get(name)
        if sample.architecture_kind is ArchitectureKind.MOE and not config.allow_moe:
            values.append(0.0)
            statuses.append(MetaFeatureStatus.UNSUPPORTED)
        elif value is None:
            values.append(0.0)
            statuses.append(MetaFeatureStatus.UNKNOWN)
        else:
            values.append(_finite(value, name))
            statuses.append(MetaFeatureStatus.MEASURED)
    return tuple(values), tuple(statuses)


def _categorical_values(sample: MetaFeatureSample, config: MetaFeatureConfig) -> tuple[str, ...]:
    values = {
        "architecture_family": sample.architecture_family,
        "structural_role": sample.structural_role,
        "quantization": sample.quantization or config.unknown_category,
        "hardware_profile": sample.hardware_profile or config.unknown_category,
    }
    return tuple(values.get(name, config.unknown_category) for name in config.categorical_names)


def _categorical_vocabularies(
    samples: Sequence[MetaFeatureSample], config: MetaFeatureConfig
) -> tuple[tuple[str, ...], ...]:
    return tuple(
        (
            config.unknown_category,
            *sorted(
                {
                    _categorical_values(sample, config)[index]
                    for sample in samples
                    if _categorical_values(sample, config)[index] != config.unknown_category
                }
            ),
        )
        for index in range(len(config.categorical_names))
    )


def fit_meta_feature_normalizer(
    source_samples: Sequence[MetaFeatureSample], config: MetaFeatureConfig | None = None
) -> MetaFeatureNormalizer:
    """Fit numeric statistics and vocabularies from source samples only."""

    resolved = config or MetaFeatureConfig()
    if not source_samples:
        raise MetaFeatureError("meta feature fitting requires source samples")
    if any(sample.partition != "source" for sample in source_samples):
        raise MetaFeatureError("normalization must be fitted from source-partition samples only")
    ids = tuple(sample.sample_id for sample in source_samples)
    if len(ids) != len(set(ids)):
        raise MetaFeatureError("source sample IDs must be unique")
    raw_rows = [_raw_numeric(sample, resolved) for sample in source_samples]
    means: list[float] = []
    scales: list[float] = []
    for index in range(len(resolved.numeric_names)):
        measured = [
            row[0][index] for row in raw_rows if row[1][index] is MetaFeatureStatus.MEASURED
        ]
        if not measured:
            means.append(0.0)
            scales.append(1.0)
            continue
        mean = math.fsum(measured) / len(measured)
        scale = max(
            math.sqrt(math.fsum((value - mean) ** 2 for value in measured) / len(measured)),
            1e-9,
        )
        means.append(mean)
        scales.append(scale)
    categorical = _categorical_vocabularies(source_samples, resolved)
    history = tuple(
        sorted(
            {
                item
                for sample in source_samples
                for item in sample.mutation_history[: resolved.max_history]
            }
        )
    )
    schema_identity = {
        "config": resolved.to_record(),
        "numeric_means": means,
        "numeric_scales": scales,
        "categorical_vocabularies": categorical,
        "history_vocabulary": history,
    }
    return MetaFeatureNormalizer(
        resolved,
        tuple(means),
        tuple(scales),
        tuple(categorical),
        history,
        tuple(sorted(ids)),
        tuple(sorted({sample.model_revision for sample in source_samples})),
        "sha256:" + hashlib.sha256(_canonical(schema_identity).encode("utf-8")).hexdigest(),
    )

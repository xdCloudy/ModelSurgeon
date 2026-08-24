"""Validated, immutable, and canonically serializable application configuration."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from modelsurgeon.adapters import ModelFormat


class StrictConfigModel(BaseModel):
    """Immutable configuration section that rejects unknown keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ComputeDType(StrEnum):
    AUTO = "auto"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class MemoryMode(StrEnum):
    AUTO = "auto"
    FULL = "full"
    TENSOR = "tensor"
    STREAMING = "streaming"


class OptimizeMetric(StrEnum):
    QUALITY = "quality"
    PERPLEXITY = "perplexity"
    PARAMETER_COUNT = "parameter_count"
    LATENCY = "latency"
    MEMORY = "memory"
    DISK_SIZE = "disk_size"


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ObjectiveNormalization(StrEnum):
    IDENTITY = "identity"
    BASELINE_RATIO = "baseline_ratio"
    MIN_MAX = "min_max"


class ObjectiveTermConfig(StrictConfigModel):
    """One configurable soft objective and its score normalization."""

    metric: OptimizeMetric
    direction: ObjectiveDirection
    weight: float = Field(default=1.0, gt=0.0)
    normalization: ObjectiveNormalization = ObjectiveNormalization.BASELINE_RATIO
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_normalization_bounds(self) -> ObjectiveTermConfig:
        has_bounds = self.minimum is not None or self.maximum is not None
        if self.normalization is ObjectiveNormalization.MIN_MAX:
            if self.minimum is None or self.maximum is None or self.minimum >= self.maximum:
                raise ValueError("min-max normalization requires ordered finite bounds")
        elif has_bounds:
            raise ValueError("normalization bounds are only valid for min-max")
        return self


class ModelConfig(StrictConfigModel):
    """Target model identity and loading preferences."""

    path: str | None = None
    revision: str | None = None
    format: ModelFormat = ModelFormat.HUGGING_FACE
    dtype: ComputeDType = ComputeDType.AUTO

    @field_validator("path", "revision")
    @classmethod
    def reject_blank_optional_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value cannot be blank")
        return value


class CalibrationConfig(StrictConfigModel):
    """Deterministic calibration sample and tokenization limits."""

    dataset: str | None = None
    dataset_revision: str | None = None
    split: str = "train"
    samples: int = Field(default=512, gt=0)
    batch_size: int = Field(default=1, gt=0)
    max_sequence_length: int = Field(default=2048, gt=0)
    seed: int = Field(default=0, ge=0)

    @field_validator("dataset", "dataset_revision")
    @classmethod
    def reject_blank_dataset_values(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("value cannot be blank")
        return value

    @field_validator("split")
    @classmethod
    def reject_blank_split(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("split cannot be blank")
        return value


class FeatureConfig(StrictConfigModel):
    """Feature extractor groups enabled for a run."""

    weights: bool = True
    spectral: bool = True
    activations: bool = True
    gradients: bool = False
    correlations: bool = False
    topology: bool = True
    runtime: bool = True


class ObjectiveConfig(StrictConfigModel):
    """Hard quality/resource constraints and optimization dimensions."""

    quality_retention: float = Field(default=0.98, ge=0.0, le=1.0)
    max_perplexity_increase: float | None = Field(default=None, ge=0.0)
    min_latency_improvement: float | None = Field(default=None, ge=0.0)
    max_vram_gb: float | None = Field(default=None, gt=0.0)
    optimize: tuple[OptimizeMetric, ...] = (
        OptimizeMetric.PARAMETER_COUNT,
        OptimizeMetric.LATENCY,
    )
    terms: tuple[ObjectiveTermConfig, ...] | None = None

    @field_validator("optimize")
    @classmethod
    def reject_duplicate_objectives(
        cls,
        value: tuple[OptimizeMetric, ...],
    ) -> tuple[OptimizeMetric, ...]:
        if len(value) != len(set(value)):
            raise ValueError("optimization dimensions must be unique")
        if not value:
            raise ValueError("at least one optimization dimension is required")
        return value

    @field_validator("terms")
    @classmethod
    def reject_duplicate_objective_terms(
        cls,
        value: tuple[ObjectiveTermConfig, ...] | None,
    ) -> tuple[ObjectiveTermConfig, ...] | None:
        if value is not None:
            metrics = [term.metric for term in value]
            if not value or len(metrics) != len(set(metrics)):
                raise ValueError("objective terms must be non-empty and unique by metric")
        return value


class ConstraintConfig(StrictConfigModel):
    """Hard search constraints with explicit units and baseline semantics."""

    min_quality_retention_ratio: float = Field(default=0.98, ge=0.0, le=1.0)
    max_perplexity_delta: float | None = Field(default=None, ge=0.0)
    min_latency_gain_ratio: float | None = Field(default=None, ge=0.0)
    max_ram_bytes: int | None = Field(default=None, gt=0)
    max_vram_bytes: int | None = Field(default=None, gt=0)
    max_disk_bytes: int | None = Field(default=None, gt=0)


class HardwareConfig(StrictConfigModel):
    """Resource ceilings used by loaders and experiment workers."""

    memory_mode: MemoryMode = MemoryMode.AUTO
    max_vram_gb: float | None = Field(default=None, gt=0.0)
    max_ram_gb: float | None = Field(default=None, gt=0.0)
    cpu_offload: bool = True
    mixed_precision: bool = True


class SafetyConfig(StrictConfigModel):
    """Checkpoint and model-code safety controls."""

    allow_overwrite: bool = False
    trust_remote_code: bool = False
    require_atomic_writes: bool = True


class Settings(BaseSettings):
    """Top-level settings populated by MODELSURGEON_* environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MODELSURGEON_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1] = 1
    artifact_dir: Path = Path("artifacts")
    model: ModelConfig = Field(default_factory=ModelConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    constraints: ConstraintConfig = Field(default_factory=ConstraintConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    def canonical_dict(self) -> dict[str, object]:
        """Return the deterministic JSON-compatible configuration mapping."""
        return self.model_dump(mode="json", round_trip=True)

    def canonical_json(self) -> str:
        """Return stable UTF-8 JSON used for run identity and provenance."""
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

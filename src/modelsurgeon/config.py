"""Validated, immutable, and canonically serializable application configuration."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from modelsurgeon.adapters import ModelFormat
from modelsurgeon.provider_kind import ProviderKind


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

    cache_dir: Path | None = None
    weights: bool = True
    spectral: bool = True
    activations: bool = True
    gradients: bool = False
    correlations: bool = False
    topology: bool = True
    runtime: bool = True


class SurgeonConfig(StrictConfigModel):
    """Optional signed Meta-Surgeon bundle used for candidate guidance."""

    registry_root: Path | None = None
    card_digest: str | None = None
    signing_env: str | None = None

    @field_validator("card_digest", "signing_env")
    @classmethod
    def reject_blank_surgeon_values(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Meta-Surgeon values cannot be blank")
        return value

    @model_validator(mode="after")
    def require_complete_surgeon_identity(self) -> SurgeonConfig:
        values = (self.registry_root, self.card_digest, self.signing_env)
        if any(value is not None for value in values) and not all(
            value is not None for value in values
        ):
            raise ValueError(
                "surgeon.registry_root, surgeon.card_digest, and surgeon.signing_env "
                "must be supplied together"
            )
        return self


class SearchConfig(StrictConfigModel):
    """Physical candidate scopes supported by the first-party search path."""

    scopes: tuple[
        Literal["mlp_channel", "attention_head", "transformer_layer", "low_rank"], ...
    ] = (
        "mlp_channel",
    )
    max_surgery_steps: int | None = Field(default=None, gt=0)
    low_rank_rank: int = Field(default=4, ge=1, le=64)

    @field_validator("scopes")
    @classmethod
    def validate_scopes(
        cls,
        value: tuple[
            Literal["mlp_channel", "attention_head", "transformer_layer", "low_rank"], ...
        ],
    ) -> tuple[
        Literal["mlp_channel", "attention_head", "transformer_layer", "low_rank"], ...
    ]:
        if not value or len(value) != len(set(value)):
            raise ValueError("search scopes must be non-empty and unique")
        return value


class RepairConfig(StrictConfigModel):
    """Optional real post-surgery repair executed by a first-party runtime."""

    method: Literal["none", "lora", "distillation"] = "none"
    target_modules: tuple[str, ...] = ()
    max_steps: int | None = Field(default=None, gt=0)
    rank: int = Field(default=4, ge=1, le=64)
    alpha: float = Field(default=8.0, gt=0.0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1e-3, gt=0.0)

    @field_validator("target_modules")
    @classmethod
    def validate_target_modules(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != tuple(sorted(set(value))) or any(
            not item.strip() or any(not part for part in item.split("."))
            for item in value
        ):
            raise ValueError("repair.target_modules must be sorted, unique canonical paths")
        return value


class QuantizationConfig(StrictConfigModel):
    """Optional first-party quantization selected by the optimize runtime."""

    method: Literal["none", "dynamic_int8"] = "none"


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


class ProviderConfig(StrictConfigModel):
    """Optional conversational provider selection and hard request budgets.

    Provider configuration is deliberately separate from optimization policy.
    The direct CLI and Python APIs use ``none`` by default and never need a
    provider adapter to build or execute deterministic work.
    """

    kind: ProviderKind = ProviderKind.NONE
    provider_id: str = "none"
    model_id: str = "none"
    model_revision: str = "none"
    model_path: Path | None = None
    runtime_revision: str | None = None
    endpoint: str | None = None
    api_key_env: str | None = None
    request_timeout_seconds: float = Field(default=30.0, gt=0.0, le=3600.0)
    max_input_size: int = Field(default=8192, gt=0)
    max_output_size: int = Field(default=2048, gt=0)

    @field_validator("provider_id", "model_id", "model_revision")
    @classmethod
    def reject_blank_identity_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provider identity values cannot be blank")
        return value

    @field_validator("runtime_revision")
    @classmethod
    def reject_blank_runtime_revision(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("runtime_revision cannot be blank")
        return value

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("provider endpoint cannot be blank")
        if value is not None:
            from urllib.parse import urlsplit

            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("provider endpoint must be an absolute http(s) URL")
            if parsed.username or parsed.password:
                raise ValueError("provider endpoint cannot contain credentials")
            if parsed.query or parsed.fragment:
                raise ValueError("provider endpoint cannot contain query or fragment data")
        return value

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_env(cls, value: str | None) -> str | None:
        if value is not None:
            import re

            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", value) is None:
                raise ValueError("api_key_env must be an uppercase environment variable name")
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> ProviderConfig:
        if self.kind is ProviderKind.NONE:
            if (
                self.provider_id != "none"
                or self.model_id != "none"
                or self.model_revision != "none"
                or self.model_path is not None
                or self.runtime_revision is not None
                or self.endpoint is not None
                or self.api_key_env is not None
            ):
                raise ValueError(
                    "provider.kind='none' requires provider_id, model_id, and "
                    "model_revision to be 'none' and no endpoint or api_key_env"
                )
            return self

        for field_name in ("provider_id", "model_id", "model_revision"):
            if getattr(self, field_name) == "none":
                raise ValueError(
                    f"provider.{field_name} is required when provider.kind is {self.kind.value!r}"
                )
        if self.kind is ProviderKind.COMPATIBLE_ENDPOINT and self.endpoint is None:
            raise ValueError(
                "provider.endpoint is required for provider.kind='compatible_endpoint'"
            )
        return self


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
    surgeon: SurgeonConfig = Field(default_factory=SurgeonConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    constraints: ConstraintConfig = Field(default_factory=ConstraintConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
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

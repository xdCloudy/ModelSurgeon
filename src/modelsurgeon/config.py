"""Validated application configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HardwareConfig(BaseModel):
    """Resource ceilings used by loaders and experiment workers."""

    max_vram_gb: float | None = Field(default=None, gt=0)
    max_ram_gb: float | None = Field(default=None, gt=0)
    cpu_offload: bool = True
    mixed_precision: bool = True


class SafetyConfig(BaseModel):
    """Checkpoint and model-code safety controls."""

    allow_overwrite: bool = False
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def reject_unsafe_default_change(self) -> SafetyConfig:
        """Keep the validation hook explicit as more safety flags are added."""
        return self


class Settings(BaseSettings):
    """Top-level settings, optionally populated with MODELSURGEON_* variables."""

    model_config = SettingsConfigDict(
        env_prefix="MODELSURGEON_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    artifact_dir: Path = Path("artifacts")
    hardware: HardwareConfig = HardwareConfig()
    safety: SafetyConfig = SafetyConfig()
    log_level: str = "INFO"


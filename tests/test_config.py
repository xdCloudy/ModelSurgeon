import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelsurgeon.adapters import ModelFormat
from modelsurgeon.config import (
    HardwareConfig,
    MemoryMode,
    ObjectiveConfig,
    OptimizeMetric,
    Settings,
)


def test_safe_defaults() -> None:
    settings = Settings()

    assert settings.safety.allow_overwrite is False
    assert settings.safety.trust_remote_code is False
    assert settings.safety.require_atomic_writes is True
    assert settings.hardware.cpu_offload is True
    assert settings.model.format is ModelFormat.HUGGING_FACE
    assert settings.objective.quality_retention == 0.98


@pytest.mark.parametrize(
    "payload",
    [
        {"unknown": True},
        {"hardware": {"unknown": True}},
        {"calibration": {"samples": 0}},
        {"calibration": {"seed": -1}},
        {"objective": {"quality_retention": 1.01}},
        {"objective": {"max_perplexity_increase": -0.1}},
        {"hardware": {"max_vram_gb": 0}},
        {"model": {"path": "  "}},
    ],
)
def test_invalid_limits_and_unknown_keys_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(payload)


def test_duplicate_or_empty_optimization_dimensions_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        ObjectiveConfig(optimize=(OptimizeMetric.LATENCY, OptimizeMetric.LATENCY))
    with pytest.raises(ValidationError, match="at least one"):
        ObjectiveConfig(optimize=())


def test_environment_overrides_nested_hardware_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODELSURGEON_HARDWARE__MAX_VRAM_GB", "11")
    monkeypatch.setenv("MODELSURGEON_HARDWARE__MEMORY_MODE", "streaming")
    monkeypatch.setenv("MODELSURGEON_SAFETY__TRUST_REMOTE_CODE", "false")

    settings = Settings()

    assert settings.hardware.max_vram_gb == 11
    assert settings.hardware.memory_mode is MemoryMode.STREAMING
    assert settings.safety.trust_remote_code is False


def test_canonical_serialization_is_stable_and_json_compatible() -> None:
    first = Settings(
        artifact_dir=Path("runs/artifacts"),
        hardware=HardwareConfig(max_ram_gb=48, max_vram_gb=11),
    )
    second = Settings.model_validate(
        {
            "hardware": {"max_vram_gb": 11, "max_ram_gb": 48},
            "artifact_dir": "runs/artifacts",
        }
    )

    assert first.canonical_json() == second.canonical_json()
    assert json.loads(first.canonical_json()) == first.canonical_dict()
    assert " " not in first.canonical_json()


def test_configuration_sections_are_immutable() -> None:
    settings = Settings()

    with pytest.raises(ValidationError):
        settings.hardware.max_vram_gb = 12  # type: ignore[misc]


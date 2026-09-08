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
    ProviderConfig,
    QuantizationConfig,
    RepairConfig,
    RuntimeConfig,
    SearchConfig,
    Settings,
)
from modelsurgeon.provider_kind import ProviderKind


def test_safe_defaults() -> None:
    settings = Settings()

    assert settings.safety.allow_overwrite is False
    assert settings.safety.trust_remote_code is False
    assert settings.safety.require_atomic_writes is True
    assert settings.hardware.cpu_offload is True
    assert settings.model.format is ModelFormat.HUGGING_FACE
    assert settings.objective.quality_retention == 0.98
    assert settings.provider.kind is ProviderKind.NONE
    assert settings.provider.provider_id == "none"
    assert settings.search.scopes == ("mlp_channel",)


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
        {"provider": {"kind": "none", "provider_id": "hosted"}},
        {"provider": {"kind": "compatible_endpoint", "provider_id": "p"}},
        {"provider": {"kind": "local", "api_key_env": "not-uppercase"}},
        {
            "provider": {
                "kind": "hosted",
                "provider_id": "p",
                "model_id": "m",
                "model_revision": "r",
                "endpoint": "https://user:pass@example.test",
            }
        },
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


def test_search_scopes_are_non_empty_and_unique() -> None:
    assert SearchConfig(scopes=("attention_head", "transformer_layer")).scopes == (
        "attention_head",
        "transformer_layer",
    )
    with pytest.raises(ValidationError, match="unique"):
        SearchConfig(scopes=("attention_head", "attention_head"))
    with pytest.raises(ValidationError, match="non-empty"):
        SearchConfig(scopes=())


def test_repair_config_is_explicit_and_canonical() -> None:
    assert Settings().repair.method == "none"
    configured = RepairConfig(
        method="lora",
        target_modules=("model.layers.0.mlp.down_proj",),
        max_steps=2,
    )
    assert configured.method == "lora"
    assert RepairConfig(method="distillation").method == "distillation"
    with pytest.raises(ValidationError, match="sorted, unique"):
        RepairConfig(method="lora", target_modules=("b", "a"))


def test_quantization_config_is_explicit_and_canonical() -> None:
    assert Settings().quantization.method == "none"
    assert QuantizationConfig(method="dynamic_int8").method == "dynamic_int8"


def test_native_runtime_requires_ordered_batch_geometry() -> None:
    configured = RuntimeConfig(context_size=1024, batch_size=512, microbatch_size=256)
    assert configured.context_size == 1024
    with pytest.raises(ValidationError, match="cannot exceed"):
        RuntimeConfig(context_size=128, batch_size=256)


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


def test_provider_config_requires_complete_non_null_identity() -> None:
    with pytest.raises(ValidationError, match=r"provider\.model_id"):
        ProviderConfig(
            kind=ProviderKind.LOCAL,
            provider_id="local",
            model_revision="revision-1",
        )

    configured = ProviderConfig(
        kind=ProviderKind.COMPATIBLE_ENDPOINT,
        provider_id="endpoint",
        model_id="model",
        model_revision="revision-1",
        endpoint="https://provider.example/v1",
        api_key_env="MODELSURGEON_PROVIDER_KEY",
    )
    assert configured.endpoint == "https://provider.example/v1"


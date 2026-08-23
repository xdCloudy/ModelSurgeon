import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from modelsurgeon.config import MemoryMode
from modelsurgeon.config_io import (
    ConfigurationFileError,
    dump_resolved_settings,
    environment_overrides,
    expand_dotted_overrides,
    load_config_file,
    load_settings,
)


@pytest.mark.parametrize("extension", ["yaml", "yml"])
def test_load_yaml_settings(extension: str, tmp_path: Path) -> None:
    path = tmp_path / f"config.{extension}"
    path.write_text(
        "hardware:\n  max_vram_gb: 11\n  memory_mode: tensor\nfeatures:\n  gradients: true\n",
        encoding="utf-8",
    )

    settings = load_settings(path, environ={})

    assert settings.hardware.max_vram_gb == 11
    assert settings.hardware.memory_mode is MemoryMode.TENSOR
    assert settings.features.gradients is True


def test_load_toml_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[calibration]\nsamples = 64\nbatch_size = 2\n[objective]\nquality_retention = 0.99\n",
        encoding="utf-8",
    )

    settings = load_settings(path, environ={})

    assert settings.calibration.samples == 64
    assert settings.calibration.batch_size == 2
    assert settings.objective.quality_retention == 0.99


def test_precedence_is_defaults_file_environment_then_cli(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("calibration:\n  samples: 100\n", encoding="utf-8")

    settings = load_settings(
        path,
        environ={"MODELSURGEON_CALIBRATION__SAMPLES": "200"},
        cli_overrides={"calibration.samples": 300},
    )

    assert settings.calibration.samples == 300
    assert settings.calibration.batch_size == 1


def test_nested_cli_overrides_merge_without_erasing_siblings() -> None:
    expanded = expand_dotted_overrides(
        {
            "hardware.max_ram_gb": 48,
            "hardware.max_vram_gb": 11,
            "features.gradients": True,
        }
    )

    assert expanded == {
        "hardware": {"max_ram_gb": 48, "max_vram_gb": 11},
        "features": {"gradients": True},
    }


def test_environment_values_are_parsed_without_touching_unrelated_variables() -> None:
    overrides = environment_overrides(
        {
            "PATH": "ignored",
            "MODELSURGEON_HARDWARE__CPU_OFFLOAD": "false",
            "MODELSURGEON_CALIBRATION__SAMPLES": "32",
            "MODELSURGEON_MODEL__PATH": "org/model",
        }
    )

    assert overrides == {
        "calibration": {"samples": 32},
        "hardware": {"cpu_offload": False},
        "model": {"path": "org/model"},
    }


def test_resolved_output_is_canonical_json_without_secret_fields() -> None:
    settings = load_settings(
        environ={},
        cli_overrides={"model.path": "org/model", "hardware.max_vram_gb": 11},
    )

    resolved = dump_resolved_settings(settings)
    payload = json.loads(resolved)

    assert resolved.endswith("\n")
    assert payload == settings.canonical_dict()
    assert "password" not in resolved.lower()
    assert "token" not in resolved.lower()
    assert "secret" not in resolved.lower()


@pytest.mark.parametrize("name", ["config.json", "config.txt", "config"])
def test_unsupported_extensions_fail(name: str, tmp_path: Path) -> None:
    path = tmp_path / name
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ConfigurationFileError, match="unsupported"):
        load_config_file(path)


def test_non_mapping_root_and_unknown_settings_fail(tmp_path: Path) -> None:
    list_path = tmp_path / "list.yaml"
    list_path.write_text("- one\n- two\n", encoding="utf-8")
    unknown_path = tmp_path / "unknown.toml"
    unknown_path.write_text("unknown = true\n", encoding="utf-8")

    with pytest.raises(ConfigurationFileError, match="mapping"):
        load_settings(list_path, environ={})
    with pytest.raises(ValidationError, match="unknown"):
        load_settings(unknown_path, environ={})


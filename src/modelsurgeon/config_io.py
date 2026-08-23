"""Configuration file loading, precedence, and resolved output."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path

from modelsurgeon.config import Settings

_ENV_PREFIX = "MODELSURGEON_"
_ENV_DELIMITER = "__"


class ConfigurationFileError(ValueError):
    """Raised when a configuration source cannot be parsed as a settings mapping."""


def _mapping(value: object, *, source: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationFileError(f"{source} must contain a mapping at its root")
    if not all(isinstance(key, str) for key in value):
        raise ConfigurationFileError(f"{source} keys must be strings")
    return {str(key): item for key, item in value.items()}


def load_config_file(path: Path) -> dict[str, object]:
    """Load one UTF-8 YAML or TOML file into an unvalidated mapping."""
    if not path.is_file():
        raise ConfigurationFileError(f"configuration file does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationFileError(f"configuration file is not valid UTF-8: {path}") from exc

    try:
        if path.suffix.lower() == ".toml":
            value: object = tomllib.loads(text)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            yaml = import_module("yaml")
            value = yaml.safe_load(text)
        else:
            raise ConfigurationFileError(
                f"unsupported configuration extension {path.suffix!r}; "
                "expected .toml, .yaml, or .yml"
            )
    except ConfigurationFileError:
        raise
    except Exception as exc:
        raise ConfigurationFileError(f"failed to parse configuration file {path}: {exc}") from exc
    return _mapping(value, source=str(path))


def _merge(base: Mapping[str, object], override: Mapping[str, object]) -> dict[str, object]:
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _merge(_mapping(existing, source=key), _mapping(value, source=key))
        else:
            result[key] = value
    return result


def _parse_environment_value(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def environment_overrides(environ: Mapping[str, str]) -> dict[str, object]:
    """Convert MODELSURGEON_* variables into a nested settings mapping."""
    result: dict[str, object] = {}
    for name, raw_value in sorted(environ.items()):
        if not name.upper().startswith(_ENV_PREFIX):
            continue
        suffix = name[len(_ENV_PREFIX) :]
        parts = [part.lower() for part in suffix.split(_ENV_DELIMITER)]
        if not all(parts):
            raise ConfigurationFileError(f"invalid environment setting name: {name}")
        cursor = result
        for part in parts[:-1]:
            existing = cursor.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ConfigurationFileError(
                    f"environment setting {name} conflicts with a scalar parent"
                )
            cursor = existing
        final = parts[-1]
        if final in cursor:
            raise ConfigurationFileError(f"duplicate environment setting: {name}")
        cursor[final] = _parse_environment_value(raw_value)
    return result


def expand_dotted_overrides(overrides: Mapping[str, object]) -> dict[str, object]:
    """Expand CLI-style dotted keys while retaining nested mapping support."""
    result: dict[str, object] = {}
    for key, value in overrides.items():
        if not isinstance(key, str) or not key:
            raise ConfigurationFileError("CLI override keys must be non-empty strings")
        parts = key.split(".")
        if not all(parts):
            raise ConfigurationFileError(f"invalid dotted CLI override: {key!r}")
        nested: dict[str, object] = {}
        cursor = nested
        for part in parts[:-1]:
            child: dict[str, object] = {}
            cursor[part] = child
            cursor = child
        cursor[parts[-1]] = value
        result = _merge(result, nested)
    return result


def load_settings(
    path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, object] | None = None,
) -> Settings:
    """Resolve defaults < file < environment < CLI into validated settings."""
    file_values = {} if path is None else load_config_file(path)
    environment_values = environment_overrides(os.environ if environ is None else environ)
    cli_values = expand_dotted_overrides({} if cli_overrides is None else cli_overrides)
    merged = _merge(_merge(file_values, environment_values), cli_values)
    return Settings.model_validate(merged)


def dump_resolved_settings(settings: Settings) -> str:
    """Emit the canonical, secret-free resolved configuration as JSON text."""
    return f"{settings.canonical_json()}\n"

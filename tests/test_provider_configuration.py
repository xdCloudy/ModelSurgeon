from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from modelsurgeon.cli.app import app
from modelsurgeon.config import ProviderConfig, Settings
from modelsurgeon.conversation import NullTextModelProvider
from modelsurgeon.provider_kind import ProviderKind
from modelsurgeon.providers import (
    ProviderConfigurationError,
    ProviderDiagnosticStatus,
    provider_diagnostics,
    resolve_text_model_provider,
)


def _configured_provider() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.COMPATIBLE_ENDPOINT,
        provider_id="endpoint",
        model_id="model",
        model_revision="revision-1",
        endpoint="https://provider.example/v1",
        api_key_env="MODELSURGEON_PROVIDER_KEY",
    )


def test_no_llm_resolves_only_the_explicit_null_provider() -> None:
    provider = resolve_text_model_provider(Settings())

    assert isinstance(provider, NullTextModelProvider)
    diagnostic = provider_diagnostics(Settings())
    assert diagnostic.status is ProviderDiagnosticStatus.DISABLED
    assert diagnostic.code == "no_llm"


def test_missing_provider_key_is_a_stable_redacted_diagnostic() -> None:
    diagnostic = provider_diagnostics(_configured_provider(), environ={})

    assert diagnostic.status is ProviderDiagnosticStatus.UNAVAILABLE
    assert diagnostic.code == "missing_api_key"
    assert "MODELSURGEON_PROVIDER_KEY" in diagnostic.message
    assert "secret" not in diagnostic.message.lower()


def test_uninstalled_adapter_fails_closed_with_actionable_error() -> None:
    config = _configured_provider().model_copy(update={"api_key_env": None})
    diagnostic = provider_diagnostics(config)
    assert diagnostic.code == "adapter_unavailable"
    assert diagnostic.dependency == "modelsurgeon-provider-compatible_endpoint"

    with pytest.raises(ProviderConfigurationError, match="adapter_unavailable") as error:
        resolve_text_model_provider(config)
    assert error.value.code == "adapter_unavailable"


def test_provider_diagnostics_cli_is_machine_readable() -> None:
    result = CliRunner().invoke(app, ["provider", "diagnostics", "--no-llm", "--json"], color=False)

    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    assert record["record_type"] == "provider_diagnostic"
    assert record["status"] == "disabled"
    assert record["code"] == "no_llm"


def test_provider_diagnostics_cli_reports_invalid_config_stably() -> None:
    result = CliRunner().invoke(
        app,
        ["provider", "diagnostics", "--provider", "compatible_endpoint", "--json"],
        color=False,
    )

    assert result.exit_code == 2
    record = json.loads(result.output)
    assert record["record_type"] == "error"
    assert record["category"] == "provider_configuration"
    assert record["code"] == "invalid_configuration"

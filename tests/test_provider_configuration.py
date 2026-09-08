from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from modelsurgeon.cli.app import app
from modelsurgeon.config import ProviderConfig, Settings
from modelsurgeon.config_io import discover_settings
from modelsurgeon.conversation import (
    NullTextModelProvider,
    ProviderCapability,
    ProviderCapabilityCard,
    ProviderLimits,
    ProviderModelIdentity,
)
from modelsurgeon.provider_kind import ProviderKind
from modelsurgeon.providers import (
    ProviderCapabilityState,
    ProviderConfigurationError,
    ProviderDiagnosticStatus,
    provider_diagnostics,
    resolve_text_model_provider,
    summarize_provider_capabilities,
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


def test_configuration_discovery_is_deterministic_and_value_free(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text(
        '[provider]\nkind = "compatible_endpoint"\nprovider_id = "endpoint"\n'
        'model_id = "model"\nmodel_revision = "revision-1"\n'
        'endpoint = "https://provider.example/v1"\n',
        encoding="utf-8",
    )
    first = discover_settings(
        path,
        environ={"MODELSURGEON_PROVIDER__API_KEY_ENV": "MODELSURGEON_TEST_SECRET"},
        cli_overrides={"provider.request_timeout_seconds": 12},
    )
    second = discover_settings(
        path,
        environ={"MODELSURGEON_PROVIDER__API_KEY_ENV": "MODELSURGEON_TEST_SECRET"},
        cli_overrides={"provider.request_timeout_seconds": 12},
    )

    assert first.canonical_json() == second.canonical_json()
    assert first.configuration_digest == second.configuration_digest
    assert "MODELSURGEON_TEST_SECRET" not in first.canonical_json()
    assert "provider.api_key_env" in first.canonical_json()
    assert [source.name for source in first.sources] == [
        "defaults",
        "file",
        "environment",
        "cli",
    ]


def test_remote_diagnostics_are_unknown_until_measured_and_redact_secrets() -> None:
    diagnostic = provider_diagnostics(
        _configured_provider(),
        environ={"MODELSURGEON_PROVIDER_KEY": "super-secret-token"},
    )

    assert diagnostic.status is ProviderDiagnosticStatus.UNKNOWN
    assert diagnostic.state is ProviderCapabilityState.UNKNOWN
    assert diagnostic.code == "remote_capabilities_unknown"
    assert "super-secret-token" not in diagnostic.canonical_json()
    assert {item.state for item in diagnostic.capabilities} == {ProviderCapabilityState.UNKNOWN}


def test_offline_remote_and_missing_local_model_are_explicit() -> None:
    remote = provider_diagnostics(_configured_provider(), offline=True)
    assert remote.status is ProviderDiagnosticStatus.UNSUPPORTED
    assert remote.code == "offline_provider_disabled"
    assert remote.state is ProviderCapabilityState.UNSUPPORTED

    local = ProviderConfig(
        kind=ProviderKind.LOCAL,
        provider_id="local",
        model_id="chat",
        model_revision="revision-1",
        model_path=Path("missing-chat.gguf"),
    )
    missing = provider_diagnostics(local)
    assert missing.status is ProviderDiagnosticStatus.UNAVAILABLE
    assert missing.code == "local_model_missing"
    assert missing.state is ProviderCapabilityState.UNKNOWN


def test_capability_summary_preserves_measured_and_unsupported_cells() -> None:
    identity = ProviderModelIdentity("endpoint", "model", "revision-1")
    card = ProviderCapabilityCard(
        identity,
        ProviderKind.COMPATIBLE_ENDPOINT,
        "fixture-provider-v1",
        (ProviderCapability.EXPLAIN_EVIDENCE,),
        ProviderLimits(100, 100, 200),
    )
    summary = summarize_provider_capabilities(
        card,
        measured=(ProviderCapability.EXPLAIN_EVIDENCE,),
    )
    states = {item.capability: item.state for item in summary}

    assert states[ProviderCapability.EXPLAIN_EVIDENCE] is ProviderCapabilityState.MEASURED
    assert states[ProviderCapability.INTERPRET_INTENT] is ProviderCapabilityState.UNSUPPORTED
    assert states[ProviderCapability.STREAMING] is ProviderCapabilityState.UNSUPPORTED


def test_cli_and_python_diagnostics_have_direct_api_parity() -> None:
    overrides = {
        "provider.kind": "none",
        "provider.provider_id": "none",
        "provider.model_id": "none",
        "provider.model_revision": "none",
        "provider.model_path": None,
        "provider.runtime_revision": None,
        "provider.endpoint": None,
        "provider.api_key_env": None,
    }
    discovery = discover_settings(environ={}, cli_overrides=overrides)
    expected = provider_diagnostics(
        discovery.settings,
        configuration_digest=discovery.configuration_digest,
    ).canonical_json()
    result = CliRunner().invoke(
        app,
        ["provider", "diagnostics", "--no-llm", "--json"],
        color=False,
        env={},
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == expected


def test_diagnostics_do_not_mutate_settings_or_provider_configuration() -> None:
    settings = Settings()
    before = settings.canonical_json()
    diagnostic = provider_diagnostics(settings)

    assert settings.canonical_json() == before
    assert diagnostic.to_record()["kind"] == "none"

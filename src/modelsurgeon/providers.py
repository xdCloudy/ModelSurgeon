"""Provider composition and diagnostics without making providers an optimizer dependency."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.config import ProviderConfig, Settings
from modelsurgeon.conversation import NullTextModelProvider, TextModelProvider
from modelsurgeon.conversation.isolation import redact_secret_text, redact_untrusted_value
from modelsurgeon.provider_kind import ProviderKind

PROVIDER_DIAGNOSTIC_SCHEMA_VERSION = 1


class ProviderDiagnosticStatus(StrEnum):
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


class ProviderConfigurationError(ValueError):
    """Stable, actionable error raised when a selected provider cannot be composed."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = redact_secret_text(message)
        super().__init__(f"{code}: {self.message}")


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic:
    """Redacted provider availability record for CLI and Python callers."""

    status: ProviderDiagnosticStatus
    code: str
    message: str
    kind: ProviderKind
    provider_id: str
    model_id: str
    dependency: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "provider_diagnostic",
            "schema_version": PROVIDER_DIAGNOSTIC_SCHEMA_VERSION,
            "status": self.status.value,
            "code": self.code,
            "message": redact_secret_text(self.message),
            "kind": self.kind.value,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "dependency": self.dependency,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            redact_untrusted_value(self.to_record()), sort_keys=True, separators=(",", ":")
        )


def provider_diagnostics(
    settings: Settings | ProviderConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> ProviderDiagnostic:
    """Describe provider availability without importing an optional adapter."""

    config = settings.provider if isinstance(settings, Settings) else settings
    if config.kind is ProviderKind.NONE:
        return ProviderDiagnostic(
            ProviderDiagnosticStatus.DISABLED,
            "no_llm",
            "no text-model provider is configured; direct CLI/Python APIs remain available",
            config.kind,
            config.provider_id,
            config.model_id,
        )

    if config.api_key_env is not None:
        values = os.environ if environ is None else environ
        if not values.get(config.api_key_env):
            return ProviderDiagnostic(
                ProviderDiagnosticStatus.UNAVAILABLE,
                "missing_api_key",
                f"provider API key environment variable {config.api_key_env!r} is not set",
                config.kind,
                config.provider_id,
                config.model_id,
            )

    dependency = f"modelsurgeon-provider-{config.kind.value}"
    return ProviderDiagnostic(
        ProviderDiagnosticStatus.UNAVAILABLE,
        "adapter_unavailable",
        f"provider kind {config.kind.value!r} has no installed adapter; "
        "choose provider.kind='none' for direct CLI/Python use",
        config.kind,
        config.provider_id,
        config.model_id,
        dependency,
    )


def resolve_text_model_provider(
    settings: Settings | ProviderConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> TextModelProvider:
    """Resolve the configured provider, failing closed for unavailable adapters."""

    config = settings.provider if isinstance(settings, Settings) else settings
    diagnostic = provider_diagnostics(config, environ=environ)
    if config.kind is ProviderKind.NONE:
        return NullTextModelProvider()
    raise ProviderConfigurationError(diagnostic.code, diagnostic.message)


__all__ = [
    "PROVIDER_DIAGNOSTIC_SCHEMA_VERSION",
    "ProviderConfigurationError",
    "ProviderDiagnostic",
    "ProviderDiagnosticStatus",
    "provider_diagnostics",
    "resolve_text_model_provider",
]

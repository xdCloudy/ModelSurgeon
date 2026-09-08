"""Provider selection, deterministic discovery, and safe user diagnostics."""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast
from urllib.parse import urlsplit

from modelsurgeon.config import ProviderConfig, Settings
from modelsurgeon.conversation import (
    AuthReference,
    CompatibleEndpointProvider,
    EndpointConfig,
    HostedProviderAdapter,
    LocalGGUFProvider,
    LocalGGUFProviderConfig,
    NullTextModelProvider,
    ProviderCapability,
    ProviderCapabilityCard,
    ProviderLimits,
    SecretResolver,
    TextModelProvider,
)
from modelsurgeon.conversation.isolation import redact_secret_text, redact_untrusted_value
from modelsurgeon.provider_kind import ProviderKind

PROVIDER_DIAGNOSTIC_SCHEMA_VERSION = 2


class ProviderDiagnosticStatus(StrEnum):
    """Coarse status retained for compatibility with the v2.2 API."""

    DISABLED = "disabled"
    SUPPORTED = "supported"
    CONFIGURED = "configured"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"
    FAILED = "failed"


class ProviderCapabilityState(StrEnum):
    """Evidence state for one provider capability."""

    DETECTED = "detected"
    MEASURED = "measured"
    CONFIGURED = "configured"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ProviderCapabilitySummary:
    """A deterministic, user-facing capability cell."""

    capability: ProviderCapability
    state: ProviderCapabilityState
    detail: str

    def to_record(self) -> dict[str, object]:
        return {
            "capability": self.capability.value,
            "state": self.state.value,
            "detail": redact_secret_text(self.detail),
        }


def summarize_provider_capabilities(
    card: ProviderCapabilityCard | None = None,
    *,
    configured: Iterable[ProviderCapability] = (),
    measured: Iterable[ProviderCapability] = (),
    unknown: Iterable[ProviderCapability] = (),
) -> tuple[ProviderCapabilitySummary, ...]:
    """Summarize every known capability without guessing unsupported cells."""

    all_capabilities = tuple(sorted(ProviderCapability, key=lambda item: item.value))
    configured_set = set(configured)
    measured_set = set(measured)
    unknown_set = set(unknown)
    summaries: list[ProviderCapabilitySummary] = []
    for capability in all_capabilities:
        if capability in measured_set:
            state = ProviderCapabilityState.MEASURED
            detail = "capability was exercised by a bounded probe"
        elif card is not None and capability in card.capabilities:
            state = ProviderCapabilityState.DETECTED
            detail = "capability is advertised by the resolved provider card"
        elif capability in configured_set:
            state = ProviderCapabilityState.CONFIGURED
            detail = "capability is selected by configuration but has not been probed"
        elif capability in unknown_set or card is None:
            state = ProviderCapabilityState.UNKNOWN
            detail = "capability has not been observed or measured"
        else:
            state = ProviderCapabilityState.UNSUPPORTED
            detail = "capability is not advertised by the resolved provider card"
        summaries.append(ProviderCapabilitySummary(capability, state, detail))
    return tuple(summaries)


class ProviderConfigurationError(ValueError):
    """Stable, actionable error raised when a selected provider cannot be composed."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = redact_secret_text(message)
        super().__init__(f"{code}: {self.message}")


def _safe_configuration(config: ProviderConfig) -> dict[str, object]:
    endpoint = config.endpoint
    if endpoint is not None:
        parsed = urlsplit(endpoint)
        endpoint = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return {
        "kind": config.kind.value,
        "provider_id": config.provider_id,
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "model_path": None if config.model_path is None else "<redacted>",
        "runtime_revision": config.runtime_revision,
        "endpoint": endpoint,
        "api_key_env": config.api_key_env,
        "request_timeout_seconds": config.request_timeout_seconds,
        "max_input_size": config.max_input_size,
        "max_output_size": config.max_output_size,
    }


def _unknown_capabilities() -> tuple[ProviderCapabilitySummary, ...]:
    return summarize_provider_capabilities()


def _unsupported_capabilities() -> tuple[ProviderCapabilitySummary, ...]:
    return tuple(
        ProviderCapabilitySummary(
            capability, ProviderCapabilityState.UNSUPPORTED, "capability is disabled"
        )
        for capability in sorted(ProviderCapability, key=lambda item: item.value)
    )


@dataclass(frozen=True, slots=True)
class ProviderDiagnostic:
    """Redacted provider availability and capability record."""

    status: ProviderDiagnosticStatus
    code: str
    message: str
    kind: ProviderKind
    provider_id: str
    model_id: str
    dependency: str | None = None
    state: ProviderCapabilityState = ProviderCapabilityState.UNKNOWN
    capabilities: tuple[ProviderCapabilitySummary, ...] = ()
    configuration: Mapping[str, object] = field(default_factory=dict)
    configuration_digest: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "record_type": "provider_diagnostic",
            "schema_version": PROVIDER_DIAGNOSTIC_SCHEMA_VERSION,
            "status": self.status.value,
            "state": self.state.value,
            "code": self.code,
            "message": redact_secret_text(self.message),
            "kind": self.kind.value,
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "dependency": self.dependency,
            "configuration": dict(self.configuration),
            "configuration_digest": self.configuration_digest,
            "capabilities": [item.to_record() for item in self.capabilities],
            "details": dict(self.details),
        }
        return redact_untrusted_value(record)  # type: ignore[return-value]

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_record(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def _diagnostic(
    status: ProviderDiagnosticStatus,
    code: str,
    message: str,
    config: ProviderConfig,
    *,
    state: ProviderCapabilityState,
    capabilities: tuple[ProviderCapabilitySummary, ...],
    dependency: str | None = None,
    configuration_digest: str | None = None,
    details: Mapping[str, object] | None = None,
) -> ProviderDiagnostic:
    return ProviderDiagnostic(
        status,
        code,
        message,
        config.kind,
        config.provider_id,
        config.model_id,
        dependency,
        state,
        capabilities,
        _safe_configuration(config),
        configuration_digest,
        {} if details is None else details,
    )


def _local_runtime_detected() -> tuple[bool, str | None]:
    try:
        available = importlib.util.find_spec("llama_cpp") is not None
    except (ImportError, ModuleNotFoundError, ValueError) as error:
        return False, f"optional local runtime could not be inspected: {type(error).__name__}"
    return available, None


def provider_diagnostics(
    settings: Settings | ProviderConfig,
    *,
    environ: Mapping[str, str] | None = None,
    offline: bool = False,
    capability_card: ProviderCapabilityCard | None = None,
    required_capabilities: Iterable[ProviderCapability] = (),
    configuration_digest: str | None = None,
) -> ProviderDiagnostic:
    """Describe provider configuration without changing policy or evidence.

    Remote providers are never contacted unless a caller separately performs
    a bounded provider probe and passes its capability card.
    """

    config = settings.provider if isinstance(settings, Settings) else settings
    required = tuple(sorted(set(required_capabilities), key=lambda item: item.value))
    if config.kind is ProviderKind.NONE:
        return _diagnostic(
            ProviderDiagnosticStatus.DISABLED,
            "no_llm",
            "no text-model provider is configured; direct CLI/Python APIs remain available",
            config,
            state=ProviderCapabilityState.CONFIGURED,
            capabilities=_unsupported_capabilities(),
            configuration_digest=configuration_digest,
        )

    if offline and config.kind in {ProviderKind.COMPATIBLE_ENDPOINT, ProviderKind.HOSTED}:
        return _diagnostic(
            ProviderDiagnosticStatus.UNSUPPORTED,
            "offline_provider_disabled",
            "offline mode permits only local or explicit no-LLM operation; "
            "no remote provider was contacted",
            config,
            state=ProviderCapabilityState.UNSUPPORTED,
            capabilities=_unsupported_capabilities(),
            configuration_digest=configuration_digest,
        )

    values = os.environ if environ is None else environ
    if config.api_key_env is not None and not values.get(config.api_key_env):
        return _diagnostic(
            ProviderDiagnosticStatus.UNAVAILABLE,
            "missing_api_key",
            f"provider API key environment variable {config.api_key_env!r} is not set",
            config,
            state=ProviderCapabilityState.CONFIGURED,
            capabilities=_unknown_capabilities(),
            configuration_digest=configuration_digest,
        )

    if config.kind is ProviderKind.LOCAL:
        if config.model_path is None:
            return _diagnostic(
                ProviderDiagnosticStatus.UNAVAILABLE,
                "adapter_unavailable",
                "local provider is selected but no local GGUF model path is configured",
                config,
                state=ProviderCapabilityState.UNKNOWN,
                capabilities=_unknown_capabilities(),
                dependency="modelsurgeon-provider-local",
                configuration_digest=configuration_digest,
            )
        path = config.model_path.expanduser()
        if not path.is_file():
            return _diagnostic(
                ProviderDiagnosticStatus.UNAVAILABLE,
                "local_model_missing",
                "configured local GGUF model file does not exist",
                config,
                state=ProviderCapabilityState.UNKNOWN,
                capabilities=_unknown_capabilities(),
                configuration_digest=configuration_digest,
            )
        if path.suffix.lower() != ".gguf":
            return _diagnostic(
                ProviderDiagnosticStatus.UNSUPPORTED,
                "local_model_unsupported",
                "local provider requires a .gguf model file",
                config,
                state=ProviderCapabilityState.UNSUPPORTED,
                capabilities=_unsupported_capabilities(),
                configuration_digest=configuration_digest,
            )
        runtime_available, runtime_detail = _local_runtime_detected()
        if not runtime_available:
            return _diagnostic(
                ProviderDiagnosticStatus.UNAVAILABLE,
                "adapter_unavailable",
                runtime_detail or "llama-cpp-python is not installed for the local provider",
                config,
                state=ProviderCapabilityState.UNKNOWN,
                capabilities=_unknown_capabilities(),
                dependency="llama-cpp-python",
                configuration_digest=configuration_digest,
            )
        configured = (
            ProviderCapability.CLARIFY_INTENT,
            ProviderCapability.EXPLAIN_EVIDENCE,
            ProviderCapability.INTERPRET_INTENT,
            ProviderCapability.STRUCTURED_OUTPUT,
        )
        return _diagnostic(
            ProviderDiagnosticStatus.SUPPORTED,
            "local_provider_detected",
            "local GGUF model and optional runtime are detected; capability execution "
            "remains bounded by the provider card",
            config,
            state=ProviderCapabilityState.DETECTED,
            capabilities=summarize_provider_capabilities(configured=configured),
            configuration_digest=configuration_digest,
            details={"model_path_present": True, "runtime_detected": True},
        )

    if config.endpoint is None:
        return _diagnostic(
            ProviderDiagnosticStatus.UNSUPPORTED,
            "remote_endpoint_missing",
            "remote provider selection requires an endpoint",
            config,
            state=ProviderCapabilityState.UNSUPPORTED,
            capabilities=_unsupported_capabilities(),
            configuration_digest=configuration_digest,
        )

    if config.api_key_env is None:
        return _diagnostic(
            ProviderDiagnosticStatus.UNAVAILABLE,
            "adapter_unavailable",
            f"provider kind {config.kind.value!r} has no credential reference; "
            "configure an API-key environment variable or choose provider.kind='none'",
            config,
            state=ProviderCapabilityState.CONFIGURED,
            capabilities=_unknown_capabilities(),
            dependency=f"modelsurgeon-provider-{config.kind.value}",
            configuration_digest=configuration_digest,
        )

    if capability_card is not None:
        summaries = summarize_provider_capabilities(capability_card)
        by_capability = {item.capability: item for item in summaries}
        missing = tuple(
            capability
            for capability in required
            if by_capability[capability].state is ProviderCapabilityState.UNSUPPORTED
        )
        if missing:
            return _diagnostic(
                ProviderDiagnosticStatus.UNSUPPORTED,
                "missing_capability",
                "resolved provider does not advertise: "
                + ", ".join(item.value for item in missing),
                config,
                state=ProviderCapabilityState.UNSUPPORTED,
                capabilities=summaries,
                configuration_digest=configuration_digest,
                details={"missing_capabilities": [item.value for item in missing]},
            )
        return _diagnostic(
            ProviderDiagnosticStatus.SUPPORTED,
            "remote_capabilities_detected",
            "remote provider capability card was supplied and matched the configured identity",
            config,
            state=ProviderCapabilityState.DETECTED,
            capabilities=summaries,
            configuration_digest=configuration_digest,
        )
    return _diagnostic(
        ProviderDiagnosticStatus.UNKNOWN,
        "remote_capabilities_unknown",
        "remote provider is configured but capabilities are unknown until a bounded "
        "endpoint probe is run",
        config,
        state=ProviderCapabilityState.UNKNOWN,
        capabilities=_unknown_capabilities(),
        configuration_digest=configuration_digest,
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
    if diagnostic.status is not ProviderDiagnosticStatus.SUPPORTED:
        raise ProviderConfigurationError(diagnostic.code, diagnostic.message)
    if config.kind is ProviderKind.LOCAL:
        assert config.model_path is not None
        return LocalGGUFProvider(
            LocalGGUFProviderConfig(
                config.model_path,
                config.model_revision,
                config.runtime_revision or "configured-runtime",
                max_input_tokens=config.max_input_size,
                max_output_tokens=config.max_output_size,
                max_context_tokens=config.max_input_size + config.max_output_size,
                max_wall_seconds=config.request_timeout_seconds,
            )
        )
    assert config.endpoint is not None
    auth = AuthReference("provider-key") if config.api_key_env is not None else None
    endpoint_config = EndpointConfig(
        "modelsurgeon-endpoint",
        config.endpoint,
        config.provider_id,
        config.model_id,
        config.model_revision,
        config.kind,
        auth=auth,
        limits=ProviderLimits(
            config.max_input_size,
            config.max_output_size,
            config.max_input_size + config.max_output_size,
            max_wall_seconds=config.request_timeout_seconds,
        ),
    )
    secret_values = os.environ if environ is None else environ
    resolver: SecretResolver | None = None
    if config.api_key_env is not None:
        def resolve_secret(_: str) -> str:
            return secret_values.get(config.api_key_env or "") or ""

        resolver = cast(SecretResolver, resolve_secret)
    provider_type = (
        CompatibleEndpointProvider
        if config.kind is ProviderKind.COMPATIBLE_ENDPOINT
        else HostedProviderAdapter
    )
    return provider_type(endpoint_config, auth_resolver=resolver)


__all__ = [
    "PROVIDER_DIAGNOSTIC_SCHEMA_VERSION",
    "ProviderCapabilityState",
    "ProviderCapabilitySummary",
    "ProviderConfigurationError",
    "ProviderDiagnostic",
    "ProviderDiagnosticStatus",
    "provider_diagnostics",
    "resolve_text_model_provider",
    "summarize_provider_capabilities",
]

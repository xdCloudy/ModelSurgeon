"""Protocol tests for hosted and compatible endpoint adapters.

These fixtures exercise transport and contract behavior only.  They are not
model-quality or optimization measurements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from modelsurgeon.conversation import (
    AuthReference,
    AuthScheme,
    CancellationToken,
    CapabilityProbeOutcome,
    CompatibleEndpointProvider,
    EndpointAdapterError,
    EndpointConfig,
    EndpointHttpRequest,
    EndpointHttpResponse,
    ExplanationRequest,
    HostedProviderAdapter,
    ProviderCapability,
    ProviderKind,
    ProviderLimits,
    ProviderOutcome,
    RetryPolicy,
)

_DEFAULT_AUTH = AuthReference("fixture-credential")
_DEFAULT_RETRY_POLICY = RetryPolicy(max_attempts=2, base_delay_seconds=0.0)


@dataclass
class FixtureTransport:
    responses: list[EndpointHttpResponse]
    requests: list[EndpointHttpRequest] = field(default_factory=list)

    def request(self, request: EndpointHttpRequest) -> EndpointHttpResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("fixture transport was called unexpectedly")
        return self.responses.pop(0)


def _config(
    *,
    kind: ProviderKind = ProviderKind.COMPATIBLE_ENDPOINT,
    auth: AuthReference | None = _DEFAULT_AUTH,
    retry_policy: RetryPolicy | None = _DEFAULT_RETRY_POLICY,
) -> EndpointConfig:
    if retry_policy is None:
        retry_policy = RetryPolicy(max_attempts=2, base_delay_seconds=0.0)
    return EndpointConfig(
        "fixture-endpoint",
        "https://provider.example/v1",
        "fixture-provider",
        "fixture-model",
        "revision-1",
        kind,
        auth=auth,
        limits=ProviderLimits(4096, 1024, 8192, max_wall_seconds=5.0),
        retry_policy=retry_policy,
    )


def _probe_response(
    config: EndpointConfig, *, capabilities: list[str] | None = None
) -> EndpointHttpResponse:
    payload = {
        "object": "model",
        "id": config.model_id,
        "modelsurgeon": {
            "protocol_version": 1,
            "provider_id": config.provider_id,
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            "capabilities": capabilities
            or [
                ProviderCapability.CLARIFY_INTENT.value,
                ProviderCapability.EXPLAIN_EVIDENCE.value,
                ProviderCapability.INTERPRET_INTENT.value,
                ProviderCapability.STRUCTURED_OUTPUT.value,
            ],
            "limits": config.limits.to_record(),
            "structured_output_schemas": ["modelsurgeon.provider_output.v1"],
        },
    }
    return _response(payload)


def _response(
    payload: object,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> EndpointHttpResponse:
    return EndpointHttpResponse(
        status,
        {} if headers is None else headers,
        json.dumps(payload).encode(),
    )


def _completion(content: object, *, model: str = "fixture-model") -> EndpointHttpResponse:
    return _response(
        {
            "id": "response-1",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content),
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    )


def _request() -> ExplanationRequest:
    return ExplanationRequest("request-1", ({"record_type": "evidence", "value": 1},))


def test_config_and_probe_records_are_secret_free() -> None:
    config = _config()
    assert "fixture-secret" not in config.canonical_json()
    assert config.to_record()["auth"] == {
        "credential_ref": "fixture-credential",
        "header_name": "authorization",
        "scheme": "bearer",
    }

    transport = FixtureTransport([_probe_response(config)])
    provider = CompatibleEndpointProvider(
        config,
        auth_resolver=lambda ref: "fixture-secret" if ref == "fixture-credential" else None,
        transport=transport,
    )
    result = provider.probe()
    assert result.outcome is CapabilityProbeOutcome.SUPPORTED
    assert result.card is not None
    assert result.card.kind is ProviderKind.COMPATIBLE_ENDPOINT
    assert "fixture-secret" not in result.canonical_json()
    assert transport.requests[0].headers[-1] == ("authorization", "Bearer fixture-secret")


def test_missing_auth_fails_closed_without_transport() -> None:
    config = _config()
    transport = FixtureTransport([])
    provider = CompatibleEndpointProvider(config, transport=transport)

    result = provider.probe()

    assert result.outcome is CapabilityProbeOutcome.FAILED
    assert result.detail == "configured endpoint credential is unavailable"
    assert transport.requests == []


def test_capability_probe_rejects_contradictory_identity_and_unknown_capability() -> None:
    config = _config(auth=None)
    contradictory = _probe_response(config)
    payload = json.loads(contradictory.body)
    payload["modelsurgeon"]["model_revision"] = "different-revision"
    provider = CompatibleEndpointProvider(
        config,
        transport=FixtureTransport([_response(payload)]),
    )
    result = provider.probe()
    assert result.outcome is CapabilityProbeOutcome.MALFORMED_OUTPUT
    assert result.card is None

    unknown = json.loads(_probe_response(config).body)
    unknown["modelsurgeon"]["capabilities"].append("execute_model")
    provider = CompatibleEndpointProvider(
        config,
        transport=FixtureTransport([_response(unknown)]),
    )
    assert provider.probe().outcome is CapabilityProbeOutcome.MALFORMED_OUTPUT


def test_rate_limit_is_retried_with_bounded_delay_and_success_is_validated() -> None:
    config = _config(auth=None)
    sleeps: list[float] = []
    transport = FixtureTransport(
        [
            _probe_response(config),
            _response({}, status=429, headers={"retry-after": "0"}),
            _completion(
                {
                    "operation": "explain_evidence",
                    "text": "measured evidence only",
                    "evidence_refs": [],
                }
            ),
        ]
    )
    provider = CompatibleEndpointProvider(config, transport=transport, sleeper=sleeps.append)
    provider.start()
    result = provider.call(_request(), cancellation=CancellationToken())

    assert result.outcome is ProviderOutcome.SUPPORTED
    assert len(transport.requests) == 3
    assert sleeps == [0.0]
    body = json.loads(transport.requests[-1].body)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True


def test_malformed_or_contradictory_completion_never_becomes_supported() -> None:
    config = _config(auth=None)
    malformed = _completion({"operation": "explain_evidence", "execute": "delete"})
    provider = CompatibleEndpointProvider(
        config,
        transport=FixtureTransport([_probe_response(config), malformed]),
    )
    provider.start()
    result = provider.call(_request(), cancellation=CancellationToken())
    assert result.outcome is ProviderOutcome.MALFORMED_OUTPUT
    assert result.output is None

    contradictory = CompatibleEndpointProvider(
        config,
        transport=FixtureTransport([_probe_response(config), _completion({}, model="other-model")]),
    )
    contradictory.start()
    result = contradictory.call(_request(), cancellation=CancellationToken())
    assert result.outcome is ProviderOutcome.MALFORMED_OUTPUT


def test_unadvertised_structured_output_is_explicitly_unsupported() -> None:
    config = _config(auth=None)
    capabilities = [
        ProviderCapability.EXPLAIN_EVIDENCE.value,
    ]
    probe = json.loads(_probe_response(config, capabilities=capabilities).body)
    probe["modelsurgeon"]["structured_output_schemas"] = []
    provider = CompatibleEndpointProvider(
        config,
        transport=FixtureTransport([_response(probe)]),
    )
    provider.start()
    result = provider.call(_request(), cancellation=CancellationToken())
    assert result.outcome is ProviderOutcome.UNSUPPORTED
    assert result.unsupported_capabilities == (ProviderCapability.STRUCTURED_OUTPUT,)


def test_hosted_adapter_preserves_kind_and_auth_scheme_is_indirected() -> None:
    config = _config(
        kind=ProviderKind.HOSTED,
        auth=AuthReference("hosted-key", header_name="x-api-key", scheme=AuthScheme.RAW),
    )
    transport = FixtureTransport([_probe_response(config)])
    provider = HostedProviderAdapter(
        config,
        auth_resolver=lambda _: "hosted-secret",
        transport=transport,
    )
    provider.start()
    assert provider.capability_card.kind is ProviderKind.HOSTED
    assert transport.requests[0].headers[-1] == ("x-api-key", "hosted-secret")
    assert "hosted-secret" not in provider.capability_card.canonical_json()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://provider.example",
        "https://user:password@provider.example",
        "https://provider.example?api_key=secret",
    ],
)
def test_endpoint_configuration_rejects_unsafe_urls(base_url: str) -> None:
    with pytest.raises(EndpointAdapterError):
        EndpointConfig(
            "fixture-endpoint",
            base_url,
            "fixture-provider",
            "fixture-model",
            "revision-1",
            ProviderKind.COMPATIBLE_ENDPOINT,
        )

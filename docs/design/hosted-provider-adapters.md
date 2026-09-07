# Hosted and compatible endpoint adapters

Status: experimental, fail-closed protocol boundary

`modelsurgeon.conversation.endpoint` provides `CompatibleEndpointProvider` and
`HostedProviderAdapter`. Both use the same narrow chat-completions-shaped wire
protocol; the `ProviderKind` remains explicit so a hosted endpoint is not
silently treated as a local or compatible implementation.

## Configuration and secrets

`EndpointConfig` contains only endpoint metadata, model identity, hard client
limits, retry policy, and an optional `AuthReference`. The reference names a
credential in a caller-owned secret store. The adapter resolves it immediately
before transport through `SecretResolver`; the credential is never included in
endpoint records, capability cards, provenance, failure details, or prompts.
HTTPS is required and URL userinfo, query strings, and fragments are rejected.

## Capability probe

The adapter first requests `/models/{model_id}` (or an explicitly configured
path). A successful response must contain exactly `object`, `id`, and
`modelsurgeon`. The extension has schema version `1`, the configured provider
and model identity, a sorted capability list, bounded limits, and sorted
structured-output schema IDs:

```json
{
  "object": "model",
  "id": "model-id",
  "modelsurgeon": {
    "protocol_version": 1,
    "provider_id": "provider-id",
    "model_id": "model-id",
    "model_revision": "revision",
    "capabilities": ["explain_evidence", "structured_output"],
    "limits": {
      "max_input_tokens": 4096,
      "max_output_tokens": 1024,
      "max_context_tokens": 8192,
      "max_concurrent_requests": 1,
      "max_wall_seconds": 30.0,
      "max_stream_events": 4096
    },
    "structured_output_schemas": ["modelsurgeon.provider_output.v1"]
  }
}
```

Missing, unknown, contradictory, or unsorted values produce a retained
`malformed_output` probe result. No capability is inferred from a provider
name, endpoint URL, HTTP success status, or ordinary `/models` metadata.
Advertised limits are intersected with local configured limits.

## Requests and responses

Requests are sent to `/chat/completions` with one system message and one user
message, `stream: false`, deterministic temperature `0`, and a negotiated
`response_format`. A provider advertising
`modelsurgeon.provider_output.v1` receives strict JSON-schema negotiation;
otherwise the adapter uses JSON-object mode only when structured output is
advertised. The returned response must contain exactly one assistant text
choice, the configured model identity, a non-truncated finish reason, and a
JSON object accepted by the typed provider decoder. Tool calls, execution
commands, multiple choices, free-form text, and unknown fields are rejected.

Every result retains the existing provider request/response digests and typed
outcome. Unsupported operations remain explicit and cannot call ModelSurgeon
execution APIs.

## Retry and failure policy

Retries are bounded to five attempts and to the request wall-time budget.
Transport errors and 408/429/5xx responses may retry; `Retry-After` is parsed
and capped. Authentication failures and other 4xx responses do not retry.
Response bodies are bounded before JSON parsing. Failure details contain only
stable generic diagnostics, never response bodies, URLs with credentials, or
resolved secrets.

The fixture transport in `tests/test_endpoint_adapters.py` is a deterministic
protocol test double. It is not optimization evidence or a hosted-provider
benchmark.

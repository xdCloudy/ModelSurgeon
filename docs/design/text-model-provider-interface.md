# Replaceable text-model provider interface

Status: implemented as the `modelsurgeon.conversation` provider contract in
v2.2. This is an adapter boundary, not a vendor SDK integration.

## Responsibility and authority

`TextModelProvider` is an optional control-plane client. Its only operations
are:

- interpret a user request into a validated `IntentRecord`;
- ask bounded clarification questions about an existing intent record; and
- explain typed, canonical evidence.

The interface has no execution method, optimizer handle, model session,
filesystem handle, shell callback, network callback, or generic tool hook.
`OptimizationSpec` validation, candidate selection, mutation, evaluation,
approval, rollback, artifact writes, and evidence recording remain in
ModelSurgeon.

## Contract records

Every provider advertises a deterministic `ProviderCapabilityCard` containing:

- `ProviderModelIdentity`: provider, model, revision, and optional content
  digest;
- `ProviderCapability`: interpretation, clarification, explanation, streaming,
  and structured-output support;
- `ProviderLimits`: input/output/context token limits, concurrency, wall time,
  and stream-event limits; and
- structured output schema identifiers.

Requests are one of `InterpretIntentRequest`, `ClarificationRequest`, or
`ExplanationRequest`. They carry a stable request ID and a positive
`ProviderBudget`. Explanation inputs are evidence records only; clarification
inputs are the typed intent record, not a free-form prompt history.

Responses are one of the typed output records. An interpretation response is
accepted only after `IntentRecord.from_json` validates it. Unknown fields,
invalid schema versions, malformed values, and operation mismatches become a
retained `MALFORMED_OUTPUT` result rather than an executable request.

## Lifecycle, streaming, and failures

The caller owns provider lifecycle: call `start()` before use and `close()` in
the owning context. `call()` is the non-streaming boundary; `stream()` is the
streaming boundary. Both receive a cooperative `CancellationToken`. The
`invoke_provider()` helper checks advertised capabilities, rejects budgets over
the provider limit, converts pre-cancelled calls to `CANCELLED`, and records a
`TIMEOUT` when the wall-time budget expires.

All result states are explicit: `SUPPORTED`, `UNSUPPORTED`, `FAILED`,
`UNKNOWN`, `TIMEOUT`, `CANCELLED`, and `MALFORMED_OUTPUT`. A failure contains a
machine-readable code and a redacted diagnostic. It never fabricates a metric,
measurement, approval, or artifact identity.

## Identity and provenance

Request and response content are canonically serialized with sorted keys and
compact JSON. `ProviderProvenance` retains the provider revision, deterministic
request digest, optional response digest, and evidence references. Runtime
timings are not part of identity. Secrets are not part of canonical records;
failure diagnostics redact common authorization, token, password, secret, and
API-key forms before serialization.

## No-LLM operation

`NullTextModelProvider` advertises no capabilities and returns explicit
`UNSUPPORTED` results. It is useful for callers that want one provider-shaped
configuration while preserving the direct CLI and Python APIs. No provider is
required to inspect models, compile an optimization plan, or execute a
campaign.

Future provider implementations may target local GGUF runtimes, compatible
endpoints, hosted providers, or another allowlisted runtime. They must conform
to these records and must not add execution authority to the provider layer.

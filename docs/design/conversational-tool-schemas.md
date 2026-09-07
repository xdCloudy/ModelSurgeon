# Capability-scoped conversational tool schemas

Status: request schema version 1, result envelope version 2, and the trusted
dispatcher boundary are implemented in `modelsurgeon.conversation`. They define
what a conversational client may request and how an engine may publish a
grounded result; they do not make the conversational product generally
available or register a default model executor.

## Boundary contract

The default catalog contains four explicit operations:

| Tool | Access | Capability | Approval |
| --- | --- | --- | --- |
| `inspect_model` | read-only | `inspect_model` | no |
| `preview_plan` | read-only | `preview_plan` | no |
| `query_evidence` | read-only | `query_evidence` | no |
| `execute_approved_plan` | consequential | `execute_approved_plan` | required |

Each `ToolDefinition` carries its owner, schema version, capability, access
classification, strict input and output JSON schemas, hard wall/memory/
evaluation/output budgets, and the complete set of supported failure states.
Schemas reject unknown fields and arbitrary additional properties. Tool and
request identities are SHA-256 identities of canonical JSON, so replay and
schema drift are detectable.

The consequential schema accepts only an existing `plan_id`, its
`plan_digest`, and an `approval_id`. It contains no tensor names, tensor
indices, removal operations, shell/Python/filesystem/network fields, or
callback/executor field. A future engine adapter must resolve and validate the
referenced plan through its existing mutation, approval, transaction,
rollback, acceptance, budget, and provenance boundaries.

## Negotiation and failure behavior

`ToolCatalog.negotiate()` is the first pre-execution gate. Unknown tool names return
`UNKNOWN`; known tools requested with the wrong capability return
`UNSUPPORTED`; unknown schema versions, tool identity drift, invalid input,
budget expansion, approval mismatch, and missing approval return typed `REFUSED`
failures. No negotiation method invokes an implementation. `negotiate_record()` converts
untrusted JSON into a typed refusal and retains the request identity when it
is safe to do so.

## Dispatch enforcement

`ToolDispatcher` is the only runtime entry point for a registered conversational
handler. `dispatch_record()` decodes untrusted data before handler lookup; a
malformed, oversized, unknown-version, or adversarial record has no typed
request and cannot reach an adapter. Handlers are registered by trusted engine
code against an existing catalog name. Requests never contain callbacks,
commands, paths, or provider selectors.

Every dispatch also owns one `ToolTransactionBoundary` handle. Read-only
operations receive a logical handle with no engine participant and cannot
commit or roll back mutable state. Their successful result closes the handle
as committed; cancellation, timeout, invalid output, and handler failure close
it as rolled back. A consequential handler receives the same narrow lifecycle
capability and must explicitly commit before a supported result can be
published. Trusted engine code may bind that handle to an already-prepared
transaction participant with commit/rollback hooks; the conversational layer
does not expose a mutation method, model object, artifact writer, or campaign
store.

Handles are generation-bound and become stale after a transition. A later
operation cannot reuse an earlier handle, and a bounded active-handle limit
refuses new work rather than growing without limit. Cancellation and timeout
roll back an active participant before a negative result is returned. A
consequential retry is rejected with `retry_not_safe` unless the trusted
handler explicitly declares the failed operation idempotent; an identical
completed request is still served from the existing replay ledger without
invoking the handler again.

The dispatcher negotiates the catalog entry, strict input schema, capability,
tool identity, and request budget before applying an optional dispatcher-wide
ceiling. Consequential requests additionally require a top-level approval
identity that matches the approval argument and a trusted approval policy that
accepts the exact request. The policy is responsible for binding the referenced
plan digest to the current plan and hard constraints; tool arguments cannot
replace that policy.

Handlers receive a cooperative cancellation token, wall-time deadline, and
transaction lifecycle capability. The
dispatcher charges one evaluation on entry, permits nested evaluations and
peak-memory reservations to be charged explicitly, and validates output against
the declared schema and output limit before constructing engine-owned
provenance. Budget exhaustion, timeout, cancellation, missing commit, unsafe
retry, unsupported capability, approval failure, invalid output, and handler
failure remain typed negative outcomes. Retryable handler failures may be
retried only up to the configured maximum of three retries, and consequential
retries additionally require the handler's idempotency declaration.

Completed request IDs are retained in a bounded replay ledger; replays return
the original result without invoking the handler again. A full ledger refuses
new work rather than silently permitting unbounded state. Wall-time enforcement
signals cancellation and returns a timeout even if a handler ignores the token,
so trusted handlers must check cancellation before consequential mutations and
use idempotent or transactionally rollback-safe engine operations. The
dispatcher does not provide process isolation for a malicious in-process
handler.

`ToolResult` is the typed result envelope for a later engine adapter. A
supported result must contain output and no failure; every non-supported result
must contain a typed failure whose request ID matches. Each result has a
deterministic `tool_result_<sha256>` ID over the complete canonical envelope,
so changing output, failure, provenance, or retained raw payload is detectable
on replay. `ToolResult.from_record()` rejects unknown fields, unknown result
versions, and tampered result IDs; `validate_request()` rejects results
replayed against a different or stale request digest.

`ToolProvenance` is engine-supplied trusted lineage, separate from provider
text. It retains the tool owner, tool identity, canonical request digest,
source digest, evidence ID, artifact ID, campaign ID, and an explicit UTC
observation timestamp where available. Provenance is either `canonical`,
`unverified`, or `unavailable`: canonical evidence requires an immutable
source digest, evidence ID, and timestamp; unavailable evidence cannot claim a
source. Failed and unsupported results retain bounded raw payloads in a
separate field, while trusted outcome and provenance fields remain unchanged.
Failed, unknown, unsupported, timed-out, cancelled, and refused states are
not converted into success or omitted.

## Versioning and limits

Request version 1 and result version 2 are fail-closed. `ToolRequest.from_record()`
and `ToolResult.from_record()` reject unknown schema versions and unknown
fields. The schema dialect is intentionally small and allows only strict
object/array/string/number/integer/boolean constructs with bounded lengths and
ranges. Tool input and raw result payloads are capped at 1 MiB; individual
tool output budgets are smaller and explicit.

This boundary does not add shell, Python, filesystem, network, provider,
model-session, or generic callback authority to the text model. Direct
CLI/Python callers remain independent of this boundary.

The deterministic adversarial corpus and its limitations are documented in
[`conversational-tool-boundary-adversarial.md`](../testing/conversational-tool-boundary-adversarial.md).

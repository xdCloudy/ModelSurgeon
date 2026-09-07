# Capability-scoped conversational tool schemas

Status: request schema version 1 and result envelope version 2 are implemented
as a schema-only boundary in `modelsurgeon.conversation.tools`. They define
what a conversational client may request and how an engine may publish a
grounded result; they do not implement an executor or make the conversational
product generally available.

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

`ToolCatalog.negotiate()` is the pre-execution gate. Unknown tool names return
`UNKNOWN`; known tools requested with the wrong capability return
`UNSUPPORTED`; unknown schema versions, tool identity drift, invalid input,
budget expansion, and missing approval return typed `REFUSED` failures. No
negotiation method invokes an implementation. `negotiate_record()` converts
untrusted JSON into a typed refusal and retains the request identity when it
is safe to do so.

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

This is a contract layer, not an execution implementation. It does not add
shell, Python, filesystem, network, provider, model-session, or generic
callback authority to the text model. Direct CLI/Python callers remain
independent of this boundary.

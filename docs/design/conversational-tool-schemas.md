# Capability-scoped conversational tool schemas

Status: version 1 is implemented as a schema-only boundary in
`modelsurgeon.conversation.tools`. It defines what a conversational client
may request; it does not implement an executor or make the conversational
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
must contain a typed failure whose request ID matches. Provenance retains the
tool owner, schema identity, and canonical request digest. Failed, unknown,
unsupported, timed-out, cancelled, and refused states are not converted into
success or omitted.

## Versioning and limits

Version 1 is fail-closed. `ToolRequest.from_record()` rejects unknown schema
versions and unknown fields. The schema dialect is intentionally small and
allows only strict object/array/string/number/integer/boolean constructs with
bounded lengths and ranges. Tool input is capped at 1 MiB before negotiation;
individual tool output budgets are smaller and explicit.

This is a contract layer, not an execution implementation. It does not add
shell, Python, filesystem, network, provider, model-session, or generic
callback authority to the text model. Direct CLI/Python callers remain
independent of this boundary.

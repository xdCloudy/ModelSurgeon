# Canonical conversational intent records

`modelsurgeon.conversation` defines the v2.1 record boundary between an
original user request and the stable objective contract. It records
interpretation without making free-form text an execution policy.

## Record contents

Each `IntentRecord` contains:

- the exact original request and its SHA-256 request identity;
- source spans whose offsets and text must match that request;
- normalized fields with explicit units, confidence, required status, and span
  references;
- ambiguity records and deterministic interpretation steps;
- provider, tool, source-revision, and evidence provenance; and
- either an exact canonical emitted-spec record or an explicit non-executable
  outcome (`clarification_required`, `unsupported`, or `refused`).

The `executable` outcome requires an emitted spec and rejects unresolved
required ambiguities. Non-executable outcomes cannot carry a spec. Confidence
is interpretation metadata only: it never proves that a mutation is safe or
relaxes the objective contract.

## Canonical and replay behavior

Records use schema version `1`, strict field sets, sorted unique references,
stable compact JSON serialization, and fail-closed unknown-version parsing.
The emitted spec receives its own digest, while the intent identity excludes
provider provenance so equivalent interpretations remain comparable across
providers. The canonical JSON still retains provenance and the original text
for audit and replay.

The record layer does not execute operations, select tensors, invent hard
constraints, or implement a text-model provider. Those responsibilities stay
with the compiler and deterministic execution boundaries described in the
conversational control-plane design.

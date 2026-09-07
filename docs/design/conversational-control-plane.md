# Conversational control-plane design

Status: the v2.1 canonical intent-record boundary, bounded objective-contract
compiler, policy evaluator, and equivalence/refusal corpus, the v2.2
replaceable provider boundary, the experimental v2.3 chat bootstrap, the
v2.6 bounded conversational tool boundary, and the v2.7 canonical campaign
state store are implemented and frozen in
`modelsurgeon.conversation` and `modelsurgeon.search`. The conversational
product remains experimental and incomplete. The [chat session bootstrap](chat-session-bootstrap.md),
the [v2.2 release boundary](../release/v2.2-provider-layer-boundary.md),
the [v2.6 release boundary](../release/v2.6-bounded-conversational-tool-boundary.md),
the [canonical campaign state design](conversational-campaign-state.md),
the [frozen v2.1 contract](conversational-intent-contract.md)
and [machine-readable release record](../research/v2.1-conversational-intent-contract-v1.json)
are normative for versioning and replay. This document is not a claim that the
full conversational product is currently implemented.

## Purpose

ModelSurgeon’s long-term product experience lets a user describe an optimization goal in plain English while keeping scientific execution deterministic, measurable and auditable. The conversational layer is therefore a control plane around ModelSurgeon, not a second optimizer and not an authority on tensor safety.

```text
User
  ↓
Text LLM / conversational harness
  ↓
validated, versioned OptimizationSpec
  ↓
ModelSurgeon deterministic execution engine
  ↕
learned Meta-Surgeon predictions
  ↓
canonical measured evidence + deployable artifacts
  ↓
Text LLM explanation
  ↓
User
```

The learned Meta-Surgeon and the text LLM are different models with different responsibilities:

- `Surgeon Tensors/` contains the learned Meta-Surgeon and its compatibility metadata.
- `Text LLM/` contains an optional, replaceable conversational model.
- `Models/` refers to user-provided target models and their derived artifacts.

## Responsibility split

| Layer | May do | Must not do |
| --- | --- | --- |
| User | State objectives, constraints, preferences and approvals. | Be assumed to have approved an unshown or changed plan. |
| Text LLM / harness | Interpret intent, identify missing fields, ask necessary questions, request typed operations and summarize evidence. | Select tensors, override policy, invent measurements, relax hard constraints or promote artifacts. |
| `OptimizationSpec` | Represent objectives, hard/soft constraints, budgets, allowed operations, hardware and provenance. | Contain hidden provider instructions or an implicit safety decision. |
| Deterministic engine | Inspect, construct capability spaces, mutate transactionally, evaluate, accept/reject, rollback, search, publish and record evidence. | Trust natural-language confidence as validation. |
| Learned Meta-Surgeon | Rank candidates or predict deltas/uncertainty inside declared policy and budgets. | Decide that an unmeasured surgery is safe or replace measured acceptance. |
| Evidence store | Persist measurements, failures, unsupported cells, lineage, approvals and artifacts. | Be overwritten by summaries or provider output. |

## Request compilation

v2.1 compiles a structured intent record into the existing stable v2 objective and constraint schema. It does not introduce a parallel optimization schema. The exact frozen boundary is documented in [conversational-intent-contract.md](conversational-intent-contract.md). A compiler result is one of:

- `executable`: all required fields validate and the spec is safe to submit;
- `clarification_required`: a bounded missing or ambiguous field prevents execution;
- `unsupported`: the request names an operation, metric, architecture or runtime outside declared capabilities;
- `refused`: the request conflicts with hard policy, cannot be represented safely or attempts to bypass a boundary.

Every result carries the original request, normalized interpretation, confidence/ambiguity information, schema/version identity, deterministic serialization and provenance. Confidence describes interpretation quality only; it is never evidence that a candidate surgery is safe.

Missing hard constraints are unresolved. The compiler never invents a threshold because a provider suggests one, and later negotiation always creates an explicit objective amendment with a visible diff and approval. Fields that the current objective contract cannot represent, such as budgets, allowed operations, or deployment targets, produce an explicit unsupported result rather than being dropped.

## Provider abstraction

The provider layer is replaceable and optional. A provider may be:

- a supported local GGUF text model;
- another user-selected local compatible runtime;
- an OpenAI-compatible endpoint;
- another supported hosted provider; or
- absent, when the user calls the direct CLI/Python APIs.

Providers implement the typed `TextModelProvider` interface documented in
[`text-model-provider-interface.md`](text-model-provider-interface.md) for
interpretation, clarification and explanation. They report capabilities,
limits, model identity, failures and provenance. Provider output is untrusted
input until it passes typed schema validation. Secrets are resolved outside
canonical evidence and are redacted from diagnostics, transcripts and
reproducibility bundles. `NullTextModelProvider` makes the no-LLM path
explicit; direct CLI/Python APIs do not depend on a provider.

## Tool boundary

The text model receives only typed, capability-scoped operations. The request
version-1/result-version-2 schema and trusted dispatcher contract is documented in
[`conversational-tool-schemas.md`](conversational-tool-schemas.md) and exposed
by `modelsurgeon.conversation`. Each operation declares:

- schema and version;
- read-only or consequential classification;
- required capability and approval scope;
- time, memory, evaluation and output budgets;
- deterministic request and result identifiers;
- cancellation/idempotency behavior; and
- supported, unsupported, failed and unknown result states.

Read-only inspection and evidence queries cannot mutate artifacts. Consequential
calls route through ModelSurgeon’s existing transaction, rollback, acceptance,
approval and provenance boundaries. The dispatcher validates the request and
approval policy before calling a trusted engine adapter, enforces the declared
budgets and retains deterministic replay results. There is no general shell,
Python, filesystem or network tool merely because a model requested it.

## Canonical campaign state

Conversation history is an ephemeral view. The authoritative state is the structured campaign store: objective/spec identity, plan version, lifecycle state, approval state, evidence cursor, accepted lineage, budgets and artifact identities. Summaries may reduce transcript cost but may not erase hard constraints, approvals, failures, negative evidence or provenance.

On pause, resume, reconnect, restart or new evidence, the engine validates state and detects stale context through the WAL-backed `CampaignStateStore`. A material plan or objective change creates a new versioned identity, visible diff and (when consequential) a new approval. Old evidence remains queryable and is not silently relabeled as current.

## Evidence-grounded explanations

The explanation layer consumes typed canonical evidence, not arbitrary files or free-form tool output. Claims must distinguish:

- measured result;
- model prediction or estimate;
- uncertainty or confidence interval;
- unsupported capability;
- failed or rolled-back experiment;
- missing evidence; and
- user-approved objective change.

Accepted, rejected, rolled-back, failed, uncertain and unsupported results remain in the scientific record. A final explanation must be able to identify why a candidate was selected, which hard constraints it satisfied, which alternatives were measured, and where evidence is unavailable.

## Approval and security model

Consequential or expensive operations require explicit approval over a specific plan/spec digest, capability scope and expiry. A material diff invalidates the prior approval. Prompt text, model metadata, provider output and tool output are potentially untrusted; trusted structured policy wins over them. Unknown or contradictory policy states fail closed.

The hardening work must cover prompt injection, instruction smuggling, malicious model metadata, forged measurements, secret requests, path attempts, malformed schemas, replayed identifiers and provider isolation. Security tests must retain failures and unresolved residual risk rather than claiming completion from a prompt or mock alone.

## Milestone sequence

1. **v2.1:** freeze request/intent/provenance compilation into `OptimizationSpec`.
2. **v2.2:** add replaceable providers and preserve no-LLM direct APIs.
3. **v2.3:** deliver the first inspect/preview/execute chat slice.
4. **v2.4–v2.5:** add necessary clarification, measured infeasibility and explicit amendments.
5. **v2.6:** formalize typed tools, budgets, transactions and result grounding.
6. **v2.7–v2.8:** add canonical state, recovery and evidence-grounded explanations.
7. **v2.9:** harden approvals, isolation and fail-closed policy precedence.
8. **v3.0:** integrate a local-first conversational product while preserving CLI/Python automation.

v3.1+ returns the primary research emphasis to scaling and improving the learned Meta-Surgeon. Any future integration must preserve this separation unless measured evidence justifies a change.

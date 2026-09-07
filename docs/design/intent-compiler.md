# Bounded conversational intent compiler

`modelsurgeon.search.intent_compiler` is the v2.1 boundary from a validated
`modelsurgeon.conversation.IntentRecord` to the existing
`modelsurgeon.search.objective_contract.ObjectiveContract` record.

The compiler consumes structured fields only. A hard constraint field has the
shape `{ "kind": "hard_constraint", "metric": ..., "direction": ..., 
"threshold": ..., "unit": ... }`; a preference is a soft objective with
`kind: "preference"` and the corresponding objective fields. The compiler
constructs the existing `HardConstraint`, `SoftObjective`, and
`ObjectiveContract` types, so contract validation and contract identity remain
owned by the search subsystem.

Compilation is deterministic and side-effect free. It does not call a provider,
run a search, execute a mutation, select tensors, inspect model weights, or
duplicate optimizer strategy logic. The returned `IntentCompilation` includes
the canonical contract record only for an executable result. Its diagnostics
retain the reason for every refusal or clarification.

The compiler refuses to reinterpret fields it cannot represent. In particular,
budgets, allowed operations, and deployment targets currently have no fields in
the v2 objective-contract record; they produce an explicit unsupported outcome
rather than being dropped or guessed. Missing hard constraints produce
`clarification_required`, and no threshold is inferred from confidence,
provider metadata, or natural-language text. An emitted spec, when present in
the intent record, must exactly match the contract compiled from typed fields.

The result schema is versioned in
`docs/research/v2.1-intent-compiler-v1.json`. Unknown or changed schemas should
be introduced with a new version and compatibility tests.

Before constructing the contract, the compiler emits deterministic conflict
witnesses. A contradictory hard-bound witness contains the smallest stable pair
of fields whose minimum exceeds its maximum; unrelated constraints are not
silently discarded. Repeated hard terms that the v1 contract cannot represent
are refused with a `conflicting-hard-constraints` diagnostic. Repeated soft
preferences for one metric are emitted as
`ambiguous-preference-ordering` and remain non-executable until the user
selects one. The witness retains all involved field IDs and source spans, and
the policy layer adds the intent's provenance references.

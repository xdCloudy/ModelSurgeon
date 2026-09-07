# Conversational intent policy evaluation

`modelsurgeon.search.intent_policy` is the safety policy boundary after the
typed intent compiler. It does not create an alternate optimizer or emit a
second specification. The compiler remains the only source of an
`ObjectiveContract`; this layer decides whether that compiled result is safe to
submit.

## Outcome precedence

The evaluator is deterministic and fail-closed. When multiple conditions are
present it chooses, from safest refusal to least restrictive:

1. `refused` for contradictory hard constraints, invalid declarations, or
   other hard-policy conflicts;
2. `unsupported` for metrics or operations outside the declared contract;
3. `clarification_required` for missing hard constraints, unresolved required
   ambiguity, low-confidence required fields, or conflicting preferences; and
4. `executable` only when the compiler emitted a valid contract and none of the
   preceding conditions applies.

No non-executable decision carries an objective contract. A confidence score is
classified as high, medium, or low for interpretation review only; it never
proves safety, measured quality, or constraint satisfaction.

## Evidence and replay

Every diagnostic links to field/source-span IDs and provenance references.
Confidence and ambiguity assessments are sorted, canonical, and retained with
the exact compilation result. The decision ID is derived from canonical JSON,
so equivalent inputs produce replayable decisions and changed provenance or
diagnostics are visible rather than silently collapsed.

# Pareto trade-off and final-selection explanations

`modelsurgeon.explain.pareto_selection` is the read-only explanation boundary
for a final candidate choice. It reuses the measured archive and conservative
frontier from `pareto_alternatives`, v2.0 decision replay, canonical evidence
query identities, and the immutable objective-amendment identities.

## Selection contract

The projection renders hard constraints first. A candidate is selectable only
when it is measured, complete, accepted, and on the measured feasible Pareto
frontier. Failed, unsupported, unknown, inconclusive, rejected, rolled-back,
incomplete, constraint-violating, and dominated records remain visible as
alternatives but cannot be selected.

Objective values are recomputed from the declared `ObjectiveContract` using
the conservative bound for each measured interval. Weighted objectives use
the contract's weighted score. Lexicographic and Pareto contracts use a
deterministic normalized objective-vector order, with candidate ID as the
stable final tie-break. The generated v2.0 decision-replay record must produce
the same selected ID; a caller-supplied selection or final-selection evidence
that disagrees is rejected.

Overlapping intervals are retained as uncertainty and do not create a
dominance claim. Exact equality is retained as a tie. An empty feasible
frontier produces `no_feasible_frontier` and no selected candidate; it never
manufactures a rationale or relaxes a hard constraint.

## Identity and approval context

Every report carries the objective contract ID, evidence archive ID, source
model digest, evidence IDs, frontier IDs, candidate evidence IDs, replay
digest, and the approval context from `FeasibilityProvenance`. When an
approved/applied material amendment is supplied, the report additionally
records the original objective ID, effective amended objective ID, amendment
ID, diff ID, amendment status, approval ID, and downstream campaign identity.
The amended objective is a new identity; the original evidence archive is not
silently rewritten.

An optional `EvidenceQueryResponse` is recorded by query ID, response digest,
snapshot identity, status, and retained evidence IDs. Query output is context
only: it cannot overwrite the typed Pareto archive or become a second
measurement authority.

## Replay and rendering

`direct_report()`, `chat_text()`, and the HTML renderer derive from one typed
result. `replay_pareto_selection_explanation` rebuilds the projection and can
assert byte-stable direct-report parity. The result ID is a SHA-256 digest of
the canonical report without its result ID. Resource limits fail closed and
never truncate alternatives.

The v2.8 release treats the typed direct report as canonical: chat and HTML
forms are presentation projections, and replay must reproduce the same
selection and report. A no-feasible-frontier or unmeasured candidate remains
unknown/non-selectable; the explanation does not claim optimizer optimality or
proof. Cross-model generalization and causal attribution require separate
held-out/intervention evidence and are deferred. See the [v2.8 release
design](evidence-grounding-release.md).

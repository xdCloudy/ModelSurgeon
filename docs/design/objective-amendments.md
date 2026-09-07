# Immutable objective amendments

`modelsurgeon.search.objective_amendments` is the control-plane boundary for
changing an objective after a measured feasibility explanation. An amendment
is an immutable proposal, not an edit to the original `ObjectiveContract`.

## Contract

Each `ObjectiveAmendment` retains:

- the original objective contract and the proposed contract;
- the #458 `FeasibilityExplanation`, archive identity, source-model digest, and
  all prior evidence IDs;
- the rationale and a deterministic `ObjectiveAmendmentDiff`;
- an `ApprovalRequest` from the v2 approval/provenance package, including the
  exact plan digest, diff ID, expiry, operator, and approval scope;
- the effective spec identity and optional parent/session/run campaign
  linkage.

The approval scope always includes `objective_amendment` and every changed
canonical path. A hard-constraint change also requires the explicit
`hard_constraints` scope marker. Provider wording, a replay, or a convenience
operation cannot mutate a hard constraint.

Canonical reordering or an otherwise identical contract is a non-material
diff. It keeps the original effective spec and campaign identity. Any actual
contract change is material, including a hard-constraint change. Applying a
material amendment requires an approved, unexpired decision and a current
objective that exactly matches the retained original. It derives a new
campaign ID from the session, run, source model, effective spec identity, and
spec digest. The original campaign and evidence archive remain unchanged and
queryable.

## Lifecycle and replay

The lifecycle is `pending`, `approved`, `rejected`, `expired`, `cancelled`, or
`applied`. Approval, rejection, expiry, and cancellation return new frozen
records. `ObjectiveAmendmentHistory` is append-only; it retains every state
transition and rejects conflicting or out-of-order replays. Applying an already
applied record through `replay_objective_amendment` returns the same immutable
application record. A proposal against a different current objective is stale
and fails closed.

Typical use:

```python
proposal = propose_objective_amendment(
    original_contract,
    proposed_contract,
    rationale="measured near-miss evidence supports this explicit trade-off",
    evidence=feasibility_explanation,
    operator_id="operator-alice",
    requested_at="2026-09-07T10:00:00+00:00",
    expires_at="2026-09-07T11:00:00+00:00",
    parent_campaign_id="campaign_original",
    session_id="session-1",
    run_id="run-1",
)
approved = approve_objective_amendment(
    proposal,
    operator_id="operator-alice",
    decided_at="2026-09-07T10:05:00+00:00",
)
application = apply_objective_amendment(
    approved,
    current_objective=original_contract,
    current_campaign_id="campaign_original",
    applied_at="2026-09-07T10:06:00+00:00",
)
```

## Compatibility guarantee

The amendment schema is versioned. Existing objective contracts, campaign
states, approval records, and feasibility evidence are read and retained
without mutation. New amendment fields are additive within schema version 1;
unknown amendment schema versions must be rejected. The public API is exported
from `modelsurgeon.search`, and the direct history/inspection APIs do not
depend on conversational transcripts or provider state.

# Scoped approvals

The direct optimize orchestrator and the chat execution adapter share the same
approval lifecycle. An approval is bound to the canonical plan ID and digest,
the content-addressed material-diff ID, operator identity/context, required
capability scope, reuse policy, and expiry. Approval records and audit evidence
are persisted with the run or campaign; provider text and secrets are never
part of the evidence contract.

## Lifecycle

An approval is issued only for the exact plan and diff being executed. The
trusted execution boundary records a decision and consumes the approval before
running the workflow. `reusable` approvals may be consumed by later resumable
invocations while their expiry and plan binding remain valid. `one_time`
approvals are consumed once and require a new decision for another invocation.
Expired, malformed, actor-mismatched, overbroad, or stale approvals fail
closed. A material plan change produces a deterministic diff ID and changed
paths; the prior approval cannot authorize the changed plan. Chat replanning
creates a new campaign approval and records the reapproval transition.

The direct CLI accepts an expiry and per-code reuse policy:

```text
--approval-expires-at 2026-09-08T18:00:00+00:00
--approval-reuse plan_review=one_time
--approval-reuse artifact_write=reusable
```

Chat exposes the same options and additionally binds the outer
`execute_approved_plan` approval to the campaign's exact plan metadata. This
keeps direct and chat execution equivalent at the trusted boundary while
allowing chat to retain the same evidence in its canonical campaign store.

## Immutable audit evidence

Every issue, decision, consumption, expiry, and chat projection is represented
by a content-addressed, chained `ApprovalAuditRecord`. Detail text is redacted
for assignment-style secrets and bearer tokens before hashing. Campaign state
stores the plan ID/digest, diff ID, scope, reuse policy, usage count, and the
immutable audit projection, so a transcript or provider summary cannot rewrite
approval history.

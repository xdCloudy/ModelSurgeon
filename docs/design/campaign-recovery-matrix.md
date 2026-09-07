# Bounded campaign recovery matrix

Issue #473 closes the v2.7 evidence gap around stateful conversational
campaign recovery. The machine-readable matrix is
[`campaign_recovery_matrix_v1.json`](../../tests/fixtures/campaign_recovery_matrix_v1.json),
and the release/audit record is
[`v2.7-campaign-recovery-v1.json`](../research/v2.7-campaign-recovery-v1.json).
The milestone-level closure and dependency reconciliation are in the [v2.7
stateful campaign release boundary](../release/v2.7-stateful-campaigns-boundary.md)
and [release manifest](../research/v2.7-stateful-campaign-release-v1.json).

## Authority boundary

The canonical `CampaignStateStore` is the only recovery authority. Direct
reconnect reads the committed WAL-backed state by campaign and session ID.
Chat reconnect first reads the same state and evidence, then validates a
bounded summary against that exact snapshot. Transcript text, omitted history,
provider memory, and a successful UI reconnect cannot advance a campaign.

Every matrix cell compares the canonical state digest, evidence cursor and
retained evidence/artifact references, next action, hard constraints, resource
budget, source-model digest, and provenance through both recovery paths.

## Matrix

| Cell | Boundary exercised | Expected result | Next action |
| --- | --- | --- | --- |
| `pause_resume` | durable operator pause and approval-valid resume | supported | execute next stage |
| `cancel` | terminal cancellation | cancelled | stop |
| `reconnect` | identity-only reconnect | supported | start |
| `restart` | pause followed by restart command | supported | execute next stage |
| `stale_plan` | evidence changes after a plan snapshot | failed and stale error retained | rebuild stale plan |
| `expired_approval` | approval expires before resume | failed; approval becomes expired | await fresh approval |
| `transcript_loss` | bounded summary omits old transcript entries | supported with loss marker | start |
| `partial_evidence` | unknown/inconclusive evidence | unknown and retained | review evidence |
| `fault_transition_after_sql` | injected failure after transition SQL | rolled back | start |
| `fault_evidence_after_sql` | injected failure after evidence SQL | rolled back | start |
| `process_restart` | separate Python process reopens the store | supported | execute next stage |
| `direct_chat_recovery` | direct and summary recovery comparison | supported | start |

Failed, unsupported, unknown, cancelled, and inconclusive results are data in
the boundary, not missing rows. A failed operation must leave its evidence or
failure reason inspectable without changing the source-model digest, hard
constraints, budget, or provenance.

## Guarantees and limits

- Resume requires an active approval bound to the exact current spec digest;
  expiry is recorded before the resume is refused.
- Hard constraints are copied from the canonical spec and are never recovered
  from transcript text or relaxed by a stale plan.
- SQLite writes are atomic at bounded transition/evidence checkpoints; a
  process fault cannot publish a partial transition or partial evidence row.
- The same canonical inputs produce the same identifiers, snapshot, and next
  action. Lifecycle-only restart does not make a semantically equal plan stale.
- Evidence cursors are monotonic and retained evidence is append-only;
  unsupported, failed, unknown, and inconclusive records remain visible.
- Resource ceilings and provenance are part of the compared recovery snapshot.
- The subprocess test proves store reopen behavior. It does not claim hostile
  process containment, live-provider correctness, model-quality recovery, or
  UI reconnect correctness. The store's lock is process-local; multi-writer
  concurrency, distributed coordination and cross-host recovery are unsupported.

Run the focused audit and tests with:

```text
uv run python tools/audit_v27_campaign_recovery.py
uv run pytest tests/test_campaign_recovery.py tests/test_v27_campaign_recovery.py -q
```

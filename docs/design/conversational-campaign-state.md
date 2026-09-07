# Canonical conversational campaign state

The conversational campaign store is the authoritative reconnect boundary for
stateful optimization campaigns. It is implemented by
`modelsurgeon.conversation.CampaignStateStore` and is intentionally separate
from the provider transcript.

## Authority and linkage

Each record links a deterministic `campaign_id` to one `session_id`, one v2.0
`run_id`, the immutable source-model digest, the exact current
`OptimizationSpec`, policy state, approval state, provider identity/context,
resource ceilings, lifecycle, outcome, and an evidence cursor. The campaign ID
is derived from the immutable linkage inputs when the caller does not provide
one.

The store never accepts transcript text, summaries, provider memory, secrets,
or untrusted tool payloads as state. Reconnect takes only the campaign and
session identities. A process restart therefore loads the latest committed
canonical record without replaying chat, and losing a transcript does not lose
the campaign.

## Versioned transitions

State is stored in SQLite with WAL journaling and a checksummed schema
migration. Every update uses an expected state version and an atomic write
transaction. A successful transition increments the version, derives a stable
transition ID from the exact next state and provenance, and retains the
provenance digest. Stale writers fail closed. Material spec changes are visible
transitions and automatically return approval to `pending` unless a new
approval bound to the new spec digest is supplied.

The source-model digest is not an update field, so campaign transitions cannot
silently retarget source data. Hard constraints are retained inside the exact
spec record and are normalized into deterministic order.

Lifecycle commands are explicit and replayable: `pause` preserves the latest
committed stage, `resume` requires a still-active approval, `cancel` is
terminal, `reconnect` is a read-only identity lookup, and `restart` marks an
unfinished paused campaign runnable again. Repeating a command with the same
operation ID returns the committed snapshot rather than writing a duplicate
transition. Invalid terminal transitions and stale versions fail closed.

## Evidence retention

Evidence is append-only and cursor-addressed. Supported, unsupported, failed,
unknown, and explicitly inconclusive records remain queryable. Evidence IDs are
immutable and duplicate retries are idempotent only when the complete evidence
payload is identical; an ID collision with different content fails closed.

The canonical state schema is versioned independently from the SQLite schema.
Unknown schema versions, migration checksum drift, malformed JSON, state
digest/version mismatches, and corrupt databases are rejected rather than
silently migrated or repaired.

The execution adapter coordinates this store with the existing deterministic
orchestrator. Canonical campaign state is stored at the adapter's `state_path`;
the orchestrator's stage cursor is kept in the sibling
`<state stem>.execution.json` file. Recovery validates session, spec, plan,
source-model, approval, and budget linkage before allowing the stage cursor to
continue, so reconnecting cannot duplicate completed mutation or evaluation
work.

## Boundary example

```python
from modelsurgeon.conversation import CampaignLifecycle, CampaignStateStore

with CampaignStateStore("campaign-state.sqlite3") as store:
    # campaign_id and session_id come from the trusted campaign linkage.
    state = store.reconnect(campaign_id, session_id)
    state = store.transition(
        campaign_id,
        expected_version=state.state_version,
        kind="paused",
        lifecycle=CampaignLifecycle.PAUSED,
        provenance={"record_type": "operator_transition", "source": "trusted_engine"},
    )
```

The store persists campaign state and canonical evidence only. It does not
execute model operations, choose tensors, relax constraints, or turn provider
output into evidence.

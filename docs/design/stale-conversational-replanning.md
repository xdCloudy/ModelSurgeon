# Stale conversational context and plan lineage

Conversational text is a client of the campaign state store, never a second
source of truth. A replan starts with a trusted `ReplanContext` snapshot. The
snapshot binds the campaign spec, retained evidence IDs and digest, provider
context, resource budget, current plan, approval identity, and transaction ID.
This is part of the closed [v2.7 stateful campaign release
boundary](../release/v2.7-stateful-campaigns-boundary.md).

`detect_stale_context` compares two snapshots in canonical field order. A
restart or lifecycle-only transition is not stale; a changed spec, evidence
set, provider, resource budget, plan, approval, or transaction is reported by
its stable reason. A stale proposal is rejected and must be rebuilt from the
new canonical state. Transcript text cannot refresh a snapshot.

Replans have two outcomes:

* A no-op keeps the existing campaign and approval. New evidence may still be
  retained on the parent campaign, but unchanged plan content is not given a
  new identity.
* A material replan emits a versioned `ReplanDiff`, requires a fresh
  `campaign_replan` approval request and decision, and creates a deterministic
  child campaign. The child records its parent, diff, transaction, provider
  and budget context, and preserved evidence IDs. Its evidence archive starts
  empty: parent evidence remains queryable only through the parent campaign
  and is never relabeled current.

Objective changes must be represented by an approved #460 immutable objective
amendment. Cancelled, expired, rejected, pending, or stale amendments cannot
gate a replan. The amendment approval is provenance for the new proposal; it
does not substitute for the child plan's fresh approval.

Replaying the exact proposal is idempotent while its parent is still valid.
Replaying a proposal after a parent state change, or after cancellation,
fails closed. Child and parent campaigns can be inspected with
`CampaignStateStore.lineage`, `children`, and `evidence`.

The public API is exported from `modelsurgeon.conversation`:

```python
from modelsurgeon.conversation import (
    CampaignStateStore,
    build_replan_approval_request,
    build_replan_proposal,
    context_from_campaign,
    apply_replan,
)
```

See `docs/examples/stale_replanning.py` for a deterministic tiny fixture.

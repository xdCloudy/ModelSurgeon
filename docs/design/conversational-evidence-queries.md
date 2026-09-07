# Canonical conversational evidence queries

The evidence query path is a typed, read-only projection over canonical
campaign state and retained campaign evidence. It is intended for a
conversation layer that needs to explain a result without becoming a second
execution or measurement authority.

## Query boundary

`EvidenceQuery` accepts a campaign identity, optional retained evidence IDs,
an optional outcome filter, an allowlisted field selection, an optional
snapshot/state identity, and a record limit. Its only access mode is
`read_only`. It does not accept paths, file names, commands, prompts,
transcripts, provider output, callbacks, or arbitrary JSON selectors. The
query ID is a SHA-256 identity of the canonical query payload, so replaying
the same request produces the same ID.

The query engine can read an existing `CampaignStateStore`, but it only calls
the store's read APIs. It never creates, transitions, appends evidence, or
opens an arbitrary user-supplied file. A provider may receive the resulting
typed response as context, but provider text cannot populate or overwrite any
trusted field.

## Source precedence and joins

The canonical source precedence is:

1. `campaign_state` — campaign identity, source model, exact spec, hard
   constraints, lifecycle, state outcome, cursor, and resource budget.
2. `campaign_evidence` — immutable retained evidence, source digest, engine
   outcome, detail, artifact reference, negative/inconclusive marker, and
   engine provenance.
3. `artifact_reference` — an immutable artifact digest only. The query path
   does not open or infer from an artifact.

Every response row joins the campaign ID, state version, evidence ID, source
digest, and safe provenance references. A source digest or artifact digest is a
reference, not a measurement. Unknown, missing, or unavailable data remains
explicit.

## Outcomes and uncertainty

Rows expose a conversational disposition of `accepted`, `rejected`,
`rolled_back`, `unsupported`, `failed`, `unknown`, or `inconclusive`.
Disposition comes from an engine-owned `provenance.decision` when present;
otherwise the retained `CampaignOutcome` and its inconclusive flag are used.
The original engine outcome is retained as `source_outcome`.

Measurements are read only from the trusted `provenance.measurements` object.
Each value has a metric identity and optional unit and uncertainty fields:
lower/upper bounds, confidence, standard error, and sample count. The query
path never fills a missing value, converts a prediction to a measurement, or
calculates an interval. Missing measurement/uncertainty/timestamp fields are
listed in `missing_fields`; fields that are outside the canonical projection
are listed in `unavailable_fields`.

## Snapshots, tampering, and staleness

`EvidenceSnapshot` captures the exact validated state and evidence records,
plus state, evidence, snapshot, and snapshot-content digests. Snapshot IDs are
deterministic. Loading a snapshot validates every nested campaign/evidence
digest and rejects altered, missing, unknown, or unsupported fields with
`EvidenceQueryIntegrityError`.

`EvidenceQueryEngine.query_current()` compares the snapshot's state and
evidence digests with the current store. Any append, transition, or other
canonical change causes `EvidenceQueryStaleError`; callers must obtain a new
snapshot and issue a new query. A query can also bind directly to an expected
snapshot ID and state digest.

## Resource limits

The boundary limits one request to 64 KiB, 256 records, 32 selected fields,
64 provenance references per row, and 512 KiB of result JSON. A caller may
choose smaller limits. Exceeding a limit is a typed
`EvidenceQueryResourceError`; the engine does not truncate evidence silently.

## Replay and direct-report parity

`EvidenceQueryResponse.canonical_json()` is the replay artifact. With the same
snapshot and query it is byte-for-byte stable, including query ID, snapshot
identity, outcome ordering, provenance joins, missing markers, and resource
usage. `direct_evidence_report()` is the structured direct API projection used
by parity tests; conversational callers consume the same response envelope.

Example:

```python
from modelsurgeon.conversation import (
    EvidenceQuery,
    EvidenceQueryEngine,
    EvidenceSnapshot,
)

snapshot = EvidenceSnapshot.from_store(store, campaign_id)
query = EvidenceQuery(
    campaign_id,
    fields=("artifact_digest", "detail", "measurements", "outcome", "provenance_refs"),
    expected_snapshot_id=snapshot.snapshot_id,
    expected_state_digest=snapshot.state_digest,
)
response = EvidenceQueryEngine(snapshot).query(query)
```

The response is evidence, not an explanation. A text model may summarize the
response, but every factual claim must retain the relevant evidence ID or be
marked unavailable.

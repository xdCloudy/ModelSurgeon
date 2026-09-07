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

## Negative-evidence explanations

`explain_negative_evidence(snapshot, query)` builds a deterministic
`NegativeEvidenceReport` from the same query response. It adds the known
canonical mutation, evaluation, and rollback records to each row, then
renders measured metrics with their value, unit, comparison direction,
threshold, and retained uncertainty. The explanation keeps the evidence ID,
source digest, mutation/evaluation/rollback IDs, parent links, and artifact
digests together so a factual statement can be replayed from the snapshot.

Negative outcomes are deliberately not interchangeable:

| Outcome | Meaning | What the explanation may say |
| --- | --- | --- |
| `rejected` | Measured evidence did not satisfy a hard qualification rule. | Name the observed metric, direction, threshold, uncertainty, and failed reason. |
| `rolled_back` | The candidate state was reverted after the lifecycle decision. | State the rollback and its lineage; rollback never implies acceptance. |
| `failed` | Execution or evaluation failed. | Preserve the failure detail; do not call it unsupported or untried. |
| `unsupported` | The requested operation is outside the verified capability boundary. | Say it was unsupported, distinct from failed, without inventing measurements. |
| `unknown` | The retained record cannot establish the outcome. | Emit an explicit unknown/incomplete qualification rather than rejection. |
| `inconclusive` | Evidence exists but cannot support a qualifying decision. | Preserve the measurements and uncertainty while marking the decision inconclusive. |

Missing mutation/evaluation/rollback records, metrics, or qualification
reasons are listed in `unknown_fields` and make the explanation
`incomplete`. They are never reconstructed from provider text, predictions,
or absent fields. `NegativeEvidenceReport.from_record()` validates the
deterministic report and explanation IDs for exact replay.

Example:

```python
from modelsurgeon.conversation import EvidenceQuery
from modelsurgeon.explain import NegativeEvidenceReport, explain_negative_evidence

report = explain_negative_evidence(
    snapshot,
    EvidenceQuery(snapshot.campaign.campaign_id),
)
print(report.to_text())
assert report.canonical_json() == NegativeEvidenceReport.from_record(
    report.to_record()
).canonical_json()
```

The response is evidence, not an explanation. A text model may summarize the
response, but every factual claim must retain the relevant evidence ID or be
marked unavailable.

The v2.8 release closes this path as the canonical direct/report authority.
Claim, negative-evidence, and measured Pareto explanations consume typed
reports from this boundary; provider output, transcript text, predictions, and
free-form narrative cannot add measurements or replace the report. See the
[v2.8 release design](evidence-grounding-release.md) and
[release manifest](../research/v2.8-evidence-grounding-release-v1.json).

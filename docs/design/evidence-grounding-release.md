# v2.8 evidence-grounding release design

The v2.8 release closes a bounded evidence-grounding milestone. Its supported
surface is the canonical path:

```text
CampaignStateStore → EvidenceSnapshot → EvidenceQuery → typed report → explanation
```

The five merged dependencies are complementary: #475 supplies the typed query
and direct report; #476 renders every query row with source identity and
measurement status; #477 explains negative and incomplete outcomes; #478
explains measured Pareto trade-offs and final selection; and #479 measures
factuality on a fixed provider-independent corpus.

## Canonical authority and reconciliation

`direct_evidence_report()` and `EvidenceQueryEngine.query()` must be byte
identical for the same snapshot and query. The claim renderer accepts only a
typed `EvidenceQueryResponse`, validates its snapshot/query projection, and
retains every row. Negative evidence uses the same response for its direct and
explanation entry points. Pareto direct, chat, and HTML forms derive from one
typed `ParetoSelectionExplanation`; replay must reproduce the direct report
and selected candidate.

Provider output, conversation transcripts, free-form narrative, predicted
values, and UI presentation are not evidence authorities. A source model is
immutable at this boundary. Tampered digests, stale snapshots, unavailable
fields, resource overruns, selection drift, and unsupported inputs fail closed.

## Claim policy

Every factual claim is linked to an evidence ID and source digest, or is
explicitly marked unavailable. Measured values retain their units and typed
uncertainty. An accepted decision without a canonical measurement is
prediction-only; it is never rendered as measured. Rejected, rolled-back,
failed, unsupported, unknown, and inconclusive outcomes remain visible and
distinct.

The release makes no claim of optimizer optimality, optimizer proof, general
correctness proof, live provider quality, external model quality, or any
measurement absent from the canonical evidence envelope.

## Factuality gate

The checked-in v2.8 corpus has six cases, three bounded replay seeds per case,
three methods, and 54 retained cells. The grounded renderer and conservative
template baseline must reach 1.0 source precision/recall, source-ID
correctness, negative-evidence coverage, uncertainty disclosure and
calibration, and deterministic IDs. Measured-vs-predicted confusion,
unsupported-claim rate, and critical escapes must be zero. The unconstrained
text control remains a retained negative control and is never shippable.

Failed and inconclusive cells are retained in the study record. Passing this
gate is evidence for this corpus and bounded renderer, not broad language-model
factuality or live provider quality.

## Supported, unsupported, and unknown

Supported means the claim is a projection of retained canonical evidence in a
verified schema and within the tested resource boundary. Unsupported means the
operation or claim is outside that verified capability boundary. Unknown means
the relevant retained record is incomplete or has no observation; unknown is
not a synonym for rejection and never becomes a positive claim.

Deferred explanation types include optimizer optimality/proof, live provider or
external-model quality, unmeasured hardware or latency, unbounded free-form
narrative, cross-model generalization without held-out evidence, and causal
attribution without a canonical intervention record.

## Duplicate scope

Issue #420 remains the offline explorer/report-projection authority. Issue #411
remains the signed immutable evidence-package authority. Issue #429 remains
the measured optimizer benchmark/reference-artifact authority. v2.8 consumes
those records for bounded explanations; it does not replace any of them.

The versioned boundary and machine-readable manifest are
[`v2.8-evidence-grounding-boundary.md`](../release/v2.8-evidence-grounding-boundary.md)
and
[`v2.8-evidence-grounding-release-v1.json`](../research/v2.8-evidence-grounding-release-v1.json).

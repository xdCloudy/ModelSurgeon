# Claim-to-evidence explanations

`modelsurgeon.explain.claim_evidence` is the presentation boundary for the
typed canonical `EvidenceQueryResponse` from the conversational evidence-query
path. It accepts that envelope only. A mapping, provider answer, transcript,
query, snapshot, or ad-hoc evidence record is rejected.

The renderer validates the nested query and snapshot digests and recomputes the
canonical query report before rendering. This makes tampered rows, changed
outcomes, altered measurements, and untrusted narrative input fail closed.
Every response row becomes one explanation block in response order, including
negative and inconclusive evidence.

## Vocabulary

Each block has both a visible claim status and a measurement status:

- `MEASURED` means the canonical row contains one or more typed measurements.
- `PREDICTED` means an accepted decision has no canonical measurement. It is
  explicitly prediction/decision-only and never supplies a value.
- `REJECTED`, `ROLLBACK`, `UNSUPPORTED`, `FAILED`, `UNKNOWN`, and
  `INCONCLUSIVE` remain distinct negative or non-final outcomes.

Metric names, values, units, and uncertainty bounds are copied from the typed
query record. Evidence IDs, source and artifact digests, provenance references,
source outcome, detail, state version, and record digest remain attached to the
block. Missing or unavailable factual fields are retained in the block and
marked `UNAVAILABLE` in the text rendering.

The structured explanation also preserves the campaign execution budget, query
record limit, query resource usage, renderer bounds, and renderer usage. Limits
are hard: records, metrics, provenance references, and output bytes are never
silently truncated.

## Determinism and parity

`ClaimEvidenceExplanation.canonical_json()` and `render_text()` are stable for
the same canonical response. The explanation stores the source response digest,
snapshot identity, query identity, and source precedence, so a direct report
and a conversationally obtained response can be compared before rendering.

```python
from modelsurgeon.explain import render_claim_evidence

explanation = render_claim_evidence(canonical_response)
print(explanation.render_text())
```

The renderer is an explanation layer, not a measurement authority. Only
`EvidenceMeasurement` values present in the canonical query envelope can appear
as measurements.

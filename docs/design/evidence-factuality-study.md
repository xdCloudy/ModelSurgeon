# v2.8 evidence-grounding coverage and factuality study

This is a bounded research protocol for deciding whether conversational optimization explanations are grounded in canonical ModelSurgeon evidence. It evaluates the merged #475 evidence snapshots, #476 claim renderer, #477 negative-evidence projection, and #478 measured Pareto-selection explanations as one end-to-end boundary.

## Protocol

The checked-in corpus contains six provider-independent campaign narratives: three fixed fixtures and three held-out narratives. Each case binds a sorted set of canonical evidence IDs to gold claim labels. The evidence includes measured, prediction-only, rejected, rolled-back, unsupported, unknown, and inconclusive outcomes. The same immutable source-model digest is used for every row, and the runner constructs an `EvidenceSnapshot`, executes a typed read-only `EvidenceQuery`, and calls the production claim renderer.

Every method/case/seed cell is retained. Seeds are `[0, 1, 2]`; they are a replay bound, not a source of provider randomness. The CPU budget is fixed at 10 seconds for the complete local replay and the output budget is 512 KiB. No live provider or model-quality benchmark is part of this decision.

The methods are:

- `grounded_renderer`: the shippable end-to-end snapshot/query/renderer path.
- `template_only`: a conservative structured baseline that emits the gold status and canonical source ID through fixed templates.
- `unconstrained_text`: a free-text negative control that deliberately has no source IDs and presents every narrative as measured. It is never shippable, even if a future corpus accidentally makes its aggregate score look acceptable.

## Metrics and fail-closed policy

Claim-source precision and recall count exact `(claim ID, source ID)` pairs. Source-ID correctness requires every emitted claim to bind to exactly its gold source set. Measured-vs-predicted confusion counts status mismatches, including presenting prediction-only evidence as measured. Negative-evidence coverage requires rejected, rolled-back, unsupported, unknown, and inconclusive rows to remain present with their gold status. Uncertainty disclosure measures required intervals/confidence being shown; calibration measures exact disclosure decisions across all claims. Unsupported-claim rate counts source mismatches, unknown claims, and measured claims without a canonical measured source. Deterministic IDs are checked by replaying each bounded cell.

The release thresholds are strict: precision, recall, source IDs, negative coverage, uncertainty disclosure/calibration, and deterministic IDs must be 1.0; measured-vs-predicted confusion and unsupported-claim rate must be 0.0. A critical escape fails closed for shippability. The study still completes the bounded negative-control replay so failures and inconclusive cells are retained rather than hidden.

## Decision

The grounded renderer and template-only baseline meet the declared thresholds on the checked-in corpus. The unconstrained-text control produces source-ID escapes, measured-vs-predicted confusion, and unsupported claims, so it remains explicitly non-shippable. The result supports shipping the bounded grounded renderer only; it does not claim general model factuality, provider quality, or coverage outside the fixed corpus.

Replay the protocol with:

```text
uv run python tools/run_evidence_factuality_study.py
uv run python tools/audit_evidence_factuality_study.py
```

The complete retained run is available with `--json`; the checked-in fixture is [evidence_factuality_study_v1.json](../../tests/fixtures/evidence_factuality_study_v1.json), and the release-threshold record is [v2.8-evidence-factuality-study-v1.json](../research/v2.8-evidence-factuality-study-v1.json).

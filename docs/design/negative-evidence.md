# Negative evidence and uncertainty boundary

Negative evidence is a first-class part of an explanation, not a failed attempt
to produce a positive story. The canonical source is the typed
`EvidenceQueryResponse` produced by `EvidenceQueryEngine`; the explanation
projection may only copy fields that are present in that response.

## Outcome vocabulary

The release keeps these outcomes distinct:

| Outcome | Meaning | Permitted claim |
| --- | --- | --- |
| `rejected` | Measured evidence did not satisfy a hard qualification rule. | Report the observed metric, direction, threshold, uncertainty, and reason. |
| `rolled_back` | The candidate was reverted after the lifecycle decision. | Report rollback lineage; rollback is never acceptance. |
| `failed` | Execution or evaluation failed. | Preserve the failure detail; do not call it unsupported or untried. |
| `unsupported` | The request is outside the verified capability boundary. | Say unsupported without inventing a measurement. |
| `unknown` | Retained evidence cannot establish a classification. | Mark the missing qualification explicitly; do not infer success or rejection. |
| `inconclusive` | Evidence exists but cannot support a qualifying decision. | Retain the measurements and uncertainty while marking the decision inconclusive. |

Missing mutation, evaluation, rollback, metric, qualification, lineage, or
other required fields remain in `unknown_fields`/`unavailable_fields`. They are
not reconstructed from provider output, transcript text, predictions, or
absence. This distinction lets a caller explain what happened without
laundering an unavailable measurement into a fact.

`direct_negative_evidence_report` and `explain_negative_evidence` use the same
canonical query projection. `NegativeEvidenceReport.from_record()` and the
stable report ID make replay and direct/report parity checkable.

## Release limitation

This boundary supports bounded, source-linked explanations of retained
evidence. It does not provide optimizer optimality or correctness proofs,
live-provider quality claims, causal attribution without a canonical
intervention, or measurements that the evidence envelope does not contain.

# v2 autonomous benchmark publication contract

`modelsurgeon.evaluation.v2_benchmark` is the release-candidate boundary for
the v2 autonomous optimizer benchmark. It records a preregistered matrix and
validates evidence; it does not download models, execute a live campaign, or
infer results from missing cells.

## Protocol

`V2BenchmarkProtocol` requires exact model checkpoints and revisions, at least
two families and sizes, held-out dataset/task/split identities and licenses,
autonomous and baseline methods, bounded hardware profiles, equal budgets,
three or more seeds, repeated measurements, the required quality/size/
RAM/VRAM/prompt/decode/cost metrics, and explicit decision thresholds. The
matrix has a hard cell limit so a malformed or accidentally expanded campaign
cannot consume unbounded resources.

The protocol identity is content addressed by `protocol_id`. A protocol is
only a preregistration record; constructing or rendering it is not benchmark
evidence.

## Cells and artifacts

`V2BenchmarkCell` is immutable and content addressed by model, dataset, method,
hardware, budget, and seed. Every cell is retained as `measured`,
`negative_result`, `unsupported`, `failed`, or `unknown`. Terminal cells must
carry a reason. Measured cells must contain every declared metric, a confidence
interval from at least three repetitions, and an available artifact.

`ReferenceArtifactLineage` requires ordered mutation, repair, quantization,
runtime, and rollback stages. Each stage records input/output/config digests.
The license decision is part of the lineage; a prohibited artifact can be
retained as an explicit non-publication result but cannot become a reference
artifact.

## Claims and publication

`V2CompetitivenessClaim` names candidate and baseline cell IDs. The
`evaluate_v2_publication` decision is publishable only when:

- the complete preregistered matrix is present;
- negative, unsupported, failed, and unknown cells are retained with reasons;
- every published claim points to measured, deployable cells with confidence
  intervals, and the candidate interval clears the baseline interval;
- an independent public audit is clean, replay passes, and no critical finding
  remains.

Any missing condition returns `withhold` with deterministic reasons. This
contract intentionally does not contain live benchmark measurements in the
repository. A future campaign must add signed evidence bundles and licensed
artifact files only after these gates pass.

`build_v2_benchmark_bundle` sorts retained cells and claims into one
content-addressed `V2BenchmarkBundle`; its `decision` and `bundle_id` are
recomputed from the canonical evidence rather than trusted from caller input.

## Public API

```python
from modelsurgeon.evaluation import (
    V2BenchmarkProtocol,
    evaluate_v2_publication,
)

decision = evaluate_v2_publication(protocol, cells, claims, audit)
if decision.outcome.value != "publish":
    raise RuntimeError("retain the bundle and do not publish")
```

The existing `modelsurgeon benchmark protocol` and matrix commands remain the
v1.1 execution surface. This v2 contract is deliberately API/schema-first
until a campaign executor can supply real, licensed, replayable evidence.

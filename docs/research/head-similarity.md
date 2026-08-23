# Attention-head similarity metric spike

Issue: #40

## Question

Which bounded redundancy signal should ModelSurgeon use when comparing attention heads?

The executable spike compares three metrics on two deterministic tiny attention fixtures,
with three candidate head pairs per fixture and a known redundancy ordering.

## Metrics

1. `weight_cosine`: absolute cosine of flattened head projection weights.
2. `output_correlation`: absolute Pearson correlation of calibration head outputs.
3. `subspace_projection`: normalized squared overlap between orthonormalized projection subspaces.

For each model/method the harness records:

- Spearman correlation against the known redundancy ranking;
- split-half ranking stability (even vs odd output observations where applicable);
- deterministic workspace bytes and operation units;
- measured wall time under a fixed ceiling.

The executable evidence lives in `modelsurgeon.features.head_similarity_spike` and
`tests/test_head_similarity_spike.py`.

## Result

The deterministic fixtures are intentionally constructed so that a functionally duplicate
head can have dissimilar raw projection weights. This tests the failure mode that matters
for pruning: parameter similarity is not the same thing as functional redundancy.

Both output correlation and projection-subspace overlap recover the known redundancy
ordering on both fixtures. Raw weight cosine does not. Output correlation has the smaller
workspace and deterministic operation cost of the two successful signals and remains stable
across even/odd calibration observations.

## Decision

**Use output correlation as the default bounded attention-head redundancy metric.**

Keep projection/subspace similarity as a secondary confirmation/research feature where
budget permits. Keep raw weight cosine as a cheap candidate screen only; do not treat it as
a sufficient redundancy decision by itself.

This is a bounded engineering decision, not a universal scientific claim. Real-model
mutation datasets should re-test whether subspace confirmation adds enough predictive value
to justify its extra cost.

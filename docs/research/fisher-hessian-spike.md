# Diagonal Fisher / Hessian sensitivity spike

Issue: #37

## Question

Which bounded sensitivity estimate should ModelSurgeon carry forward after the first-order
`weight × gradient` features from #36?

The comparison is intentionally small and deterministic. It uses two four-parameter
quadratic probes whose exact zeroing/removal loss deltas are known analytically. Each
probe supplies four signed gradient observations and an exact diagonal Hessian.

## Candidates

1. `first_order`: `abs(w * mean(g))`.
2. `empirical_fisher`: `0.5 * w^2 * mean(g^2)`.
3. `diagonal_hessian`: `0.5 * w^2 * H_diag`.

For each candidate the harness records:

- Spearman ranking correlation with exact removal deltas;
- split-half ranking stability;
- peak accumulator/workspace bytes;
- deterministic operation units;
- measured wall time under a fixed per-probe time ceiling.

The executable evidence lives in `modelsurgeon.features.sensitivity_spike` and is
covered by `tests/test_sensitivity_spike.py`.

## Deterministic small-case result

With the default two probes and a 4096-byte workspace budget:

| Method | Mean predictive Spearman | Split-half stability | Peak bytes | Total operation units |
| --- | ---: | ---: | ---: | ---: |
| first_order | 0.0 | 1.0 | 32 | 64 |
| empirical_fisher | 1.0 | 1.0 | 32 | 96 |
| diagonal_hessian | 1.0 | 1.0 | 64 | 256 |

Wall time is measured on each execution and must remain below the supplied fixed
budget; it is not hard-coded into this note because that value is host-dependent.

## Decision

**Carry forward empirical Fisher as the bounded second-order proxy.**

On the controlled probes it preserves the exact removal ranking obtained by the
true diagonal Hessian while using the same one-float-per-parameter accumulator
footprint as the first-order gradient path. It also reuses gradients already collected
by #35 instead of requiring a second-order backward/Hessian pass.

The diagonal-Hessian implementation remains a validation/reference method for tiny
or explicitly budgeted experiments. It should not become the default consumer-hardware
feature path unless later real-model evidence shows a material predictive advantage over
empirical Fisher.

## Boundary

This spike does not claim that empirical Fisher is universally superior. The decision
is specifically that it is the best bounded implementation to advance into the next
ModelSurgeon experiments given equal small-case predictive ranking and materially lower
resource cost. Real-model mutation datasets must re-test that assumption.

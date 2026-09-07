# Long interaction-aware sequence study

The long-sequence study closes a matched matrix over two model families,
stateless/additive/state-aware policies, horizons 10/20/50, and at least three
seeds. Every horizon is retained as measured, negative, failed, unsupported, or
unknown. A measured horizon must bind a complete checkpoint state ID, output
artifact digest, quality, cumulative regret, violations, evaluation count,
rollback count, optimization cost, canonical provenance, and all three
adversarial interaction classes: same-layer, cross-layer, and distribution
shift.

The study records bootstrap intervals for regret and means for violations and
evaluations. A physical directional claim is intentionally not inferred from
one horizon or from missing cells; the default preregistered matrix is
unsupported until licensed checkpoints, held-out corpora, repeated deployment
measurements, and rollback lineage are available. Failed and adversarial cells
remain evidence rather than disappearing from the denominator.

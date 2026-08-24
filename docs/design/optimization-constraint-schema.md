# Optimization constraint schema v1

Search hard constraints are independent, canonically ordered predicates. Version 1
supports minimum quality-retention ratio, maximum perplexity delta, minimum latency
gain ratio, and maximum peak RAM, peak VRAM, and disk bytes. Ratios are fractions
(`0.10` means 10%), perplexity delta is measured in perplexity points, and all
resource limits use bytes.

Quality, perplexity, and latency observations name the immutable source checkpoint
or parent candidate used as their baseline. Resource observations use the explicit
`absolute` baseline. A missing observation or mismatched baseline fails the complete
constraint set closed; threshold comparisons run in canonical metric-name order, so
input ordering cannot change decisions or the content-addressed constraint-set ID.

Resolved application configuration exposes these fields under `constraints`.
Quality retention is always present; other predicates are opt-in. The older
`objective` configuration remains readable for compatibility but does not define
the v1 hard-constraint evaluation contract.

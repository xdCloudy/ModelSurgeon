# Active-learning sample-efficiency study

Issue #116 requires active, seeded-random, and utility-only strategies to use identical
experiment-count grids and seed sets. Each curve records predictive performance versus
experiments and computes normalized trapezoidal area under the learning curve (AULC).

Strategy summaries report mean AULC and seeded bootstrap confidence intervals across runs.
If active mean AULC does not exceed the strongest equal-budget baseline, the versioned study
contains an explicit negative-result statement. A dependency-free SVG renderer plots mean
performance against experiments and carries that statement into the visual artifact.

Missing strategies, unequal budgets, unequal seed sets, non-finite measurements, or invalid
confidence/bootstrap bounds fail closed.

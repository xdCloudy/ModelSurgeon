# Tree surgeon uncertainty comparison

Issue #106 defines one bounded comparison contract for ensemble, bootstrap, and quantile
tree methods. Producers may train models independently, but all three must submit aligned
validation intervals under the same per-method fit, CPU-time, thread, prediction-value,
and consumer-memory ceilings.

The comparison records empirical interval coverage, target coverage, mean interval width,
Spearman correlation between uncertainty and absolute prediction error, fit count, CPU
seconds, serialized model bytes, and retained prediction values. Selection first minimizes
absolute coverage error, then maximizes error-ranking utility, then minimizes CPU time and
model bytes, with method name as the deterministic final tie-break.

Ensemble and bootstrap uncertainty is the sample standard deviation across independently
trained members, with empirical member quantiles supplying the interval. Quantile trees
supply lower, central, and upper predictions directly; their half-width is treated as the
uncertainty proxy for this comparison. The distinction is retained in `method` and
`technique_version` rather than presenting quantile width as identical to ensemble
epistemic variance.

Every selected prediction exposes schema-versioned point, lower, upper, and uncertainty
values for downstream acquisition (#112). Comparisons fail closed when a method is absent,
cost or memory ceilings are exceeded, values are non-finite, arrays do not align, or an
interval does not contain its point estimate.

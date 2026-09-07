# Measurable target elicitation

The conversational layer does not convert words such as “faster”, “better”,
or “fit on my GPU” into an invented threshold. `assess_measurable_targets`
classifies the typed intent against the verified measure set and returns one
of three outcomes:

- `complete` when the executable objective already contains metric, direction,
  unit and required threshold fields;
- `needs_clarification` with one deterministic, targeted question when an
  actionable target is missing; or
- `unsupported` when a declared metric is outside the verified boundary.

Questions are request-linked and digest-stable. Latency/throughput prompts
require a value and unit; memory prompts require a limit and runtime/device
context; quality prompts require a named metric and threshold or baseline; and
deployment prompts require a declared target plus measurable budgets.

No default threshold, baseline, quality claim, or deployment compatibility is
created. Answers remain typed clarification records and must pass the existing
canonical intent compiler and policy evaluator before an `OptimizationSpec` can
be emitted.

The recognized measure labels and their downstream limitations are a bounded
policy surface, not a general metric parser. Unknown metrics are retained as
unsupported, and deployment compatibility is never inferred; the complete
matrix and known skips are frozen in the [v2.4 release boundary](../release/v2.4-clarification-boundary.md).

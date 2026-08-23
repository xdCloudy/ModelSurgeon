# Activation summary features

An `ActivationBatch` contains ordered sample IDs, `[batch, token, feature]` values,
and a matching token mask. `ActivationSummaryCollector` flattens only included token
vectors into the bounded streaming accumulator; masked padding or control tokens can
never affect moments, extrema, RMS, sparsity, activation frequency, or percentiles.

Each emitted schema-v1 feature names its canonical component, extractor/version,
precision provenance, complete ordered sample context, aggregation axes, feature
width, and included scalar observation count. Duplicate sample IDs, inconsistent
feature widths, malformed masks, mismatched contexts, and all-masked inputs fail
explicitly rather than emitting misleading summaries.

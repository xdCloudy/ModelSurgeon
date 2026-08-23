# Mergeable bounded-memory statistics

`StreamingStatistics` combines Welford online moments, Chan pairwise merge formulas,
an exact sum of squares, extrema and zero/positive counters with a fixed-size linear
histogram. Memory is `O(histogram_bins)` and independent of batches, tokens, and model
size. Compatible accumulators merge without replaying observations.

Snapshots expose count, population mean/variance, RMS, min/max, configurable
near-zero frequency, positive activation frequency, and requested percentile
estimates. Percentile accuracy is bounded by configured histogram width inside its
range; outliers occupy the end bins while exact extrema remain available. Non-finite
values, empty summaries, invalid quantiles, and incompatible merges fail explicitly.

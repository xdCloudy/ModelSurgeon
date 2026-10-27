# Safe-mutation probability calibration

Issue #105 introduces a validation-only calibration boundary for binary safe-mutation
probabilities. Calibration never consumes the held-out test partition: the fitting API
accepts parameters named `validation_probabilities` and `validation_labels`, and the
serialized selection evidence records `fit_partition: validation`.

Two deterministic methods are compared with out-of-fold predictions on the
validation observations. When the training matrix supplies split group IDs, all
rows from each group remain together and a deterministic greedy stratifier
balances the class proportions across folds. The fold count is capped at five
and at the number of distinct groups containing each class; at least two groups
per class are required so every held-out fold's training complement retains
both classes. Calibration is skipped when that group support is insufficient.
Direct calls without group IDs use the row-wise stratified round-robin protocol.
After method selection, the winning method is refit on the complete validation
partition for inference.

- Platt scaling applies a two-parameter logistic transform to clipped probability logits.
- Isotonic regression uses pair-adjacent violators after deterministically grouping equal
  input probabilities, then interpolates its monotone fitted points at inference.

Selection minimizes cross-fitted Brier score, then expected calibration error, then the
method name as a stable final tie-break. The artifact records the fold count and assignment
protocol and both candidates rather than only the winner. Group-aware training records
`stratified_group_greedy_v1`; row-wise direct calls record `stratified_round_robin_v1`.
Each candidate carries
out-of-fold Brier score, fixed-bin ECE, and a complete reliability curve whose empty bins
remain explicit. These scores compare calibration methods without evaluating each fitted
transform on the same rows used to fit it; they are still validation-based and are not a
substitute for a final untouched test partition.

The cross-fitting pattern is informed by scikit-learn's `CalibratedClassifierCV` guidance
and was independently implemented for ModelSurgeon's already-fitted probability stream;
no upstream code was copied and no dependency was added. Upstream inspected:
[`scikit-learn/scikit-learn`](https://github.com/scikit-learn/scikit-learn), revision
`bb9cefde592afccd276c5b2d75957dfde91d5b3e`, BSD-3-Clause (`COPYING`). The relevant
guidance is the [probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html).

Serialized calibrators are schema-versioned and fail closed for unknown versions,
non-finite parameters, invalid probabilities, non-binary labels, single-class fitting
sets, non-monotone isotonic values, or malformed threshold arrays. Test-partition metrics
must be computed after selection with `calibration_metrics`; they cannot influence fitting
or method choice.

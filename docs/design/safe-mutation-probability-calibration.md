# Safe-mutation probability calibration

Issue #105 introduces a validation-only calibration boundary for binary safe-mutation
probabilities. Calibration never consumes the held-out test partition: the fitting API
accepts parameters named `validation_probabilities` and `validation_labels`, and the
serialized selection evidence records `fit_partition: validation`.

Two deterministic methods are fitted to the same validation observations:

- Platt scaling applies a two-parameter logistic transform to clipped probability logits.
- Isotonic regression uses pair-adjacent violators after deterministically grouping equal
  input probabilities, then interpolates its monotone fitted points at inference.

Selection minimizes validation Brier score, then expected calibration error, then the
method name as a stable final tie-break. The artifact records both candidates rather than
only the winner. Each candidate carries Brier score, fixed-bin ECE, and a complete
reliability curve whose empty bins remain explicit. This makes a poor or negative result
inspectable instead of hiding it behind the selected transform.

Serialized calibrators are schema-versioned and fail closed for unknown versions,
non-finite parameters, invalid probabilities, non-binary labels, single-class fitting
sets, non-monotone isotonic values, or malformed threshold arrays. Test-partition metrics
must be computed after selection with `calibration_metrics`; they cannot influence fitting
or method choice.

# Active-learning candidate scoring

Issue #109 scores utility, named outcomes, calibrated safe probability, and uncertainty in
bounded batches of at most 4,096 rows. Compatible candidates retain their canonical input
order, outcome names are canonicalized, and every scalar output is checked for alignment,
finiteness, probability bounds, or non-negative uncertainty.

The scoring boundary requires the expected feature schema version and exact ordered feature
names. Version, name, width, and finite-value incompatibilities are quarantined with an
explicit reason and observed schema rather than entering a model batch. Model-output faults
fail the operation instead of being mislabeled as candidate schema drift.

Calibration uses the versioned validation-only calibrator from issue #105. Reports retain
model and tool revisions, batch size, compatible scores, and quarantined candidates. Since
no batch-local normalization or random state is allowed, the same predictors and candidate
order produce identical scores across batch sizes.

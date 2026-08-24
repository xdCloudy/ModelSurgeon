# MLP surgeon uncertainty comparison

Issue #107 compares independently seeded deep ensembles with Monte Carlo dropout under
shared training, stochastic-pass, CPU-time, and retained-prediction ceilings. Both methods
produce schema-versioned point estimates, empirical intervals, and sample standard
deviations.

Validation reports interval coverage and absolute coverage error as the calibration
measure. Active-selection value is the lift in mean absolute error among the most uncertain
configured fraction over mean error across the validation set; values above one indicate
that uncertainty concentrates expensive mistakes. Selection minimizes calibration error,
then maximizes active-selection lift, then minimizes observed CPU and model bytes.

The feature is optional. Setting `MLPUncertaintyBudget.enabled` to false returns before
evidence validation or any PyTorch work. Tree uncertainty modules neither import nor call
the MLP runner, so disabling this research path cannot affect tree acquisition workflows.

`MLPConfig.dropout` defaults to zero. Existing serialized MLP bundles that predate the
field load as zero-dropout models, while dropout-trained bundles can run deterministic
Monte Carlo pass sequences from an explicit seed.

# Surrogate-assisted architecture search

Surrogate search v1 consumes the complete legal architecture states defined by
the candidate-space and policy contracts. It encodes conditional mixed axes,
hardware placement, analytic parameter/storage cost, and source distance into a
bounded normalized feature vector. Non-numeric axis values use stable
content-addressed encodings; Python process hash state is never used.

`fit_surrogate` trains a small RBF-compatible kernel model from measured
complete states only. Unsupported, failed, unknown, and otherwise missing
results remain in the fit record but cannot become training labels. The model
retains objective confidence intervals, a feasibility probability, held-out
objective RMSE/coverage, and held-out feasibility Brier error. Training-example,
operation, model-memory, and wall-clock budgets are checked before and during
fit.

Expected-improvement and hypervolume-improvement acquisition share a bounded
batch selector. Hard constraints are evaluated before acquisition, and a
surrogate feasibility threshold is applied before a candidate can enter the
batch. Deterministic seeded ties plus assignment diversity make batch and
resume decisions reproducible. If the model is insufficiently trained,
uncalibrated, unsupported, or over budget, the policy records the reason and
falls back explicitly to the supplied candidate evidence rather than treating
an uncalibrated prediction as measured support.

`SurrogateModel`, `SurrogateFit`, and `SurrogatePolicyState` have typed
round-trip records. The policy identity commits the acquisition configuration
and objective/constraint definitions, so a resume cannot silently change the
search contract.

The checked-in evidence is synthetic contract evidence. It demonstrates
calibration and bounded/fallback behavior but does not claim a physical
hypervolume or regret improvement. The decision to retain a surrogate as a
default search method requires the downstream physical study with paired
budgets, held-out models, and measured deployment cost.

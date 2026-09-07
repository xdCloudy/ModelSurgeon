# Conditional repair-recoverability predictors

`modelsurgeon.surgeon.recoverability` predicts held-out quality gain and repair success for a
candidate state while retaining no-repair as an explicit action. It is a bounded research
contract, not permission to execute a repair automatically.

## Dataset and actions

Each `RecoverabilitySample` joins fixed-width damage/state/meta features with a model revision,
state/candidate identity, model family, lineage group, and a conditional `RecoverabilityAction`.
Actions are keyed by repair method and budget name; `no_repair` with the `zero` budget is required
as a control. Outcomes are measured, unsupported, failed, or unknown. Group-disjoint splits reject
model, state, and candidate leakage across train, validation, and test.

## Predictor and policy

The implementation fits a deterministic action-specific ridge baseline, calibrates an absolute
residual interval from validation observations, and estimates action success with a Laplace-smoothed
training frequency. It retains per-action test MAE against the global action-mean baseline,
success Brier evidence, and measured/negative/unsupported/unknown status.

Recommendations are conservative: an ordinary repair must clear both the lower gain interval and
success-probability threshold. Otherwise the predictor chooses the measured no-repair arm when it
is supported. Unseen parent states, out-of-range features, incompatible actions, missing arms, and
inconclusive evidence produce out-of-distribution, unsupported, or unknown decisions instead of
an extrapolated repair.

## Research protocol

Evaluate leave-model/state/family folds with repeated seeds, grouped intervals, damage-only and
global-mean baselines, calibration/selective-risk, regret against the oracle action, and explicit
no-repair false-skip risk. Retain failures, unsupported cells, negative results, and resource
limits; do not convert inconclusive evidence into a success claim.

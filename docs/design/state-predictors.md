# State-dependent predictors

`modelsurgeon.surgeon.state_predictors` is the v1.5 calibrated prediction
boundary for safety probability, quality delta, latency delta, memory delta,
and cumulative non-additivity error.

## Dataset and leakage contract

Each sample binds one current-state embedding to a candidate, model revision,
parent state, lineage group, target unit, and explicit measured/unsupported/
failed/unknown outcome. `StatePredictionDataset` rejects reuse of model,
parent-state, or candidate keys across train, validation, and test partitions.
Embeddings must have one fixed width and feature vocabulary throughout the
dataset.

The target observation can also retain an additive baseline. The state model
and additive comparator are evaluated separately; an unavailable additive
baseline is not treated as a zero target.

## Training and inference

`fit_state_predictors()` trains the existing bounded linear ridge surgeon model
on state values and masks only. Absolute validation residual quantiles produce
the prediction interval. A smoothed training failure rate is retained as a
separate failure probability rather than folded into the numeric delta.

Inference rejects an unseen parent state or any feature outside the training
range as `out_of_distribution` and returns no numeric delta. A predicted value
always carries its calibrated interval and failure probability. The model card
retains the train state IDs, feature bounds, dataset identity, protocol,
configuration, and baseline/test scores.

## Claim policy

Scores compare state-conditioned MAE/RMSE with stateless mean and additive
baselines, and report failure Brier error independently. A target is marked
`measured` only when held-out state error improves on every available baseline;
otherwise the result is retained as `negative_result`, `unknown`, or
`unsupported`. The contract makes no superiority claim without the required
held-out evidence, grouped bootstrap analysis, horizon strata, and repeated
seed/corpus protocol.

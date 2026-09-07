# Repair cost predictors and budget preflight

`modelsurgeon.surgeon.repair_cost_predictors` predicts bounded repair resource use by method and
budget. It covers tokens, wall/GPU/CPU seconds, peak RAM/VRAM, artifact/cache bytes, and optional
energy joules.

## Contract

`RepairCostSample` joins fixed-width damage/state/configuration features with model, hardware,
state, candidate, lineage, action, and measured resource evidence. Group-disjoint splits reject
model, hardware, state, and candidate leakage. Measured, unsupported, failed, and unknown cells
remain explicit. Energy is optional: an absent sensor creates an unsupported energy target while
other resource predictors remain usable.

Each action/target model is a deterministic ridge baseline with a validation-calibrated absolute
residual interval, training-state support, feature bounds, and immutable predictor identity.
Held-out MAE is retained against an action-mean baseline; negative and missing evidence are not
rewritten as success.

## Conservative preflight

`RepairCostStudy.preflight()` evaluates an action against `RepairCostLimits` using the upper end of
each calibrated interval. It refuses actions that are out of distribution, lack required target
evidence, or exceed a hard limit. Optional energy remains absent rather than becoming a zero or an
implicit blocker. Compatibility rejection is carried by the action and is fail-closed.

## Research protocol

Use the repair dataset across methods, budgets, model/hardware-held-out folds, repeated seeds, and
analytic/mean baselines. Report interval coverage, error, conservative budget cost, sensor
availability, failures, unsupported cells, and any negative result with complete provenance.

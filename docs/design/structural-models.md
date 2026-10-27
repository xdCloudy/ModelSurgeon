# Structural feature-path study

`modelsurgeon.surgeon.structural_models` is an experimental, framework-neutral
comparison of two linear predictors: pooled node/parameter features and the
same predictor with topology and mutation-history aggregates. It does **not**
implement a neural MLP, Set Transformer, or graph neural network. The v1 enum
names remain as source-compatibility aliases, but v2 study records use the
accurate feature-path names.

## Input and resource contract

Each example carries node features, graph edges, bounded order history,
parameter count, model revision, parent state, lineage group, split, and an
explicit measured/unsupported/failed/unknown outcome. Node, edge, history,
parameter, memory, and latency limits are validated before fitting. Model and
state identities must be disjoint across train, validation, and test splits;
training outcomes cannot be inferred from censored records.

All paths share the same standardized linear regressor, training budget, seed
set, and held-out protocol. Reported parameter counts are derived from this
predictor. Peak-memory and latency fields are analytical feature-path proxies,
not measured runtime or physical hardware results, and are not physical
performance evidence.

## Selection and evaluation

Model selection uses only validation MAE intervals. The test split is evaluated
after the decision and cannot change which path is selected. Test metrics are
reported separately for final comparison. The topology/history ablations are
also diagnostic results and do not select the model. This split is required to
avoid selecting an architecture on the same outcomes used to report its
generalization.

The study does not establish that topology or history improves physical model
optimization. It only tests whether these aggregate input features improve a
bounded linear outcome predictor. True set/graph representations, predictors
trained on provenance-verified physical outcomes, measured inference costs,
and downstream optimization decisions remain future research.

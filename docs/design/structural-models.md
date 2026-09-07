# Structural surgeon model study

`modelsurgeon.surgeon.structural_models` is the v1.5 evidence boundary for
comparing a simple tabular MLP baseline with bounded set and graph surgeon
encoders. The module is framework-neutral: it provides deterministic,
resource-bounded feature paths and study records that can be consumed by a
larger backend later without changing the claim policy.

## Input and resource contract

Each example carries node features, graph edges, bounded order history,
parameter count, model revision, parent state, lineage group, split, and an
explicit measured/unsupported/failed/unknown outcome. Node, edge, history,
parameter, memory, and latency limits are validated before fitting. Model and
state identities must be disjoint across train, validation, and test splits;
training outcomes cannot be inferred from censored records.

Every family uses the same training steps, tuning-trial count, seed set, and
held-out protocol. The default study requires five unique seeds. Predictor
records expose deterministic inference and retain feature normalization,
coefficients, seed, protocol, and training budget. Resource records are
explicit deterministic upper bounds, not unmeasured claims of GPU behavior.

## Model paths and ablations

The MLP baseline uses pooled node and parameter signals. The set path uses
order-invariant node statistics and does not consume graph topology. The graph
path adds edge density and degree signals plus bounded order history. The study
also reruns the selected-or-graph path with topology, order history, and
parameter-count ablations so the source of any gain remains visible.

## Selection policy

Held-out MAE, RMSE, and ranking accuracy are reported with deterministic
list-level bootstrap intervals across test examples and seeds. A set or graph
model is selected only when its held-out MAE interval is entirely below the
MLP interval and its resource record stays within budget. Otherwise the study
retains the models and ablations but rejects the added complexity with an
explicit negative decision.

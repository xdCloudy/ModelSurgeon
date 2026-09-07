# Current-state embedding

`modelsurgeon.surgeon.state_embedding` defines the v1.5 current-state input
contract for state-dependent prediction. It is a fixed-width, versioned
representation of a `DeployableArchitectureState` plus already accepted
mutations, current cumulative metrics, remaining constraints, and retained
evidence outcomes.

## Encoding

The vector contains normalized architecture scalars, axis-status masks,
layer-width summary statistics and histograms, an order-independent hashed set
encoding of accepted mutations, a position-sensitive hashed encoding of the
bounded mutation history, metric values/statuses, constraint buckets, and
measured/unsupported/failed/unknown evidence counts. `StateEmbeddingConfig`
fixes every width and vocabulary dimension; missing axes and evidence are
represented by masks rather than zero-filled claims.

Set-like inputs are canonicalized before encoding, so input record order does
not change the result. Mutation history preserves order. When history exceeds
the configured bound, the encoder retains the most recent entries, a
truncation count, and a digest of the omitted prefix. The truncation policy is
therefore explicit and auditable instead of silently treating long histories
as equivalent.

## Leakage boundary

`CurrentStateInput` accepts only current architecture facts, already accepted
mutations, current observed metrics, current constraints, and retained
evidence outcomes. It has no future-candidate or held-out-outcome field.
Unknown, unsupported, and failed observations carry status masks and no
numeric value. A caller must construct a new state after a mutation; a future
mutation cannot enter the current state by being appended to the input set.

## Identity and claim boundary

The state ID, protocol version, feature names, values, masks, configuration,
and history truncation metadata produce a deterministic `embedding_id`.
Changing architecture, mutation order, status, or retained evidence changes
the identity even when a bounded numeric bucket collides. The contract does
not choose a neural architecture or claim downstream predictive superiority;
collision, coverage, resource, and ablation results remain evidence to be
measured by downstream work.

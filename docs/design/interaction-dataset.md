# Mutation interaction dataset

`modelsurgeon.datasets.interaction` is the v1.5 evidence boundary for
state-dependent mutation interactions. It stores measured single mutations,
ordered pairs, and longer cumulative sequences without collapsing unsupported,
failed, or unknown cells into missing data.

## Record contract

Each `InteractionExample` binds a mutation sequence to the model revision,
source artifact digest, root and parent/result state IDs, ancestor state IDs,
corpus, hardware profile, runtime, seed, topology distance, deployment
metrics, and immutable benchmark provenance. The content-addressed
`example_id` includes these values and the retained outcome, so changing a
measurement cannot silently overwrite an earlier cell.

`InteractionOutcome` is one of `measured`, `unsupported`, `failed`, or
`unknown`. Terminal non-measured cells retain a required reason and no partial
metrics. This makes a failed second mutation an explicit censored observation
rather than an apparently successful zero-cost result.

## Interaction reconciliation

`build_pairwise_interaction()` validates A→B or B→A order and matching model,
state-root, corpus, hardware, runtime, artifact, and seed context. It records
the additive expectation from the two independent single cells and the
observed-minus-expected non-additivity for every metric.

`build_cumulative_interaction()` applies the same contract to a two-step or
longer cumulative sequence. It supports the preregistered 5–20-step history
shape while remaining bounded to the supplied evidence. A cumulative cell is
never inferred from a missing or incompatible single cell.

## Leakage-safe splits

`InteractionSplit` assigns immutable lineage groups to train, validation, and
test. `InteractionDataset` rejects cross-partition reuse of model revision,
source artifact, corpus, root state, or any ancestor state. Descendants of one
architecture therefore remain in one partition even when their mutation order
or interaction kind differs. Dataset and example IDs are deterministic JSON
SHA-256 identities, and `to_record()` emits coverage counts for outcomes,
interaction kinds, and reconciled cells.

## Reproduction and claim boundary

The dataset contract does not claim a trained state model or generalize beyond
the retained evidence. A research run must persist both mutation orders, same-
layer and cross-layer pairs, bounded cumulative sequences, held-out text and
deployment metrics, fixed seeds/budgets, full provenance, and a leakage audit.
Negative or inconclusive cells remain part of the published manifest.

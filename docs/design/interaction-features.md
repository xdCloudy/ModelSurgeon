# Interaction-aware feature extraction

`modelsurgeon.features.interaction_features` is the v1.5 integration boundary
between the bounded feature primitives and the mutation interaction dataset.
It emits one deterministic record per explicitly supplied component pair or
cumulative interaction cell.

## Inputs and leakage boundary

`InteractionFeatureCell` accepts only pre-mutation activation, gradient,
topology, and redundancy observations. The source phase is a required
`pre_mutation` value; post-mutation cells are rejected before extraction. The
interaction example remains the target/evidence reference, so failed,
unsupported, and unknown targets remain visible without being replaced by
zeroes. The sample context records the dataset revision, split, sample IDs,
preprocessing, and tokenizer identities.

The extractor does not receive post-mutation vectors or a separate target
sample stream. Callers must collect held-out samples in the declared context
and cannot make them appear in a pre-mutation feature record by changing a
missing value.

## Features

For each explicit cell the extractor can emit activation overlap, gradient
overlap, redundancy score, decomposed topology distance, mutation-order
length, order sensitivity, and cumulative non-additivity errors. Missing
primitives are named in `missing_features`; they are never silently emitted as
zero. Pair symmetry is represented by canonical component endpoints while
mutation order remains in the interaction sequence and order-sensitive value.

## Resource and reproducibility contract

`InteractionFeatureBudget` bounds explicit cells, block size, planned RAM,
planned VRAM, and elapsed time. The extractor consumes the supplied cells
incrementally and never constructs an all-pairs Cartesian product. Its
workspace estimate is linear in the configured cell cap. Deterministic feature
IDs include the interaction example, component endpoints, sample context,
values, missingness, source phase, and extractor revision.

This module extracts evidence; it does not train a state model or make a
superiority claim. Seed/corpus ablations and cross-platform stability remain
research evidence to be retained by the caller.

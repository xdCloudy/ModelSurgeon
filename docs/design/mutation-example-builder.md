# Mutation example builder

Issue #84 constructs supervised `MutationExampleRecord` values from immutable experiment records and explicitly provenanced pre-mutation feature partitions.

## Input boundary

The builder accepts `FeaturePartition` inputs rather than unkeyed `FeatureRecord` values. Every partition must match the experiment's exact pre-mutation model revision and configured input revision. The default input identity is the dataset manifest ID; callers may explicitly choose the dataset revision when that is the revision used by feature extraction.

Data-derived features that carry `FeatureSampleContext` must also match the experiment dataset identifier, dataset revision, split, tokenizer, and tokenizer revision. Static model features may omit sample context. Duplicate feature identities across partitions fail closed.

These checks make post-mutation feature snapshots ineligible as supervised inputs: a partition from a changed model revision cannot be joined to the original experiment.

## Targets and missingness

Baseline, post-mutation, and delta `MetricObservation` values are copied without reinterpretation. `absent`, `skipped`, and `failed` states retain their original reasons; the builder never replaces unavailable targets with numeric zeroes.

The default target policy requires at least one measured delta metric. Records without one are excluded with a machine-readable `no_measured_delta` reason. `preserve_missing` may be selected explicitly when downstream work needs examples whose delta targets are entirely unavailable.

## Outcome policy

Successful and threshold-rejected experiments are included by default. Failed experiments are excluded because an execution failure is not automatically a meaningful supervised target. A caller may explicitly allow failed outcomes when a dataset design requires them.

## Stable identity and retries

Example identity hashes immutable logical experiment identity, mutation identity, model/dataset/config revisions, the builder version, and a digest of the pre-mutation feature snapshot. Execution attempt IDs are intentionally excluded.

Exact logical retries therefore resolve to the same example ID and are emitted once. If the same stable example ID is observed with different supervised content, construction fails closed instead of selecting one retry arbitrarily.

## Scope boundary

This builder performs canonical joins and local leakage guards only. Dataset-wide duplicate, ancestry, component-overlap, and split leakage auditing belongs to the dedicated split and leakage-audit stages (#86-#88).

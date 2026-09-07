# Hardware cost training dataset

`modelsurgeon.datasets.hardware_cost` is the v1.3 dataset boundary for joining physical
artifact outcomes, deployment costs, hardware profiles, runtime settings, and repeated
measurements. It is a dataset contract, not a predictor or mutation selector.

## Example identity and targets

Measured examples bind model and architecture revisions, source/output artifact SHA-256
digests and bytes, hardware profile/context IDs, runtime revision/settings, seed, benchmark
record ID, protocol/tool revisions, and the full sample distribution for load, prefill,
decode, latency, RAM, VRAM, and disk metrics. Summaries are recomputed from samples and
reject unit, finiteness, or summary drift.

Unsupported, failed, unknown, and unstable outcomes remain explicit with reasons and do not
publish partial targets.

## Leakage audit

Every example has a deterministic lineage group derived from model revision, source artifact
digest, and hardware profile. `HardwareCostSplit` rejects group IDs shared by train,
validation, and test partitions; the dataset also requires every example to belong to one
declared partition. `partition_by_profile()` keeps storage and downstream normalization
profile-specific so CPU/GPU measurements cannot be silently mixed.

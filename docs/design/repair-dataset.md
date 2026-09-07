# Mutation-specific repair and recoverability dataset

The repair dataset is a versioned, leakage-safe collection of paired outcomes for a damaged
candidate. Every repair method and budget is paired with a no-repair control evaluated on the
same held-out groups, so recovery claims cannot be confused with ordinary candidate quality.

## Contract

`modelsurgeon.datasets.RepairDatasetExample` retains:

- model family and revision, mutation state/kind/severity, source and damaged-candidate artifact
  digests, dataset/tokenizer revision, hardware profile, budget name, seed, and lineage group;
- a `RepairBaseline` with zero publication cost and held-out evidence for the unchanged damaged
  candidate; and
- a complete `RepairResult`, including accepted, rejected, failed, unsupported, and unknown
  outcomes, bounded step/token/time/GPU/energy/disk cost, rollback/search-parent state, child
  artifact lineage, and held-out evidence when available.

The pair rejects mismatched candidate identities, source lineage, data/tokenizer revisions, or
budgets. A published example therefore cannot silently compare different mutation states,
evaluators, or resource envelopes.

## Coverage and splits

`RepairDatasetConfig` requires the declared dataset revision and, by default, zero, short, and
medium budgets, at least two model families, two mutation kinds, and two seeds. These are
validation requirements rather than claims that a mock fixture is a benchmark result. The
dataset keeps terminal failure and unsupported cells instead of dropping them.

`RepairDatasetSplit` assigns complete lineage groups to train, validation, or test and rejects
cross-partition reuse. The resulting `dataset_id` hashes the schema, protocol, configuration,
split, and every paired record. Provenance retains the source revision, tool revision, command,
configuration, and record identity needed to reproduce or audit each measurement.

## Benchmark protocol

The v1.7 collection should cover LoRA, selected/full fine-tuning, and logit/feature distillation
across at least two architecture families, multiple mutation types and severities, and zero,
short, and medium budgets. Use fixed seeds and held-out groups, retain OOM/non-finite/overfit/
non-beneficial outcomes, and publish complete cost and artifact lineage. A negative or
inconclusive cell remains a result; it is never rewritten as successful recovery.

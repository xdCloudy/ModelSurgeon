# Cross-model transfer suite

The transfer suite is a deterministic manifest layer for evaluating surgeon
predictors when the target checkpoint, size, or family is held out. It keeps
source fitting separate from target adaptation and target testing so that
target rows, statistics, vocabularies, ancestry groups, and held-out corpora
cannot silently enter source training.

## Protocols

- `leave_one_checkpoint`: hold out one checkpoint and revision.
- `leave_one_size`: hold out all samples at one model size.
- `leave_one_family`: hold out all samples in one model family.
- `zero_target`: evaluate a checkpoint with no target adaptation examples.
- `few_shot`: reserve a deterministic, seed-ranked prefix for target
  adaptation and keep the remainder for testing.

Every fold records source sample IDs, target adaptation IDs, target test IDs,
lineage groups, a stable fold identity, and the protocol revision. Folds with
ancestry overlap, no source data, or no target test data are omitted rather
than silently weakening the isolation contract.

## Evidence contract

Results are attached to folds only through
`record_transfer_fold_result`. Each result must report ranking, calibration,
numeric error, and frontier-quality metrics, together with evaluation count,
failure evidence when applicable, and optional provenance. Metric intervals are
stored alongside point values so grouped bootstrap confidence intervals can be
computed by the caller without changing fold identity or source membership.

The manifest is intentionally evaluator-neutral: existing benchmark and
surgeon evaluators produce the measurements, while this contract fixes the
split, adaptation, held-out test, and evidence boundaries.

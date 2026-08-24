# Calibration batch planning

ModelSurgeon plans calibration work one batch at a time so later batches can adapt to measured memory without changing sample order or metric membership.

## Inputs

`CalibrationBatchPlanner` consumes the ordered tokenized calibration samples produced by the calibration pipeline, a `StageResourceBudget`, the current hardware inventory, and a conservative `CalibrationBatchMemoryModel`.

The planner hashes each selected sample ID, content digest, and exact token sequence into a calibration manifest digest. Resume cursors and telemetry observations are bound to that digest so evidence from a different calibration input cannot silently influence a run.

## Batch invariants

A planned batch is always a contiguous range of complete samples. The planner:

- never reorders selected calibration samples;
- never token-splits a sample to satisfy a batch limit;
- stops before `max_batch_size` or `max_batch_tokens`;
- preflights estimated RAM/VRAM against configured ceilings and known host capacity;
- fails closed when even one whole sample cannot fit.

Because only boundaries change, evaluating every emitted batch in sequence visits the same selected samples in the same order as sample-at-a-time evaluation.

## Telemetry adaptation

After a batch executes, callers may record a `CalibrationBatchObservation` with its sample range, token count, stage-local RAM/VRAM baseline and peak values, and an optional RAM/VRAM exhaustion reason.

Measured baseline-to-peak deltas raise the planner's per-token memory estimates monotonically. A memory-exhausted observation also halves the effective maximum batch size, bounded by `min_batch_size`. The next decision is then replanned from the exact sample cursor rather than rewriting earlier batches.

## Resume

`CalibrationBatchCursor` contains the manifest digest and the next whole-sample index. Persist the cursor only after the preceding batch reaches the desired checkpoint. On restart, passing that cursor causes the next batch to begin at exactly that sample index. A cursor from another manifest or beyond the selected sample count is rejected.

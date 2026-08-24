# Transactional short fine-tuning repair v1

Short repair can train either every candidate parameter or an exact canonical name
set. A hard trainable-parameter ceiling is checked before snapshots or optimizer
creation. Steps, learning rate, seed, wall time, validation patience, minimum
improvement, and allowed validation-loss increase are all bounded configuration.

The no-repair candidate validation loss is measured before training. Every step is
validated; early stopping retains the best trained weights. If no trained state
stays within the configured overfit bound, or wall budget expires, the exact
pre-repair snapshots are restored and no output checkpoint ID is created. Accepted
weights receive a content-derived child checkpoint ID whose record names the
immutable source and candidate parent.

Results include status, selected parameter count, completed steps, seed, full train
and validation histories, no-repair and repaired losses/delta, early-stop and restore
flags, wall time, peak RSS, and CUDA peaks. Exceptions also restore snapshots,
parameter trainability, and the model's original train/eval mode.

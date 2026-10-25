# GGUF exact-byte quantization planner

## Implemented boundary

`plan_gguf_quantization_budget` selects a quantization type for each tensor
under a hard output-file byte ceiling. It uses each supported codec's actual
block layout, source tensor ordering, GGUF data offset, and alignment padding.
Changing a tensor type does not change the descriptor length, so the file-size
calculation assumes the source metadata, names, dimensions, tensor order and
alignment remain fixed. The selected size is cross-checked against
`plan_gguf_output` in the tests.

Each tensor option carries an externally supplied non-negative loss estimate.
The bounded Pareto dynamic program minimizes their sum while retaining the
source codec as a zero-loss, no-rewrite option. It rejects changed codecs
without a registered writer, infeasible budgets, invalid estimates, and search
frontiers beyond configured limits.

## Research provenance

This is an independent implementation informed by RCO's discrete
budget-constrained assignment search at
[IST-DASLab/RCO `9a1e09c07d468109cbe60a1b87d5036034a79d10`](https://github.com/IST-DASLab/RCO/tree/9a1e09c07d468109cbe60a1b87d5036034a79d10/src/search/quant.py).
The upstream repository declares Apache-2.0. No upstream source code is copied
and no dependency is added. RCO's implementation optimizes discretized average
bit costs; this planner uses exact GGUF encoded bytes and layout alignment.

## Evidence and limits

Tests compare assignments against exhaustive search for a hand-checked budget
and 40 seeded tiny cases, then compare planned file sizes with the GGUF writer's
layout planner. These tests establish the discrete planner's cost accounting
and deterministic optimum for the tested cases. They do not validate tensor
encoding, additive loss estimates as predictors of model quality, physical
artifact reload, or inference performance. The planner is not yet connected to
the optimization orchestrator; physical use requires a calibration-backed
loss source, transactional tensor rewriting, runtime reload, and held-out
quality/performance evaluation.

# GGUF exact-byte quantization planner

## Implemented boundary

`plan_gguf_quantization_budget` selects a quantization type for each tensor
under a hard output-file byte ceiling. It uses each supported codec's actual
block layout, source tensor ordering, GGUF data offset, and alignment padding.
Changing a tensor type does not change the descriptor length, so the file-size
calculation assumes the source metadata, names, dimensions, tensor order and
alignment remain fixed. The selected size is cross-checked against
`plan_gguf_output` in the tests.

Each tensor option can carry an externally supplied non-negative loss estimate
or use `estimate_tensor_quantization_losses` to measure a streaming codec
round-trip error. The estimator decodes bounded source blocks, re-encodes them
with each available destination codec, and compares against the decoded source
weights. Its MSE score is summed across tensor elements for planner input. The
bounded Pareto dynamic program minimizes these additive scores while retaining
the source codec as a zero-loss, no-rewrite option. It rejects changed codecs
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
layout planner. A writer-created Q8_0 fixture is reopened, streamed through the
Q4_K/Q5_K/Q6_K codecs, and its measured round-trip losses are passed to the
planner. These tests establish codec-local measurements and discrete size
accounting on a tiny fixture. They do not establish that tensor-local MSE
predicts model quality, or validate full-model quantization, inference reload,
or deployment performance. The estimator/planner are not yet connected to the
optimization orchestrator; physical use still requires transactional tensor
rewriting, model runtime reload, and held-out quality/performance evaluation.

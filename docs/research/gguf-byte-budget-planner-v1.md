# GGUF exact-byte quantization planner

## Implemented boundary

`plan_gguf_quantization_budget` selects a quantization type for each tensor
under a hard output-file byte ceiling. It uses supported codecs' actual block
layouts, source tensor ordering, GGUF data offset, and alignment padding. The
calculation preserves source metadata except for the summary
`general.file_type`, which is regenerated to reflect the output tensor types;
if absent, it is added. Tensor names, dimensions, order, and alignment remain
fixed. Planned sizes and offsets include this metadata update and are checked
against `plan_gguf_output` before an artifact is written.

Each tensor option can use a supplied non-negative loss estimate or
`estimate_tensor_quantization_losses` to measure a streaming codec round trip.
The estimator decodes bounded source blocks, re-encodes them with each
available destination codec, and compares the decoded values against the
decoded source weights. Its MSE score is summed across tensor elements for
planner input. The bounded Pareto dynamic program minimizes these additive
scores while retaining each source codec as a zero-loss, no-rewrite option.
Unavailable codecs, infeasible budgets, invalid estimates, and search
frontiers beyond configured limits are rejected.

For a declared input axis on a 2D weight tensor, the estimator can also accept
per-input-feature activation second moments. It then scores reconstruction
error as `sum((W - Wq)^2 * E[x^2])`, the expected squared layer-output error
under a diagonal activation-covariance approximation. Raw moments are retained
so the planner can compare contributions across tensors. This is inspired by
WANDA's use of calibration activation norms with weight magnitudes; it is a
separate quantization-error objective and does not copy WANDA's pruning code.
The caller is responsible for collecting moments from representative,
revision-pinned calibration data and choosing the correct GGUF input axis.

`execute_gguf_quantization_plan` streams the selected plan through the existing
transactional GGUF writer. Changed tensors are decoded and re-encoded in
bounded chunks; unchanged tensor payloads are copied byte-for-byte. The writer
stages the file, reparses it, and atomically publishes the candidate. The
executor leaves the source unchanged and does not promote the candidate. It
also updates `general.file_type` to llama.cpp's guessed dominant output type
(`LLAMA_FTYPE_GUESSED`); this reports the largest tensor codec family without
pretending the artifact follows a named quantizer recipe such as Q4_K_M. If
the source lacks the field, the writer adds it before planning the output
layout.

## Research provenance

This is an independent implementation informed by RCO's discrete
budget-constrained assignment search at
[IST-DASLab/RCO `9a1e09c07d468109cbe60a1b87d5036034a79d10`](https://github.com/IST-DASLab/RCO/tree/9a1e09c07d468109cbe60a1b87d5036034a79d10/src/search/quant.py).
The upstream repository declares Apache-2.0. No upstream source code is copied
and no dependency is added. RCO optimizes discretized average-bit costs; this
planner accounts for exact GGUF encoded bytes and alignment.

The output file-type enum and guessed flag are aligned with
[llama.cpp `465e49b9cea78a68b9c244ffb48d0ee24a82873d`](https://github.com/ggml-org/llama.cpp/tree/465e49b9cea78a68b9c244ffb48d0ee24a82873d/include/llama.h)
and its dominant-tensor loader behavior in
[`llama-model-loader.cpp`](https://github.com/ggml-org/llama.cpp/blob/465e49b9cea78a68b9c244ffb48d0ee24a82873d/src/llama-model-loader.cpp).
The source project declares MIT; no implementation code is copied.

The activation-weighted error option is independently implemented from the
expected linear-layer output-error expression. Its research basis includes
WANDA's activation-aware importance method, inspected at
[locuslab/wanda `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`](https://github.com/locuslab/wanda/tree/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py).
That repository declares MIT; no WANDA source is copied and no dependency is
added. Its score scales weight magnitude by input activation norm for pruning;
this estimator instead weights squared quantization reconstruction error by
per-feature activation second moments.

## Evidence and limits

Tests compare assignments against exhaustive search for hand-checked and 40
seeded tiny cases, then compare planned file sizes and offsets with the GGUF
writer's layout planner. A writer-created Q8_0 fixture is reopened, streamed
through Q4_K/Q5_K/Q6_K, and its measured round-trip losses are passed to the
planner. A selected plan is then written to a candidate GGUF, reparsed, and
checked for output types, source preservation, and byte-identical copying of an
untouched tensor. These tests validate local codec behavior and physical file
construction on a tiny fixture. They do not establish that tensor-local MSE
or the diagonal activation-weighted proxy predicts model quality, or validate
task quality or deployment performance. One real-model CPU smoke experiment is
recorded in
[`gguf-byte-budget-experiment-2026-10-25.md`](gguf-byte-budget-experiment-2026-10-25.md):
the output reloaded, but its short WikiText perplexity was worse. This is
negative evidence, not a quality pass. The estimator and planner are not yet
connected to the optimization orchestrator; broader held-out quality and
deployment-performance evaluation remain necessary before promotion. A full
WikiText-2 test comparison of activation-weighted and weight-only GGUF
allocation is recorded in
[`gguf-activation-weighted-allocation-2026-10-25.md`](gguf-activation-weighted-allocation-2026-10-25.md).
At approximately equal 5% file reduction, activation weighting reduced the
PPL penalty versus weight-only allocation, but both candidates still regressed
materially against the source and failed quality retention. The result is
promising ranking evidence for one model/dataset pair, not a promotion signal
or general claim.

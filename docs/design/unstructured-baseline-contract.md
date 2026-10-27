# Wanda and SparseGPT equal-budget baseline contract

`modelsurgeon.evaluation.unstructured_baselines` freezes the v1.1 baseline
matrix for two published external methods without importing either algorithm
into ModelSurgeon. Each method is represented by a license- and revision-pinned
`UnstructuredMethodSpec` and is executed through the fail-closed subprocess
adapter.

Protocol schema v2 binds pruning calibration to WikiText's `train` split and
held-out perplexity evaluation to its separate `test` split. The v1 descriptor
is retained for audit; v2 supersedes it because v1 did not identify separate
calibration and evaluation splits.

## Matched matrix

The protocol selects the 100M Llama-family and 500M Qwen-family ladder targets,
20% and 50% unstructured sparsity, and seeds 11, 23, and 47. Every Wanda and
SparseGPT cell shares the same source checkpoint, training-split calibration
corpus/tokenizer identity, held-out test split, hardware identity, and
optimization/evaluation budgets. Cells record both split identities. The
matrix retains 24 cells and includes magnitude-mean-absolute and seeded-random
comparison controls.

Quality is held-out perplexity in nats/token with token-weighted aggregation.
Artifact size is measured in bytes; wall time and GPU time are seconds; peak
memory is bytes; and failure rate is failed repetitions divided by attempts.
No structured speedup claim is permitted for an unstructured mask.

## Support and evidence boundary

The compatibility matrix covers both selected models, both methods, and both
Safetensors and GGUF formats. Safetensors remains `unknown` until a pinned
executable produces an artifact-bound success through the competitor adapter;
GGUF is explicitly `unsupported` for these dense published baselines. The
checked-in evidence cells are `unsupported` because the external executables
are not bundled in this repository or CI. This is a diagnosed availability
state, not a quality result, and no performance claim is made.

The exact Wanda and SparseGPT source revisions and licenses are retained in the
protocol. Downstream execution must publish a successful artifact, matched
resource telemetry, confidence intervals, and any negative or failed cells.

Protocol ID: `baseline_protocol_023229364e26356db2382022fb28e844c539a2a7b37dd3e6ae99561cc06d27ce`

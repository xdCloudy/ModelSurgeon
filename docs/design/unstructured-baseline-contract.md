# Wanda and SparseGPT equal-budget baseline contract

`modelsurgeon.evaluation.unstructured_baselines` freezes the v1.1 baseline
matrix for two published external methods without importing either algorithm
into ModelSurgeon. Each method is represented by a license- and revision-pinned
`UnstructuredMethodSpec` and is executed through the fail-closed subprocess
adapter.

## Matched matrix

The protocol selects the 100M Llama-family and 500M Qwen-family ladder targets,
20% and 50% unstructured sparsity, and seeds 11, 23, and 47. Every Wanda and
SparseGPT cell shares the same source checkpoint, calibration corpus/tokenizer
identity, held-out evaluator, hardware identity, and optimization/evaluation
budgets. The matrix retains 24 cells and includes magnitude-mean-absolute and
seeded-random comparison controls.

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

Protocol ID: `baseline_protocol_d22086836d3b996a8ddad3ae78536494dd7180ffa0d7e11e9e4a77ad7d6646ae`

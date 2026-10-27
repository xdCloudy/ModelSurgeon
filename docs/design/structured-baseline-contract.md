# Structured baseline contract

`modelsurgeon.evaluation.structured_baselines` freezes four independent
structured-baseline cards: LLM-Pruner, SliceGPT, ShortGPT, and Minitron-style
pruning. Each card retains its exact upstream revision, license status, model
families, operations, checkpoint format, and recovery-stage requirement.

Schema v2 records WikiText's `train` split for calibration and its distinct
`test` split for held-out evaluation. The v1 evidence descriptor remains
available for audit; schema v2 supersedes it because v1 did not identify the
two split roles independently.

The matrix uses the 100M Llama and 500M Qwen ladder targets, physical 10% and
20% compression targets, and seeds 11, 23, and 47. Every cell carries a
training-split calibration corpus, held-out evaluation split, hardware,
budget, outcome, and reconciliation record.
Measured cells must bind requested and achieved compression, before/after
parameter and file-size counts, artifact digest, reloadability, generation
smoke, wall time, and GPU time. No metric is silently filled with a proxy.

Compatibility is explicit across model family, operation, and Safetensors/GGUF.
GGUF is unsupported for these published structured baselines. Safetensors cells
remain unknown until a pinned executable produces a reloadable, generative
artifact. Minitron is unsupported because the upstream repository does not
declare a license in its GitHub metadata. The 192 evidence cells are retained
as unsupported because external executables are not bundled in this repository
or CI; this is an availability/incompatibility record, not a performance claim.

Protocol ID: `structured_protocol_fa0479321c19ab01e80659c73ff6110e8bca475bbdb7960b112ecc1548b80db3`


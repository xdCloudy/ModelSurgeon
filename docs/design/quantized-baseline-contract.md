# Quantized baseline contract

Issue #358 preregisters matched quantization-only and
pruning-plus-quantization arms for the two permissive v1.1 model families.
The six profiles are BF16, F16, Q8_0, Q6_K, Q5_K_M, and Q4_K_M. Every arm
uses the same held-out corpus, three seeds, evaluator revision, and bounded
optimization/evaluation budgets. The combined arm has a fixed 20% structural
target; the quantization-only arm has no structural target.

Each cell retains an outcome, artifact lineage, reload/generation checks, quality,
artifact-size, runtime, RAM, and VRAM metrics, plus a three-term loss
decomposition. Missing external tooling is recorded as unsupported; it never
becomes a zero-sized artifact or an unearned quality result. Compatibility is
also recorded separately for Safetensors and GGUF.

The pinned quantizer provenance is the exact llama.cpp revision already used
by the repository's GGUF conformance boundary, under its MIT license. A future
measured cell must include the converter recipe, checksum, source and output
sizes, successful reload, and generation smoke before metrics can count.

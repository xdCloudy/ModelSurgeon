# Bounded LoRA repair v3

Post-surgery LoRA repair targets an explicit canonical set of PyTorch linear modules
on a candidate checkpoint. Base parameters are frozen; only rank-bounded A/B adapter
matrices train on the supplied finite repair set for the configured maximum steps,
seed, learning rate, scale, and dropout. Every example must produce a finite measured
loss. When validation examples are provided, they must be token-disjoint from the
training examples. The untouched adapter (step zero) is evaluated before training;
only a trained checkpoint with strictly lower held-out loss can replace it. If every
trained checkpoint is worse or tied, the zero-update adapter is restored. Baseline and
selected validation losses, whether the update improved validation, and the selected
step are recorded alongside both example digests. The first-party runtime
uses the last calibration chunk or the configured external validation corpus for
selection; it excludes the default held-out chunk from LoRA training.

Validation losses are aggregated by the number of non-ignored next-token targets
(`labels[..., 1:] != -100`) in each example. This keeps checkpoint selection aligned
with corpus-level token loss when validation sequences have different lengths or
padding. The result records the aggregation rule; training loss reporting remains
per update and is not affected.

Separate mode retains adapters around byte-identical base weights and exposes a
tensor-only adapter state dictionary suitable for safetensors. Merge mode deliberately
adds the learned low-rank delta to the in-memory candidate weights and removes the
wrappers. Neither mode writes to or aliases the immutable source checkpoint identity.

The result records exact target paths, trainable parameter count, completed and
selected steps, baseline and selected validation losses, whether training improved
validation, train and validation losses, example and optional dataset digests, seed,
deterministic adapter SHA-256, output mode, wall time, peak RSS, and CUDA
allocation/reservation peaks. Direct API callers may omit validation examples; that
compatibility path retains the final training step. Any setup or training failure
restores replaced modules and original parameter trainability before propagating.

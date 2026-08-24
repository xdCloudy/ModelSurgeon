# Bounded LoRA repair v1

Post-surgery LoRA repair targets an explicit canonical set of PyTorch linear modules
on a candidate checkpoint. Base parameters are frozen; only rank-bounded A/B adapter
matrices train on the supplied finite repair set for the configured maximum steps,
seed, learning rate, scale, and dropout. Every example must produce a finite measured
loss.

Separate mode retains adapters around byte-identical base weights and exposes a
tensor-only adapter state dictionary suitable for safetensors. Merge mode deliberately
adds the learned low-rank delta to the in-memory candidate weights and removes the
wrappers. Neither mode writes to or aliases the immutable source checkpoint identity.

The result records exact target paths, trainable parameter count, completed steps,
seed, every training loss, deterministic adapter SHA-256, output mode, wall time,
peak RSS, and CUDA allocation/reservation peaks. Any setup or training failure
restores replaced modules and original parameter trainability before propagating.

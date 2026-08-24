# Native GGUF low-rank tensor replacement

Issue #154 changes exactly one selected quantized matrix without decoding a complete model. The
caller supplies an explicit architecture family, tensor name, rank, codec registry, output disk
plan, and limits for encoded-copy chunks, decoded values, SVD workspace, and requantization.

The selected payload is decoded in codec-block chunks into bounded FP32 storage. Its GGML axis-0
layout is viewed as a row-major `(dimension 1, dimension 0)` matrix and converted to FP64 only
after a conservative workspace preflight. Exact truncated SVD records requested/effective rank,
relative Frobenius error, and maximum absolute reconstruction error.

The approximation is streamed through the existing single-tensor selective requantizer using
the original codec and byte order. Its independently decoded validation records weighted mean
squared and maximum absolute requantization error. Every other encoded tensor is direct-copied
under a byte ceiling and must retain its SHA-256. Output rediscovery must preserve the selected
shape and architecture, and the source file checksum must not change.

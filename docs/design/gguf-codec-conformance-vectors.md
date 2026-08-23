# GGUF codec conformance vectors

The suite contains one exact encoded block, decoded values, logical shape, byte order,
field spans, source URL, MIT license declaration, and pinned upstream revision for every
`ggml_type` in the native codec contract. Dense and Q8_0 cases are non-zero upstream
`gguf-py` vectors; K-quant and IQ baseline blocks use the upstream-defined all-zero
encoded block and all-zero decode, including exact physical type sizes.

Vectors are pinned to llama.cpp revision
`95b8e33e16bb9a60de780a70930ebf729db6a90a` and gguf-py quants blob
`80966b6ef1518a45b86745d94eb70d05c3c5490f`. Validation checks one-block shape,
encoded size, every named byte-field span, little-endian scalar decoding, signed Q8
packing, decoded values, and encoded SHA-256 identity. Deliberate byte swaps and field
offset changes are covered by negative tests.

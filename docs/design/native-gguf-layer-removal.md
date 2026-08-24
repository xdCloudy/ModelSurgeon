# Native GGUF transformer-layer removal

Issue #153 removes complete quantized transformer blocks without decoding or requantizing any
retained tensor. Planning requires a non-empty canonical layer set, rejects out-of-range or
all-layer removal, and produces an explicit old-to-new mapping. Removed indices map to null;
retained indices are dense and order-preserving.

Execution uses the resolved architecture contract to omit every tensor in removed blocks and
rename following `blk.N.*` tensors. Global tensors retain their names. The architecture-specific
`block_count` metadata is updated while all other metadata values, GGUF version, byte order, and
alignment are preserved.

Every retained encoded payload is read and written in bounded chunks without floating-point
materialization. Source and output tensor SHA-256 values must agree under their old/new names,
the source file checksum must remain stable, and a complete output discovery pass must reconcile
the new layer count and exact canonical tensor-name set before success is returned.

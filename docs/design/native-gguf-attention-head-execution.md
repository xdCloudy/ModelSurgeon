# Native GGUF Attention-Head Execution

Issue #152 applies the model-wide Q/K/V/O rules resolved by issue #151 to a new
GGUF without materializing a floating-point model.

## Execution paths

Every source tensor is checked against its validated name, dimensions, and exact
codec before output begins. Changed Q/K/V outer-axis heads are omitted while
retained encoded rows are copied directly. O-axis removals copy retained encoded
blocks when every touched block is removed completely. Unaligned but
block-representable O removals decode, filter, and requantize one row at a time.

The repack path obeys simultaneous encoded-chunk, validation-value,
requantization-working-memory, and total row-working-memory ceilings. Every
emitted block is structurally validated and round-trip decoded. Configured
maximum absolute and mean-squared error ceilings fail the staged operation before
publication.

## Metadata and transaction behavior

The source GGUF version, endianness, alignment, tensor order, tensor names, and
GGML type IDs are preserved. Query-head and KV-head counts are updated, while
explicit key/value head lengths are added or replaced so runtime tensor geometry
does not fall back to `embedding_length / new_head_count`.

Output uses the tensor-boundary resumable writer. Untouched tensors, including
unchanged GQA/MQA K/V projections, are copied byte-for-byte and verified by
SHA-256. The published file is rediscovered and must match the new head counts,
fixed head lengths, and every Q/K/V/O destination shape.

The result records output discovery, untouched hashes, per-row/chunk
requantization errors, and peak row working bytes. Runtime llama.cpp command and
forward-pass provenance are layered on by the dedicated generated-GGUF validation
workflow rather than being hidden inside mutation execution.

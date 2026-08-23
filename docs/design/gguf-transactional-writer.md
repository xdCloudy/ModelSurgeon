# Transactional streaming GGUF writer

The writer plans metadata, tensor descriptors, aligned relative offsets, and final
file size before consuming any tensor payload. Callers preflight that size with the
disk planner. Metadata values retain their exact scalar or array type, v2 remains
little-endian, and v3 can preserve little- or big-endian source order.

Output is created as an exclusive staging file beside the destination. Header and
zero alignment padding are followed by one bounded chunk at a time; chunks are never
collected into a tensor-sized allocation. Every tensor must supply exactly its planned
encoded byte count. The staged file is flushed and fsynced, reopened through the GGUF
parser to validate tensor offsets and sizes, and only then atomically published. Existing
destinations are not overwritten. Publication uses an atomic no-replace hard link in
the destination directory, closing the race between the initial path check and final
publication.

Failure or cancellation removes the private staging file and leaves the destination
absent. A successful result records the physical tensor layout, final byte size, and
SHA-256 digest as integrity provenance. The source checkpoint is never opened for
writing by this API.

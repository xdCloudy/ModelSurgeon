# GGUF container parser contract

`MemoryMappedGGUF` opens a v2 or v3 GGUF source read-only and builds an immutable
metadata and tensor-descriptor index. Indexing reads only header, metadata, and
descriptor bytes. Tensor payload pages are not copied, decoded, exposed as arrays,
or retained in Python objects. A 40 GiB sparse conformance fixture therefore has
constant payload memory and index memory proportional only to metadata and tensor
counts.

The parser detects little-endian v2/v3 and byte-swapped v3 from the version word.
It preserves metadata order, explicit value and array-element types, and raw source
spans so a later transactional writer can copy unchanged records exactly. Tensor
descriptors report physical names, storage-order dimensions, exact `ggml_type`,
relative and absolute offsets, and encoded byte sizes from the pinned codec layouts.

Inputs fail closed before payload access when versions or types are unsupported,
resource counts exceed configured bounds, UTF-8 or lengths are malformed, names or
keys repeat, dimensions are invalid, block divisibility fails, alignment is unsafe,
offset arithmetic overflows, byte ranges leave the file, or tensor ranges overlap.
The public bounded `raw_bytes` method returns copies rather than live views, keeping
the mmap handle private to the adapter boundary.

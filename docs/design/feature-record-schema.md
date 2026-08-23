# Feature record and precision provenance

Feature schema v1 stores one canonical component ID, scalar or ordered vector value,
declared dtype, versioned extractor identity, optional deterministic sample context,
and primitive extension metadata. Records contain no framework tensor objects and are
directly JSON serializable.

Precision provenance distinguishes features computed directly over encoded quantized
blocks, features computed after local bounded dequantization, and high-precision
sources. Quantized paths require the exact tensor codec rather than a file recipe.
Locally dequantized values additionally require finite measured absolute/relative
error, reference dtype, and method. Sample-dependent features retain dataset revision,
split, ordered unique sample IDs, preprocessing version, and tokenizer revision.

Unknown future facts belong in the primitive metadata map. Changing required fields,
value semantics, or precision categories requires a new schema version.

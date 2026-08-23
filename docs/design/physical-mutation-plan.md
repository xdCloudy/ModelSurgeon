# Physical mutation plan compiler

The physical compiler consumes an already closed transactional mutation plan,
a complete adapter-supplied physical tensor descriptor scope, axis-removal
intents, metadata updates, and explicit post-surgery identities. It validates
the entire request before any changed tensor is allocated.

Each edit records a format-neutral component ID and opaque tensor locator, old
and new shapes, compact removed-index transforms, and exact adapter-estimated
old/new storage. Every affected physical descriptor must have an edit, every
edit must belong to the mutation closure, and every affected component must have
an explicit identity mapping. Metadata keys and all collections are canonical.

The compiler derives the total storage delta and requires it to match the base
mutation plan. Codec-specific block and axis representability remain a separate
validation layer, allowing this plan schema to serve Hugging Face, safetensors,
and GGUF adapters without embedding a storage format.

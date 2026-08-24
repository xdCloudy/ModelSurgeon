# Coordinated hidden-dimension feasibility study

Issue #155 deliberately stops short of physical mutation. The bounded evaluator requires exactly
one reconciled Llama discovery and one reconciled Qwen discovery. It inventories every physical
tensor axis declared as input, output, or hidden features; separately records all normalization
tensors; captures attention/KV head dimensions; and computes the least-common-multiple of codec
block sizes for every affected contiguous axis-0.

Both current family contracts prove many tensor couplings, but neither physical graph proves all
consumers needed to shrink a model-wide embedding dimension:

- Rotary dimension and rotary-frequency configuration consumers are not graph nodes.
- Token-embedding/output-weight tying provenance is absent from GGUF descriptors.
- Tokenizer and runtime configuration consumers of embedding width are outside the physical plan.

Consequently both family assessments contain no feasible physical operation class, explicit
rejection reasons, and `physical_mutation_implemented = false`. This fail-closed result prevents
the presence of a large tensor inventory from being mistaken for complete coupling proof.

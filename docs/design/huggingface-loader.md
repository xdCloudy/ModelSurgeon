# Safe Hugging Face loader

`load_causal_lm` is the single causal-language-model loading boundary. It imports the optional PyTorch and Transformers dependencies lazily, disables repository code by default, and places the complete model on CPU unless another supported device-map strategy is explicitly requested.

The immutable `HuggingFaceLoadRequest` controls revision, dtype, placement, local-only access, and low-CPU-memory loading. The result keeps the loaded model separate from a primitive `HuggingFaceLoadProvenance` record containing the requested source, the resolved Hub commit (or explicit/local revision), and every effective loader option.

An unpinned remote load fails after loading if Transformers does not expose its resolved commit. This prevents a mutable Hub branch from entering downstream artifacts without a reproducible identity. Enabling `trust_remote_code` is an explicit security decision and is never inferred from model metadata.

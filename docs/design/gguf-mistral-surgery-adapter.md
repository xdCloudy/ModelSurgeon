# Mistral GGUF surgery adapter

The Mistral adapter supports the native `mistral` GGUF architecture and the
legacy `llama` architecture/prefix emitted by some converters when the model
family has already been resolved explicitly as Mistral. The chosen prefix is
retained so metadata edits cannot cross naming variants.

A positive `<prefix>.attention.sliding_window` value is required and recorded
with validated MHA/GQA geometry. Every block must contain normalization,
Q/K/V/O, and gate/up/down physical weights and complete attention-head, KV-head,
and MLP-channel coupling groups. Non-divisible GQA, absent window metadata,
unknown architecture variants, and future contract versions fail before a
surgery view is returned.

Compatibility tests pin both native and legacy-prefix fixtures with identical
physical tensor topology and verify that their metadata namespaces remain
distinct.

# Qwen GGUF surgery adapter

The Qwen adapter exposes an explicit physical-surgery contract for dense
`qwen2` and `qwen3` GGUF architectures. Each supported fixture must first pass
generic GGUF discovery, shape reconciliation, and component-graph validation.
The resulting adapter records the architecture generation, MHA/GQA geometry,
per-layer tensor mappings, and attention-head, KV-head, and MLP-channel coupled
axes.

Architecture differences are a finite data contract rather than aliases hidden
inside planning code. `qwen2moe` and `qwen3moe` are recognized as MoE variants
but native surgery is rejected until expert and router tensor mappings are
defined. Unknown variants and future contract versions also fail closed.

Dense layers require input and post-attention normalization, Q/K/V/O weights,
and gate/up/down MLP weights. Query-head counts must divide evenly into KV heads
before a surgery view is returned. The compatibility fixture covers both dense
generations and both explicitly unsupported MoE spellings.

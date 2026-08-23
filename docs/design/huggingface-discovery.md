# Hugging Face component discovery

The Hugging Face discovery boundary converts a selected Llama, Qwen, Mistral, or Gemma family model into canonical `ComponentDescriptor` records. Framework module ordering is not trusted: physical module and parameter records are sorted by canonical component ID, duplicate canonical paths fail, and tied parameters are counted once.

Every record has a non-empty semantic kind. Common transformer modules are classified as layers, attention, projections, MLPs, embeddings, normalization, experts, or routers; unknown modules remain explicit `module` records. Parameters are children of their owning canonical module and carry their element count.

Logical attention heads, KV heads, and MLP channels are generated lazily from validated configuration dimensions, including explicit collection nodes so every logical record has a canonical parent. This avoids allocating a descriptor tuple proportional to all channels in a large model. The discovery summary reconciles the unique parameter element counts and records physical and logical component totals for downstream graph construction.

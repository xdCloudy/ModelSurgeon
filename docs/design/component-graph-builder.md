# Transformer component graph builder

`build_component_graph` consumes framework-neutral discovery records and creates the canonical v1 component graph. Input order is irrelevant. Every non-root node must have its canonical parent, and the builder emits consistent inverse `parent` and `child` edges.

For supported dense Llama, Qwen, Mistral, and Gemma blocks, the builder recognizes Q/K/V/O attention projections, gate/up/down MLP projections, input and post-attention normalization, and two explicit residual-path nodes. Paired `produces` and `consumes` edges describe the block flow from normalization through attention, the first residual, normalization, the MLP, and the second residual.

Attention projection and MLP projection sets each form a complete pairwise coupling closure and reference a versioned mutation constraint. A partially recognized projection set, missing norm, duplicate component, missing parent, or conflicting edge fails closed. This keeps unsupported architecture variants from producing a plausible but unsafe mutation graph.

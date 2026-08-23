# Llama GGUF surgery adapter

The Llama physical-surgery adapter is a versioned, fail-closed view built on the
generic GGUF discovery graph. It accepts only an explicitly resolved `llama`
family and `general.architecture=llama` contract. Loading first validates all
required metadata, physical tensor names, ranks, dimensions, and block indices.

The adapter exposes per-layer Q/K/V/O, normalization, and gate/up/down weight
mappings together with the declared attention-head, KV-head, and MLP-channel
coupling groups. MHA and GQA geometry includes the query and KV widths, head
width, and the number of query heads per KV head. Non-divisible GQA layouts are
rejected before a physical edit is planned.

`output.weight` remains optional because tied-output Llama files can omit it.
The token embedding and output normalization plus all nine structural weights
per block are required. Unknown tensors, missing required tensors, unsupported
contract versions, and out-of-range layer requests fail explicitly. Metadata
updates and layer renames delegate to the same architecture contract used for
discovery so planning and writing cannot disagree on naming.

The fixture in `tests/fixtures/llama_gguf_surgery_v1.json` pins the supported
metadata and physical tensor topology for compatibility regression tests.

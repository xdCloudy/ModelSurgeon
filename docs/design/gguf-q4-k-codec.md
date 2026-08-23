# Q4_K codec and recipe metadata

`Q4_K` is one physical 256-value tensor codec: two endian-aware float16 deltas,
twelve packed 6-bit scale/minimum values, and 128 paired low/high nibbles. Its encoder,
decoder, validation, and range API operate only on that exact 144-byte super-block.

`Q4_K_S` and `Q4_K_M` remain whole-file recipes represented by `general.file_type`
values 14 and 15. Both resolve to physical `Q4_K` tensor identity; they do not create
variant block layouts or permit codec substitution. This preserves practical recipe
provenance while native edits continue to dispatch solely from each tensor's
`ggml_type`.

# Q6_K GGUF codec

Q6_K is implemented as its own 256-value, 210-byte super-block codec. Its physical
layout is fixed to 128 low-bit bytes, 64 high-bit bytes, sixteen signed subgroup scales,
and one endian-aware float16 super-block scale. No other K-quant layout is accepted or
substituted.

Encoding follows the pinned llama.cpp reference grouping and scale refinement, then
recomputes 6-bit levels against the stored float16 scale before packing the four
32-value lanes. Decoding reverses those exact low/high-bit lanes and subgroup-scale
assignments. Range access is expressed only in whole super-blocks; partial buffers,
escaping ranges, non-finite scales, and wrong-size destinations fail closed.

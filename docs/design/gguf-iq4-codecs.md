# Native IQ4 codecs

ModelSurgeon v1 writes exactly the two IQ-family tensor types selected by the
revision-pinned support study: `IQ4_NL` and `IQ4_XS`. Dispatch is by concrete
`ggml_type`; it never silently substitutes another IQ layout. All read-only and
deferred IQ types fail before an output buffer is mutated.

`IQ4_NL` stores 32 values in 18 bytes: one endian-aware binary16 scale followed
by 16 bytes whose low and high nibbles index the pinned nonlinear table
`[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`.
The low nibbles encode values 0–15 and the high nibbles values 16–31.

`IQ4_XS` stores 256 values in 136 bytes. It combines one endian-aware binary16
super-block scale, eight signed six-bit local scale codes split across two high
bytes and four low-nibble bytes, and eight 32-value IQ4_NL index groups. The
six-bit codes use a bias of 32. Multi-byte numeric fields follow the GGUF byte
order; packed nibble order is fixed by the physical layout.

Both encoders operate on complete physical blocks, reject non-finite values and
wrong-sized destinations, and expose quantization error measurement. Identity
metadata pins the implementation to the same upstream llama.cpp revision used
by the conformance layouts and vectors.

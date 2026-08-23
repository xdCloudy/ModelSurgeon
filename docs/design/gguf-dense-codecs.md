# Dense GGUF codecs

F32, F16, and BF16 are implemented as exact-type bounded block codecs with block size
one. They encode and decode arbitrary—even odd—element counts directly into caller-owned
buffers and preserve GGUF v3 little- or big-endian byte order. F32 uses IEEE binary32,
F16 uses IEEE binary16, and BF16 encoding applies round-to-nearest-even before storing
the high 16 binary32 bits.

Validation rejects partial scalar bytes before decoding. Encoding requires an exact-size
writable destination and finite representable input; it never allocates a tensor-sized
return value. Dense codecs expose the shared error estimator and exact registry identity
pinned to the native GGUF upstream revision.

# Q8_0 GGUF codec

Q8_0 operates on exact 32-value blocks stored as one endian-aware float16 scale followed
by 32 signed int8 quants. Encoding follows the pinned upstream algorithm: divide the
maximum absolute value by 127, quantize with C `roundf` half-away-from-zero behavior,
and store the rounded float16 scale without changing quant decisions.

All operations use caller-owned buffers and complete encoded blocks. Block-range access
accepts logical block offsets and counts only, preventing byte-unaligned slices. Partial
blocks, escaping ranges, non-finite values or scales, and wrong-size destinations fail
before producing decoded output. `encode_with_error` reports operation counts and stable
MAE, MSE, maximum error, and reference norm for every encoded range.

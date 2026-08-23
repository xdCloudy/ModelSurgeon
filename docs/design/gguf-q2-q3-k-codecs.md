# Q2_K and Q3_K codecs

Q2_K and Q3_K are independent 256-value tensor codecs. Q2_K stores sixteen combined
4-bit scale/minimum values, 64 packed 2-bit payload bytes, and two float16 deltas in 84
bytes. Q3_K stores a 32-byte sign/high mask, 64 packed low bits, sixteen signed 6-bit
scales packed into twelve bytes, and one float16 delta in 110 bytes.

Each codec owns separate validation, encode, decode, error, and whole-super-block range
operations. A buffer valid for one physical size is rejected by the other; there is no
K-family fallback or shared interpretation. Endian conversion applies only to the
declared float16 fields while packed bit lanes retain their byte-defined ordering.

# Q5_K codec and recipe metadata

`Q5_K` is the only physical tensor codec in this family: one 256-value block contains
two float16 deltas, twelve packed 6-bit scale/minimum values, 32 high-bit bytes, and 128
low-nibble bytes. Encoding and decoding preserve that exact layout, endian-aware deltas,
and whole-super-block range boundaries.

`Q5_K_S` and `Q5_K_M` are whole-file quantization recipes, not alternate `ggml_type`
values. `general.file_type` values 16 and 17 resolve those recipes for provenance while
both retain the exact `Q5_K` tensor identity. Recipe metadata can therefore influence
which tensors a broader quantization campaign selects, but it can never substitute a
different block codec during native surgery.

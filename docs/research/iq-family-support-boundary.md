# Native IQ-family surgery support boundary

The v1 native-write boundary is deliberately narrow. `IQ4_NL` is priority 1 because it
uses a small fixed nonlinear table; `IQ4_XS` is priority 2 because it composes that table
with explicit super-block scales. These are the supported implementation targets for
#141.

`IQ2_XXS`, `IQ2_XS`, and `IQ2_S` are read-only: pinned gguf-py can decode them for
analysis, but native encoding requires large generated grids, sign tables, search logic,
and often importance data that are not yet independently pinned. `IQ3_XXS`, `IQ3_S`,
`IQ1_S`, and `IQ1_M` are deferred because their generated codebooks and scale/index
packing carry still higher implementation and conformance risk.

The machine-readable matrix records every current IQ tensor type, codebook dependencies,
upstream C and gguf-py encode/decode availability, priority, revision, blob, and decision.
It is exhaustive against `QUANT_LAYOUTS`. Read-only and deferred writes fail explicitly;
they never fall back to another IQ type, a K-quant, or an external whole-file recipe.

The decision is pinned to llama.cpp
`95b8e33e16bb9a60de780a70930ebf729db6a90a` and gguf-py quants blob
`80966b6ef1518a45b86745d94eb70d05c3c5490f`. It must be revisited with new vectors and
generated-table provenance before expanding the native-write set.

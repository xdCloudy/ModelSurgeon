# Gemma GGUF surgery adapter

The finite Gemma compatibility contract supports the dense Gemma v1 GGUF
topology with RMS normalization, Q/K/V/O attention projections, gate/up/down
MLP projections, and explicit head, KV-head, and MLP coupling groups. GQA
dimensions must be divisible and tied output weights may remain physically
absent.

`gemma2` and `gemma3` are recognized but fail closed: their extra normalization,
attention-dimension, and local/global-attention rules are not represented by the
v1 physical mapping. Recording those variants explicitly prevents the common
tensor-name rules from being mistaken for a native surgery support claim.
Unknown variants and future contract versions are also rejected before planning.

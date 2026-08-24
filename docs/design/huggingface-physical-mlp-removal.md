# Hugging Face physical MLP channel removal

Issue #119 removes the same canonical intermediate-channel indices from every standard
gated MLP layer. A global Hugging Face `intermediate_size` cannot represent asymmetric
per-layer widths, so missing layers, noncanonical indices, shape disagreement, or removal
of the entire axis fails before modifying modules.

For every layer, gate/up weight rows and optional biases are retained by index while down
weight columns use the same retained order. Replacement modules preserve the original
module class, dtype, device, parameter gradient flags, and unaffected state; linear
`in_features`/`out_features` metadata and global configuration are updated. The result
records old/new shapes and reconciled whole-model parameter counts.

Mask equivalence is mathematical but differently shaped GEMMs may change floating-point
reduction order. Tests require tight agreement on a gated reference model. Real-model
evidence reports max, mean, and relative-L2 logit error before repair, then requires exact
forward equality after save/reload.

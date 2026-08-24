# Hugging Face low-rank Linear replacement

Issue #122 replaces explicitly selected `torch.nn.Linear` weights with two physical Linear
factors derived by truncated SVD. A rank `r` replacement for an `m` by `n` weight contains
`r(m+n)` weight parameters and performs `2r(m+n)` multiply/add operations per token. Bias,
when present, belongs to the output factor and is counted in both the original and replacement
parameter totals.

Selection paths must be unique and canonically sorted. Every path and rank is validated before
the first mutation, so invalid multi-module requests fail without partially changing the model.
The SVD runs in float64 on CPU under explicit matrix-element and estimated-workspace ceilings;
the factors are then copied back to the original module device and dtype.

The report records requested and effective rank, relative Frobenius reconstruction error, and
old/new parameter and per-token FLOP counts for each actual replacement. The factorized state
uses `down` and `up` keys and therefore requires the same structural replacement before loading
that state into a standard Hugging Face architecture; it is not falsely advertised as a stock
`from_pretrained` checkpoint.

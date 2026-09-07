# Mutation property suite

`modelsurgeon.properties` provides the deterministic CPU contract for generated
structural and transactional property cases. The default campaign has 2,048
seeded cases and covers every writable native GGUF codec currently exposed by
the adapter (`F32`, `F16`, `BF16`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`,
`Q8_0`, `IQ4_NL`, and `IQ4_XS`) plus HF mutation sequences, native attention,
layer, low-rank, MLP-channel, and direct-copy operations.

Cases retain their seed, failure injection stage, operation, target, and
dimensions. A failed evaluator is shrunk deterministically by halving each
dimension while the failure persists. The original and shrunk case are both
retained, so a counterexample becomes a reproducible regression rather than a
transient random failure. Unsupported codec/operation cells and timeouts are
retained explicitly.

The suite is a harness boundary: concrete evaluators can assert source
immutability, transaction rollback, idempotence, remap composition, parameter
and storage reconciliation, direct-copy identity, error bounds, and working
memory without changing the evidence contract.


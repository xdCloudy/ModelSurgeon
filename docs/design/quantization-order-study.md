# Quantization-order study

The quantization-order contract compares four matched arms:

1. quantization only;
2. surgery then quantization;
3. quantization then surgery; and
4. high-precision surgery.

Every cell shares the source digest, model family, compression level, codec,
corpus, evaluator, runtime, seed, and achieved structure identity. A measured
cell must also publish an artifact digest and all five common metrics:
`quality`, `artifact_size_bytes`, `decode_tokens_per_second`,
`peak_ram_bytes`, and `optimization_seconds`.

The comparison reports three reconciled deltas: quantization effect compares
high-precision surgery with surgery-then-quantize; surgery effect compares
quantization-only with quantization-then-surgery; and interaction effect
compares the two mixed-order arms. An arm is preferred only when all four
cells are measured. Missing tools, failed executions, and non-equivalent
structures remain explicit unsupported or inconclusive outcomes.

The preregistered minimum matrix covers two families, two compression levels,
Q8_0/Q6_K/Q4_K_M where supported, and three seeds. The schema intentionally
does not turn unsupported codecs or changed feasible edit sets into a numeric
claim.

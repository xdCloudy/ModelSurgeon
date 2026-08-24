# Architecture compatibility

This table is generated from the schema-versioned capability rules in
`modelsurgeon.evaluation.architecture_compatibility`. Its matrix ID is
`36d88705abec4eb6f6008dfecedab92ea9923efd082fd2fe94a1235ada171d79`.

- **V** — verified support with focused tests and the declared real/structural evidence.
- **E** — experimental implementation with focused structural tests, but no matching
  family/profile real-checkpoint proof. It is not a supported claim.
- **—** — deliberately unsupported; callers fail closed.
- **?** — unknown because no matching fixture/evidence exists. Unknown never means supported.

<!-- BEGIN GENERATED MATRIX -->
| Operation | HF dense llama | HF dense qwen | HF dense mistral | HF dense gemma | GGUF F16 llama | GGUF F16 qwen | GGUF F16 mistral | GGUF F16 gemma | GGUF Q4_K_M llama | GGUF Q4_K_M qwen | GGUF Q4_K_M mistral | GGUF Q4_K_M gemma |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analysis | V | V | V | V | V | V | V | V | V | ? | ? | ? |
| MLP mask | E | E | E | E | — | — | — | — | — | — | — | — |
| attention mask | E | E | E | E | — | — | — | — | — | — | — | — |
| layer mask | E | E | E | E | — | — | — | — | — | — | — | — |
| physical MLP | V | E | E | E | E | E | — | — | V | E | — | — |
| physical attention | V | E | E | E | E | E | — | — | E | E | — | — |
| physical layer | V | E | E | E | E | E | E | E | E | E | E | E |
| low rank | V | E | E | E | E | E | E | E | E | E | E | E |
| checkpoint write | V | E | E | E | E | E | E | E | V | E | E | E |
<!-- END GENERATED MATRIX -->

## Interpretation and constraints

HF physical implementations currently require the canonical `model.layers` module layout.
Only Llama/SmolLM2 has real save/reload evidence for every physical operation; other family
cells remain experimental even where the same shape-level implementation and family discovery
tests pass.

GGUF masking is unsupported because GGUF is an offline container rather than a live execution
session. Native MLP and attention removal are intentionally limited to Llama and dense Qwen.
Layer removal is byte-preserving, while low-rank replacement requires an exact native codec.
A whole-file Q4_K_M recipe is mixed-codec: storage-only tensors are eligible only for aligned
unchanged-codec copies, never float repack or codec substitution.

The Llama Q4_K_M physical-MLP and checkpoint cells are verified by the real model-wide proof in
`docs/research/v0.7-native-q4-k-m-mlp-proof.md`. Q4_K_M analysis for the other three families
remains unknown because no pinned quantized family fixture has passed the external runtime gate.

## Generation and regression behavior

The matrix generator produces every family × profile × operation cell and rejects omissions or
duplicates. Verified and experimental cells must cite automated-test evidence. The focused suite
also asserts the state totals and exact high-risk boundaries, preventing an unknown,
experimental, or unsupported cell from silently becoming a support claim.

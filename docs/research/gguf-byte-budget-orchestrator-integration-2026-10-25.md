# GGUF byte-budget orchestration smoke run

## Integration

The optional `gguf_byte_budget` quantization method connects the byte-budget
planner and transactional writer from PR #737 to the native GGUF optimization
stage. The optimizer runs it after candidate selection, validates the written
GGUF with the configured llama.cpp runtime, compares held-out perplexity with
the immutable source, and applies the existing quality and disk constraints.
Rejected or infeasible plans leave the currently selected artifact in place and
publish evidence for the attempted result.

This is a smoke run of the quantization stage starting from the pinned source
artifact as the selected candidate. It isolates the new stage; it does not
measure a candidate that has already undergone structural surgery.

## Physical result

| Measurement | Source | Quantized candidate |
| --- | ---: | ---: |
| GGUF bytes | 105,454,144 | 100,145,760 |
| File reduction | — | 5.034% |
| Held-out WikiText-2 PPL | 19.8433 ± 1.28446 | 24.7233 ± 1.59424 |

The candidate changed 18 tensors: 4 from Q4_K to Q3_K and 14 from Q6_K to
Q3_K. It reloaded and completed `llama-cli` validation successfully. The
perplexity delta was +4.88 and measured quality retention was 0.8026. The
existing 0.98 minimum-retention requirement and 0.05 quality-profile guard
both failed, so the optimizer **rejected** this smaller candidate and retained
the source artifact. This is a negative quality result, not evidence of a
quality-preserving compression win or faster inference.

The recorded planner estimate was decoded-weight MSE 19,230.6565 over 30
eligible tensor estimates; the budget search explored 6,405 states. The
physical candidate SHA-256 was
`ab390de77e83d183c2fe725578e08e8420fab67f394133bd22752018a4a10d8f`.

## Provenance and limits

- Source: `unsloth/SmolLM2-135M-Instruct-GGUF`, revision
  `9e6855bc4be717fca1ef21360a1db4b29d5c559a`, source SHA-256
  `ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747`.
- Evaluation text: Salesforce WikiText-2 raw test split, content SHA-256
  `a1ac9f35d8d3014f89b8d5cb4cfd5e55516c0126b3a8de75447b7d3084fc5011`.
- Runtime: llama.cpp commit `c7fc50941`, CPU, 8 threads, context/batch/
  microbatch 512, 8 perplexity chunks.
- The source model and data remain external. The physical observation is
  preserved in the local optimization evidence store as observation
  `obs_4424af070267451c9b3c76f7562442350aead0b2b9a5eb11bee8a7ee31cd9743`
  with record SHA-256
  `dba6eadc63a310f4102afca210d38c245927a7bf316e6599343064fb05a72363`.
- No latency, throughput, peak-memory, or task-accuracy claim is made. The
  added stage measures perplexity and artifact validity; deployment benchmarking
  remains responsible for measured runtime-cost constraints.

The planner implementation and its source-code provenance are documented in
PR #737 and `gguf-byte-budget-planner-v1.md`. This change integrates that
first-party ModelSurgeon implementation; it introduces no new optimizer
dependency or copied upstream code.

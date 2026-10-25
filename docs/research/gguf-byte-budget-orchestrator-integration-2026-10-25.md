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

## Activation-weighted follow-up

The stage also supports an explicit experimental activation-weighted loss
metric when given a bounded activation-moment manifest whose source digest
matches the exact selected GGUF. On the same source, target byte ceiling, held-
out text, and llama.cpp build, this objective produced a 100,125,024-byte
candidate (5.053% smaller) and changed 24 tensors. Its PPL was 21.681 ± 1.40764
versus 24.7233 ± 1.59424 for weight-MSE under the same 8-chunk runtime, reducing
the measured PPL penalty from +4.88 to +1.8377. This is a 62.3% smaller penalty
on this single model/corpus pair, not a generalization or quality-preservation
claim.

The activation-weighted artifact reloaded successfully, but its 0.9152 measured
quality retention still missed the existing 0.98 minimum and the 0.05
quality-profile guard. The optimizer rejected it and retained the source.
Activation moments were source-bound to SHA-256
`ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747`; their
manifest SHA-256 is
`2dd1e613509d58b89e44c7db178c614015697d46884954a9daf97d0ecbfa0584`. The
candidate SHA-256 is
`905a45028be5b2f99d7a297952ae28fe6bd68f368057bc1955e5c4e53664be24`. The
local physical evidence is observation
`obs_a1e4e3bc8f5da5d4c05008e048460b199ae5d4110544920fb974145f721caeb2`,
record SHA-256
`78da999b84b2efe24fb302d692b4d6636a6bc5693b16fddf8b30726794fad17b`.

## Mixed-codec candidate-set follow-up

The runtime and research benchmark now accept an explicit set of cheaper
per-tensor codec alternatives. This lets the exact byte planner allocate
different quant types to different tensors under one whole-file ceiling;
the existing `target_type` setting remains the single-type shorthand. On the
same source, activation moments, held-out text, and llama.cpp build, two
mixed-codec searches did **not** beat the Q3_K-only result:

| Search | Candidate bytes | PPL ± uncertainty | PPL delta | Changed tensors |
| --- | ---: | ---: | ---: | ---: |
| Q3_K only (reference) | 100,125,024 | 21.681 ± 1.40764 | +1.8377 | 24 |
| Q2_K, Q3_K, Q4_K, Q5_0, Q5_K, Q6_K | 100,180,320 | 22.0409 ± 1.43180 | +2.1976 | 34 |
| Q2_K, Q3_K, Q4_K | 100,111,200 | 22.4142 ± 1.44659 | +2.5709 | 22 |

All three candidates loaded and completed the same eight-chunk CPU perplexity
evaluation. The planner explored 68,503 states for the six-type search and
24,101 for the three-type search. Mixed quantization therefore remains an
optional search capability, not a demonstrated quality improvement. The
existing quality gate would reject all three on this profile. A separate
order-balanced CPU/RTX 3060 throughput check also found no reliable speedup;
its confidence intervals include zero for every candidate and workload. See
[`gguf-mixed-codec-throughput-2026-10-25.md`](gguf-mixed-codec-throughput-2026-10-25.md).
The six-type candidate SHA-256 is
`1068294ece77b4e9dc2e33fb698dcce4e4ed480dc23419accd7d21b306e4499d`, and the
three-type candidate SHA-256 is
`d3bb59bcb47b4ecaf4b338f651c7f6f2fb05616c62f4efe0925339cf4f8aae7d`.

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
- The optimizer stage measures perplexity and artifact validity, not latency,
  throughput, peak-memory, or task accuracy. A separate paired deployment
  benchmark found no reliable throughput change on this small model; runtime
  cost claims remain limited to the hardware and workload recorded there.

The planner implementation and its source-code provenance are documented in
PR #737 and `gguf-byte-budget-planner-v1.md`. Activation weighting is an
experimental diagonal-covariance proxy documented in
`gguf-activation-weighted-allocation-2026-10-25.md`; missing or shape-mismatched
moment entries preserve the source tensor codec. The optimizer requires a
source-bound manifest and still uses physical perplexity gates to decide
promotion. This change introduces no new optimizer dependency or copied
upstream code.

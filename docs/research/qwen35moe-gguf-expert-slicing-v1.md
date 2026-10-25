# Qwen3.5 MoE GGUF routed-expert slicing: implementation note

Status: low-level experimental adapter; not exposed as an autonomous optimizer
scope. Quality retention and deployment improvement remain unproven.

## Upstream study and acquisition decision

| Source and revision | Component | License | ModelSurgeon equivalent and limits | Acquisition decision |
| --- | --- | --- | --- | --- |
| [llama.cpp `qwen35moe.cpp`, `74bca3950054eed1795f525cb9cf06bfe796492d`](https://github.com/ggml-org/llama.cpp/blob/74bca3950054eed1795f525cb9cf06bfe796492d/src/models/qwen35moe.cpp); [tensor mappings at the same revision](https://github.com/ggml-org/llama.cpp/blob/74bca3950054eed1795f525cb9cf06bfe796492d/gguf-py/gguf/tensor_mapping.py) | Qwen3.5 MoE GGUF tensor names, routed-expert axes, metadata, and runtime interpretation | MIT | Existing native GGUF writer and opaque-tensor copy path existed; no Qwen3.5 routed-expert surgery contract existed. The general dense Qwen path is not valid for this hybrid MoE architecture. | **REIMPLEMENT** the narrow layout checks and slicing using ModelSurgeon's transactional GGUF writer. Upstream code was inspected, not copied. |
| [Crucible REAP, `dacdb43eef0630de6dafe5c54bdbd2961ea6295`](https://github.com/peonist-ai/crucible/blob/dacdb43eef0630de6dafe5c54bdbd2961ea6295/packages/crucible/crucible/methods/reap.py) | Router-weight and activation-informed expert importance | Apache-2.0 | No MoE expert ranker existed. Current router-column L2 is a static, activation-free baseline and does not reproduce REAP. | **REIMPLEMENT** only the explicitly labeled static baseline for a first physical format test. Study REAP-style router/activation scores before candidate ranking is wired into optimization. No source was copied. |

The pinned llama.cpp implementation declares `ffn_gate_inp.weight` with logical
dimensions `(embedding_length, expert_count)`. Routed gate/up/down tensors put
the expert dimension last; Qwen3.5 also has shared-expert weights and hybrid
linear-attention state. The implementation changes only routed gate/up/down
payloads, routed router columns, and `qwen35moe.expert_count`. It retains
`expert_used_count`, shared experts, attention/SSM tensors, MTP tensors, and
all other metadata. Encoded quantization blocks are copied byte-for-byte; the
operation does not dequantize and requantize expert weights.

## Acquisition matrix entry

| Field | Assessment |
| --- | --- |
| Component | Uniform per-layer routed-expert slicing for exact `qwen35moe` GGUF layout; static router-column squared L2 ranking as an exploratory baseline. |
| Current behavior | The existing first-party GGUF optimizer supports dense Llama and Qwen MLP-channel removal only. It does not optimize MoE expert sets. |
| Technical advantage | Can physically remove selected quantized expert blocks without loading a full model into RAM or changing the encoding of retained weights. Fail-closed tensor geometry avoids treating the hybrid model as dense Qwen. |
| Hardware and dependencies | Bounded streaming through the existing GGUF reader/writer; no Torch, CUDA, or added runtime dependency required for slicing. Quality and inference evaluation still requires a compatible pinned llama.cpp build and a held-out corpus. |
| Supported architecture | Exact GGUF architecture string `qwen35moe` and validated routed tensor names/shapes only. One local Qwen3.6-derived `qwen35moe` GGUF was physically sliced and loaded. That single artifact does not establish broad Qwen3.5/Qwen3.6 compatibility. |
| Integration approach | Independent adapter implementation informed by upstream format and runtime source. Existing transactional writer, direct-copy tensor path, and disk preflight are reused. No upstream code is incorporated. |
| Expected benefit | A 32-of-256 expert reduction should physically reduce routed-expert bytes by approximately 12% in an expert-dominated file. Success as an optimizer requires held-out perplexity/task quality within configured limits and measured deployment benefit against equal-budget baselines. |
| Risk | Static router weight norm is not a reliable proxy for expert usage or quality contribution. Uniform expert-count metadata is required by the current llama.cpp runtime contract. Existing routing weights can still send tokens to removed experts; current implementation preserves the router columns selected by the caller, but does not rewrite router behavior or prove runtime quality. |
| Validation | Synthetic exact-byte Q4_K/Q6_K slicing and failure tests; real-model layout validation, physical output, untouched-tensor hashing, and llama.cpp reload/generation. Add held-out perplexity/task evaluation, throughput/RAM/VRAM measurement, multiple selections, and runtime-version coverage before optimizer exposure. |
| Priority | High-value research target, but only a low-level slice is currently safe to ship. Keep autonomous selection and promotion disabled until quality-gated physical evaluations are available. |

## Physical smoke evidence

On a local 21,713,462,496-byte Qwen3.6-derived `qwen35moe` Q4_K_M GGUF,
32 experts were removed from each of 41 blocks (256 to 224; top-k remained 8).
The separate output was 19,199,501,568 bytes: a reduction of 2,513,960,928
bytes (11.58%). The writer compared all 589 untouched tensor hashes, and they
matched the source. The operation used a reported maximum payload buffer of
16 MiB. The source was opened for read-only mapping and the candidate was
published to a separate destination.

Both source and output loaded with local llama.cpp
`0.7.0-dev`, commit `c7fc50941`, on CPU and generated a short continuation
from the same prompt. This establishes reload compatibility for that exact
artifact/runtime pair. At that point, no suitable held-out corpus was cached
locally, so no quality or latency claim was made.

## Exploratory quality screen

On 2026-10-25, the source and three physical candidates were scored on the first
four 512-token chunks of the WikiText-2 raw test split using paired settings:
llama.cpp `c7fc50941`, CPU-only, 12 threads, context/batch 512, microbatch 256,
no warmup. The corpus came from the llama.cpp CI mirror's
`wikitext-2-raw-v1.zip`; its SHA-256 was
`ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11`, matching
the published WikiText-2 archive checksum. The extracted test text SHA-256 was
`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. The
dataset card identifies WikiText as CC BY-SA/GFDL; no corpus text is committed
or redistributed here. See the
[Salesforce WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
and [llama.cpp's test-corpus mirror](https://huggingface.co/datasets/ggml-org/ci).

| Artifact | Experts retained (of 256, each block) | GGUF bytes | PPL ± reported uncertainty | Delta vs source | PPL retention (`source / candidate`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 256 | 21,713,462,496 | 4.9739 ± 0.36559 | — | 100% |
| Router norm, 32 removed | 224 | 19,199,501,568 | 7.4661 ± 0.58849 | +2.4922 | 66.6% |
| Seeded random (`20261025`), 32 removed | 224 | 19,199,501,568 | 6.4236 ± 0.49709 | +1.4497 | 77.4% |
| Router norm, 8 removed | 248 | 21,084,972,288 | 5.3220 ± 0.39128 | +0.3481 | 93.5% |
| Seeded random (`20261025`), 4 removed | 252 | 21,399,217,408 | 5.1149 ± 0.37713 | +0.1410 | 97.2% |
| Router norm, 4 removed | 252 | 21,399,217,408 | 5.0868 ± 0.37537 | +0.1129 | 97.8% |

This short screen puts all five settings below ModelSurgeon's 98% minimum
point-estimate PPL-retention requirement. At 224 experts, router-norm ranking
scored worse than the one seeded-random selection at equal compression. At 252
experts, router-norm's 0.0281 lower PPL point estimate than random is much
smaller than either reported uncertainty; it does not demonstrate a ranking
advantage. The 252-expert outputs reduce disk size by 314,245,088 bytes
(1.45%), so this potential benefit is modest. The reported uncertainty
intervals for both 252-expert candidates overlap the source; this four-chunk
screen cannot confirm or rule out a 98% retention result. Treat these as
diagnostic measurements, not final acceptance decisions. WikiText is a
general-language corpus, not a cyber-security task benchmark. Do not expose
these rankings for autonomous selection based on this evidence. No speedup is
claimed.

The repository's `task-quality-code-exact-match-v1.jsonl` is also not an
adequate substitute for a cyber-security benchmark: inspection shows only four
generic Python function prompts, and existing acceptance evidence documents
that this tiny set establishes neither broad coding ability nor security
behavior.

As a limited follow-up, on 2026-10-25 the source and the 252-expert
router-norm candidate each received the same four manually screened benign
defensive prompts (MITRE FRR dataset indices 0, 1, 7, and 10). Both models had
0/4 responses classified as refusals by the pinned upstream
`is_response_llm_refusal` scorer (0% on this subset). Inference used the same
llama.cpp `c7fc50941` CPU server, 12 threads, context 512, temperature 0, seed
`20261025`, and 64-token output cap. The scorer applies keyword patterns to
the first 220 response characters and treats code blocks as non-refusals, so
this result only checks gross false-refusal behavior on four prompts; it does
not assess code correctness, defensive utility, security, or quality retention.
The complete paired responses and settings are retained outside the
repository at `D:\ModelSurgeonEval\cyberseceval\mitre-frr-four-prompt-paired-smoke.json`.

Dataset and scorer provenance: [Meta PurpleLlama CyberSecEval 2 at pinned
revision `d934ea285d124d81c10f7e15a06a1e1f74eff2c3`](https://github.com/meta-llama/PurpleLlama/tree/d934ea285d124d81c10f7e15a06a1e1f74eff2c3/CybersecurityBenchmarks);
the MITRE FRR dataset is covered by that directory's MIT license. Only its
selected prompt indices and aggregated refusal result are recorded here; no
dataset or upstream scorer code is copied into ModelSurgeon. This four-prompt
smoke is not a benchmark score. A meaningful task-quality study still needs a
larger, pinned, license-compatible cybersecurity evaluation with benign
utility and safe-code measures, scored for output correctness as well as
refusals rather than treating this refusal subset as validation.

The source model SHA-256 is
`ed118e7768330eb440ddc8ade42a04005ecab37f131d7e84bb08d14d1e315ae1`. Candidate
SHA-256 values, in table order after the source, are:

- Router norm, 224 experts: `3c76db1ba5dd1f7ad3dc12fc41590a6da9519a27d79a74597adaf0e0635ec869`
- Seeded random, 224 experts: `e66234dbda6333dbf485aba2bb185f30b971bdd2700198014c99506a5b72ae62`
- Router norm, 248 experts: `6b2a0bd324c9d6ccf067d6982a020101df07fc3e070e48bde1afb742479ad898`
- Seeded random, 252 experts: `17bbf662f37f0b454583c4847bb1adce3af1b1f45478697ec48b609de7a1042d`
- Router norm, 252 experts: `06969df1ce0505ad2f7d506497c9f5f9731c3a3887d07351ccafbd3edd7d3fe3`

## Current integration boundary

The exported `build_qwen35moe_gguf_layout`,
`rank_qwen35moe_experts_by_router_norm`, and
`execute_qwen35moe_expert_pruning` functions are adapter-level primitives.
They are not accepted by `build_optimize_plan`, do not appear in the central
optimizer's supported scopes, and cannot publish an accepted optimization
outcome. This avoids implying that a router-norm-ranked artifact has passed
ModelSurgeon's quality gate. The next integration step is a bounded MoE
candidate-search scope that runs held-out quality and deployment evaluation
for every candidate before selection or publication.

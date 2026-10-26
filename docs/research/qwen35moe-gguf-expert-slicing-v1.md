# Qwen3.5 MoE GGUF routed-expert slicing: implementation note

Status: low-level experimental adapter; not exposed as an autonomous optimizer
scope. Quality retention and deployment improvement remain unproven.

## Upstream study and acquisition decision

| Source and revision | Component | License | ModelSurgeon equivalent and limits | Acquisition decision |
| --- | --- | --- | --- | --- |
| [llama.cpp `qwen35moe.cpp`, `74bca3950054eed1795f525cb9cf06bfe796492d`](https://github.com/ggml-org/llama.cpp/blob/74bca3950054eed1795f525cb9cf06bfe796492d/src/models/qwen35moe.cpp); [tensor mappings at the same revision](https://github.com/ggml-org/llama.cpp/blob/74bca3950054eed1795f525cb9cf06bfe796492d/gguf-py/gguf/tensor_mapping.py) | Qwen3.5 MoE GGUF tensor names, routed-expert axes, metadata, and runtime interpretation | MIT | Existing native GGUF writer and opaque-tensor copy path existed; no Qwen3.5 routed-expert surgery contract existed. The general dense Qwen path is not valid for this hybrid MoE architecture. | **REIMPLEMENT** the narrow layout checks and slicing using ModelSurgeon's transactional GGUF writer. Upstream code was inspected, not copied. |
| [Crucible REAP, `dacdb43eef0630de6dafe5c54bdbd2961ea6295`](https://github.com/peonist-ai/crucible/blob/dacdb43eef0630de6dafe5c54bdbd2961ea6295/packages/crucible/crucible/methods/reap.py) | Router-weight and activation-informed expert importance | Apache-2.0 | No MoE expert ranker existed. Current router-column L2 is a static, activation-free baseline and does not reproduce REAP. | **REIMPLEMENT** only the explicitly labeled static baseline for a first physical format test. Study REAP-style router/activation scores before candidate ranking is wired into optimization. No source was copied. |
| [Cerebras REAP, `1970473c51ca3caeb98c10392f15b3a08a672974`](https://github.com/CerebrasResearch/reap/tree/1970473c51ca3caeb98c10392f15b3a08a672974) | Calibration-based router-weighted expert-output saliency | Apache-2.0 | Router-column L2 omits expert outputs, observed routing frequency, and calibration data. The pinned implementation accumulates the per-expert mean of output L2 norm multiplied by the selected router probability; it can renormalize top-k probabilities. | **ADAPT** the scoring rule behind an architecture-specific calibration adapter if equal-budget held-out evaluation demonstrates an advantage. No source was copied. |

The official REAP implementation is more appropriate than the earlier Crucible
reference for this comparison. Its observer obtains routed expert outputs,
selected expert IDs, and router logits, then updates the per-layer saliency
state. This requires actual model forwards and is more expensive than reading
router weights from GGUF. REAP's Qwen3.5 architecture support is currently in
[open PR #25](https://github.com/CerebrasResearch/reap/pull/25); its
disk-streamed layerwise calibration work is in [open PR #26](https://github.com/CerebrasResearch/reap/pull/26),
stacked on #25. These are unmerged changes, so their Qwen3.5 support and
resource claims are not treated as released, independently verified capability.
The upstream PR file lists contain no added tests for these changes. A source
trace of PR #26's pinned
[`layerwise_observer.py`](https://github.com/CerebrasResearch/reap/blob/f0caca21435b2b8c85b48508394c8e4e38885be5/src/reap/layerwise_observer.py)
also found a material resource issue in its Qwen3.5 fused-expert path: it builds
a dense `[experts, tokens,
hidden]` input, evaluates the fused expert weights for every expert/token pair,
and stores an `[experts, tokens, hidden]` output buffer even though REAP updates
scores only for selected experts. At 256 experts and top-8 routing, this is 32x
the routed expert evaluation count. With hidden size 2,048 and 2,048 calibration
tokens, the output buffer alone is 4 GiB because `torch.zeros` uses the default
float32 dtype; the fused gate/up, intermediate, and down-projection tensors add
further peak memory. These are allocation and operation-count estimates from
the source, not measured peak-memory or runtime results. They make the PR's
author-reported RSS claims insufficient evidence for GPU feasibility. Before
adapting this observer, require routed-token-only expert evaluation, bounded
temporary allocations, numerical agreement with a dense reference on toy
weights, architecture tests for Qwen3.5's fused expert/router signatures, and
measured CPU/RAM and GPU/VRAM profiles at the target calibration lengths. The
model weights used for the current GGUF study are from
[`endystrike/Endy-Qwen3.6-CyberSec-35B-A3B` at `eed558458ca9ff6972063c486c7f1ab9e0b0c93e`](https://huggingface.co/endystrike/Endy-Qwen3.6-CyberSec-35B-A3B/tree/eed558458ca9ff6972063c486c7f1ab9e0b0c93e),
whose card declares AGPL-3.0. The 71.9 GB BF16 checkpoint is for local
calibration only; model weights and per-expert score vectors are not included
in this repository. Evaluation results must identify the model revision and
license and must not be generalized to other Qwen3.5/3.6 models.

For a sparse implementation reference, Hugging Face Transformers' pinned
[`Qwen3_5MoeExperts`](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L752-L785)
dispatches only token indices selected for each expert, computes the fused
gate/up and down projections on those tokens, then weights and accumulates the
outputs. That implementation is Apache-2.0
([pinned license](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/LICENSE)); no source is copied.
Its public return is the weighted sum, so REAP still needs an architecture-aware
observation point before router weighting, or a separately computed sparse
expert pass. The latter is a plausible way to avoid the dense PR #26 path, but
its calibration time and peak memory remain to be measured.

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

## Paired eight-chunk follow-up

On 2026-10-25, the same source, router-norm selection, and seeded-random
selection retaining 252 experts per block were re-evaluated on the same
llama.cpp build and corpus, now using the first eight consecutive 512-token
chunks (4,096 tokens total). The tokenizer produced the same eight token
offsets (`0` through `3584`) for all three artifacts. Settings were CPU-only,
12 threads, context/batch 512, microbatch 256, and no warmup.

| Artifact | Experts retained (of 256, each block) | GGUF bytes | PPL ± reported uncertainty | Delta vs paired source | PPL retention (`source / candidate`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 256 | 21,713,462,496 | 6.2085 ± 0.33009 | — | 100% |
| Router norm, 4 removed | 252 | 21,399,217,408 | 6.2792 ± 0.33493 | +0.0707 | 98.87% |
| Seeded random (`20261025`), 4 removed | 252 | 21,399,217,408 | 6.4208 ± 0.34295 | +0.2123 | 96.69% |

The router-norm point estimate clears the 98% retention threshold on this
larger prefix, and its PPL is 0.1416 below the equal-size random candidate.
That gap is smaller than either candidate's reported uncertainty; the
uncertainties overlap, so this is not evidence of a reliable ranking
advantage. The source estimate also shifts from 4.9739 on the first four
chunks to 6.2085 on the first eight, demonstrating material variation as
chunks are added. This paired screen is still only one checkpoint, one random
seed, one contiguous 4,096-token prefix, and a general-language corpus. It
does not validate cyber-security task retention or throughput, and it does
not justify autonomous ranking or promotion.

## Paired 64-window follow-up

On 2026-10-25, the same three artifacts were evaluated on the first 64
consecutive 512-token WikiText-2 test contexts (32,768 tokens each), using the
same llama.cpp `c7fc50941` CPU settings. The full held-out test file was not
processed. The expanded prefix shifts the source PPL relative to the earlier
eight-window run, which is a reminder that short prefixes are sample-dependent.

| Artifact | Experts retained (of 256, each block) | GGUF bytes | PPL ± reported uncertainty | Delta vs paired source | PPL retention (`source / candidate`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 256 | 21,713,462,496 | 6.4083 ± 0.11895 | — | 100% |
| Router norm, 4 removed | 252 | 21,399,217,408 | 6.4863 ± 0.12059 | +0.0780 | 98.80% |
| Seeded random (`20261025`), 4 removed | 252 | 21,399,217,408 | 6.5198 ± 0.12151 | +0.1115 | 98.29% |

Both candidates meet the 98% point-estimate retention threshold on this prefix,
but the llama.cpp `+/-` values are propagated uncertainty, not formal
confidence intervals. The source and candidate bands overlap, as do the two
candidate bands. Router norm's mean PPL is 0.0335 lower than random at equal
bytes; this run does not establish a reliable ranking advantage or a formal
quality-gate pass. No inference-speed or cybersecurity-task result was
measured. The exact hashes, settings, return codes, and log digests are in the
[64-window evidence manifest](qwen35moe-gguf-expert-slicing-64chunk-2026-10-25.json).

The four- and 64-window screens alone do not justify exposing the scorer to
autonomous selection. The full-split comparison below adds one random control;
independent random selections and representative cybersecurity task-quality
measurements remain necessary.

## Full-test baseline and resource-limited comparison

On 2026-10-25, the pinned CPU runtime processed the complete WikiText-2 test
file for the source artifact: 580 contexts (296,960 tokens), PPL 6.7090 ±
0.04243, exit code 0. This source-only result is a corpus baseline, not evidence
about expert pruning. Its PPL differs from the 64-context prefix estimate,
confirming that the shorter prefix should not be used as a stand-in for the
complete test split.

The equal-size router-norm candidate used the same full-split command, but
available system memory fell to 852 MiB while the inference process held about
14.31 GiB. The run was interrupted after 107 output rows (last recorded offset
54,272); it produced no final PPL estimate. At that time, the random candidate
had not been run.
These incomplete outputs are retained as a resource-pressure record only. They
are not quality results, and no quality gate or ranking conclusion follows.
The [full-test run manifest](qwen35moe-gguf-expert-slicing-fulltest-2026-10-25.json)
records the completed source result, interrupted router run, exact inputs and
runtime, and log digests.

For the earlier eight-window run, the detailed manifest and per-run logs
remain local rather than being redistributed. That manifest's SHA-256 is
`1b2ac496b5a116ea0796afba0528bdd137cf72face56d57d7a270d27ca387875`.

The repository's `task-quality-code-exact-match-v1.jsonl` is also not an
adequate substitute for a cyber-security benchmark: inspection shows only four
generic Python function prompts, and existing acceptance evidence documents
that this tiny set establishes neither broad coding ability nor security
behavior.

## Pinned three-way full-test comparison (2026-10-26)

The earlier source-only full-test baseline and resource-limited candidate
attempt are retained in the [2026-10-25 full-test manifest](qwen35moe-gguf-expert-slicing-fulltest-2026-10-25.json).
The router-norm 252-expert candidate was re-evaluated on 2026-10-26 using the
same complete WikiText-2 test split, source artifact, llama.cpp build, and CPU
settings. The local ZIP archive SHA-256 matched the pinned archive, and a fresh
extraction matched the recorded test-text SHA-256 before inference. The
router-norm candidate SHA-256 matched the previously recorded artifact. The
equal-size seeded-random candidate was evaluated with the same command and
runtime, and its SHA-256 matched the pinned artifact.

| Artifact | GGUF bytes | Full-test contexts / tokens | PPL ± propagated uncertainty | Delta vs source | Point-estimate PPL retention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source | 21,713,462,496 | 580 / 296,960 | 6.7090 ± 0.04243 | — | 100% |
| Router norm, 252 experts | 21,399,217,408 | 580 / 296,960 | 6.8094 ± 0.04317 | +0.1004 | 98.53% |
| Seeded random (`20261025`), 252 experts | 21,399,217,408 | 580 / 296,960 | 6.8291 ± 0.04336 | +0.1201 | 98.24% |

Both candidates are 314,245,088 bytes (1.45%) smaller, and both point estimates
clear the configured 98% PPL-retention floor. Router norm's PPL point estimate
is 0.0197 lower than the one seeded-random selection, but the reported
uncertainty bands overlap; llama.cpp's propagated uncertainty is not a formal
confidence interval. This does not establish a reliable ranking advantage,
pass ModelSurgeon's first-party quality gate, or demonstrate task-quality
retention. It is one model, one corpus, and one random seed. Additional random
seeds and representative cybersecurity task-quality measurements remain
necessary. No inference speed, latency, load-time, or deployment-memory
benefit was measured.

The exact artifact and dataset hashes, command, settings, runtime, metrics,
raw-log digests, and limitations are in the
[full-test three-way comparison manifest](qwen35moe-gguf-expert-slicing-fulltest-router-norm-2026-10-26.json).

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
The raw paired-response JSON is retained locally and is not included in this
PR; its SHA-256 is
`46177ed8d772435dac9faceb2ba053156517c2cb044e6ca989b9389ebc60b0d3`.

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

## Calibration-score selection bridge

`select_qwen35moe_experts_by_scores` accepts layer-major finite saliency values
and produces the per-layer retained-ID mapping consumed by the GGUF slicer.
Its input shape is compatible with the `scores_by_layer` field proposed by
[open PR #738](https://github.com/xdCloudy/ModelSurgeon/pull/738), but this
selector has no runtime dependency on that PR or on PyTorch/Hugging Face. It
keeps the highest scores independently within each layer, uses expert ID as a
deterministic tie-break, and retains `None` entries because they were not
observed. If unseen entries exceed the retained count, it fails closed.

This is candidate construction only. The GGUF contract still requires the
same expert count in each layer; the retained IDs may differ by layer. The
synthetic test proves that the score mapping reaches the transactional writer
and produces valid router tensor dimensions. It does not establish that REAP
scores improve held-out quality, that the experimental scorer has merged, or
that a candidate can pass ModelSurgeon's quality gate. No Cerebras REAP or
Transformers implementation was copied; this adapter consumes numeric results
only. REAP's pinned Apache-2.0 provenance remains recorded above.

An additional upstream review found CerebrasResearch/REAP's still-open
[Qwen3.5-MoE support PR #25](https://github.com/CerebrasResearch/REAP/pull/25),
head `ed8f8b776394d863bd0ff800e3498825d1d6d62a` (2026-07-19). Its proposed
changes register the nested `Qwen3_5MoeForConditionalGeneration` architecture,
handle the raw-parameter router and fused expert tensors, and extract
`select_retained_experts` from the in-memory pruning path. The PR inherits the
upstream repository's Apache-2.0 `LICENSE` (blob
`f49a4e16e68b128803cc2dcea614603632b04eac`); no files from that PR are copied
here. Because it is not merged into REAP's pinned main revision, ModelSurgeon's
selector remains an independent numeric adapter and does not rely on the PR's
model registry or implementation. This review reinforces the architecture
boundary but adds no model-quality evidence.

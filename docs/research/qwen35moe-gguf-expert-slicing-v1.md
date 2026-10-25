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
artifact/runtime pair. It is not a quality benchmark: there was no suitable
held-out corpus available locally, and the short generations cannot establish
quality retention. No latency or throughput improvement is claimed.

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

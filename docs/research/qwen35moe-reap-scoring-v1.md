# Qwen3.5 REAP-style expert scoring adapter

Status: experimental calibration adapter. It is not connected to candidate
selection, structural surgery, or autonomous optimization. A single full
WikiText-2 held-out comparison on one Qwen3.5-family GGUF found better PPL than
router-norm and two random 12.5% expert-pruning baselines, but the candidate
failed the repository's 98% quality-retention gate. This is promising
one-cell evidence, not a promoted method or a latency/resource result. See the
[full experiment record](qwen35moe-reap-224-wikitext-fulltest-2026-10-27.md).

## Acquisition decision

| Field | Assessment |
| --- | --- |
| Source | CerebrasResearch/REAP at `1970473c51ca3caeb98c10392f15b3a08a672974`; Hugging Face Transformers at `50e35cc0ca18ec19b80653690de553e6decf74f5`. |
| Component | Per-layer expert saliency: mean selected router probability multiplied by the L2 norm of that expert's output for routed calibration tokens. |
| License | Both pinned source trees declare Apache-2.0. The adapter below was independently implemented; no upstream source was copied. |
| Current equivalent | ModelSurgeon's Qwen MoE GGUF adapter ranks experts by static router-column norm. It has no activation-based expert score and remains outside autonomous selection. |
| Advantage hypothesis | Calibration-aware scores may rank removable experts more safely than router norm or random selection at equal retained-expert counts. The advantage is unproven. |
| Hardware | Requires PyTorch and a loaded fused Qwen3.5 expert layer. Working activations scale with routed tokens and expert hidden/intermediate sizes; expert modules must already fit on the chosen device. No model loading policy or resource cap is provided here. |
| Architecture | Explicit tensor contract for Hugging Face `Qwen3_5MoeExperts`: `[tokens, hidden]` states, `[tokens, top_k]` expert indices and weights, stacked fused `gate_up_proj` and `down_proj`. Other implementations and layouts are unsupported. |
| Acquisition | **ADAPT** the published score independently; use sparse routed-token slices rather than evaluating every expert on every token. The Transformers dispatch path is a source reference, not imported code. |
| Risk | Incorrect masks, architecture drift, calibration distribution shift, expert output compute overhead, and noisy rankings. Hook callers must provide an explicit valid-token mask policy. |
| Validation | Dense-reference numerical parity; route, padding, accumulation, and hook tests. Before optimizer integration: pinned real checkpoints, held-out task and perplexity evaluation against random and router-norm baselines, multiple seeds, plus peak RAM/VRAM and calibration-time measurements. |
| Priority | P0 research candidate; do not promote until paired held-out evaluation shows a repeatable improvement over simple baselines without violating quality and hardware constraints. |

## Upstream implementation review

### Current upstream status

On 2026-10-26, `CerebrasResearch/reap` `main` still resolved to
`1970473c51ca3caeb98c10392f15b3a08a672974`, the revision reviewed below. The
upstream repository has an open Qwen3.5-MoE support proposal, PR
[#25](https://github.com/CerebrasResearch/reap/pull/25), at head
`ed8f8b776394d863bd0ff800e3498825d1d6d62a`. Its proposed changes are limited
to `model_util.py`, `observer.py`, and `prune.py`; they are not part of upstream
`main`. This adapter remains an independent implementation of the observer
contract at the Hugging Face fused-expert boundary; it copies no code from
that unmerged proposal. The upstream `LICENSE` declares Apache-2.0, compatible
with ModelSurgeon's Apache-2.0 distribution; the adapter introduces no copied
source or new dependency.

At CerebrasResearch/REAP revision
[`1970473c51ca3caeb98c10392f15b3a08a672974`](https://github.com/CerebrasResearch/reap/tree/1970473c51ca3caeb98c10392f15b3a08a672974),
the score update entry point is
[`src/reap/pruning_metrics.py:update_pruning_state`](https://github.com/CerebrasResearch/reap/blob/1970473c51ca3caeb98c10392f15b3a08a672974/src/reap/pruning_metrics.py).
It softmaxes full router logits, gathers selected probabilities, optionally
renormalizes them over each token's selected experts, then accumulates the
mean router-weighted activation norm for each routed expert. The standard
hook path is
[`src/reap/observer.py:MoETransformerObserver._hook_factory`](https://github.com/CerebrasResearch/reap/blob/1970473c51ca3caeb98c10392f15b3a08a672974/src/reap/observer.py);
the layerwise alternative is
[`src/reap/layerwise_observer.py:LayerwiseMoEObserver`](https://github.com/CerebrasResearch/reap/blob/1970473c51ca3caeb98c10392f15b3a08a672974/src/reap/layerwise_observer.py).
The latter captures the first block's inputs and replays blocks one at a time,
offloading each block after use. Its parity checks against the ordinary
observer are in
[`tests/test_layerwise_observer.py`](https://github.com/CerebrasResearch/reap/blob/1970473c51ca3caeb98c10392f15b3a08a672974/tests/test_layerwise_observer.py).

The pinned
[`Qwen3_5MoeTopKRouter.forward`](https://github.com/huggingface/transformers/blob/50e35cc0ca18ec19b80653690de553e6decf74f5/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
already applies full softmax, top-k selection, and selected-weight
renormalization before calling `Qwen3_5MoeExperts`. ModelSurgeon observes the
expert boundary and therefore consumes these final weights directly; applying
the generic observer's optional renormalization again is unnecessary. The
new router-contract test verifies this path against a direct expert-output
reference.

This is a Qwen-weighted REAP-style variant, not a numerical reproduction of
REAP's default score. At the pinned REAP revision,
`MoETransformerObserverConfig.renormalize_router_weights` defaults to `False`,
so its default score uses full-softmax probabilities for selected experts;
Qwen3.5's expert boundary instead supplies the selected probabilities
renormalized to sum to one for each token. Every serialized score record now
includes `router_weight_semantics: qwen35_top_k_weights_renormalized_per_token`
so later comparisons can distinguish the observed weighting policy. A fair
comparison should either reproduce the matching REAP renormalized mode or
report the full-softmax and Qwen-renormalized variants separately.

REAP's separate
[`src/reap/prune.py:prune`](https://github.com/CerebrasResearch/reap/blob/1970473c51ca3caeb98c10392f15b3a08a672974/src/reap/prune.py)
uses per-layer low-score selection and mutates architecture-specific expert
and router tensors. ModelSurgeon adapts the score only; it does not reuse that
mutation path, which is coupled to REAP's model registry and does not provide
ModelSurgeon's transactional planning and artifact-publication contract. The
upstream repository declares Apache-2.0, compatible with this project, but no
REAP code or dependency is copied. Its end-to-end environment also includes
evaluation and serving dependencies that are not required for the isolated
score calculation.

## Implementation boundary

`Qwen35MoeReapObserver` accepts the tensors at the fused-expert boundary or
installs forward hooks on each layer's expert module. It recomputes only the
selected expert/token pairs to obtain per-expert outputs, accumulates weighted
L2 means and route counts, and reports `None` for experts not observed. It
does not change router decisions, prune or merge experts, normalize top-k
weights again, choose an artifact, or publish model output. The pinned
Transformers Qwen3.5 router already softmaxes all expert logits, selects the
top-k probabilities, and renormalizes those selected probabilities before
passing them as `top_k_weights` to the fused expert module. The observer uses
those final values verbatim; it does not accept router logits or apply a
second normalization. Its score uses the unweighted expert output before
applying the selected router weight, matching REAP's observation semantics.

The implementation processes at most 1,024 routed tokens per update chunk by
default (`max_tokens_per_update`) and at most 1,024 routes per expert projection
chunk (`max_routes_per_expert_batch`). Both are positive-integer overrides for
smaller hardware budgets. Route-index bookkeeping and projection intermediates
are therefore bounded by their respective chunk limits rather than the full
calibration batch. Per-expert route counts are transferred once per token
chunk. An update accumulates scores locally and commits only after all chunks
are valid, so invalid data or non-finite output in a later chunk does not leave
partial observer state. The implementation avoids discovering route positions
with a separate data-dependent `where` operation for every expert and avoids
materializing a tensor of every selected expert weight. The structural
reduction in route-discovery synchronizations is not a measured CUDA latency
result; current host VRAM availability prevented an RTX 3060 timing run.

Padding is deliberately explicit: hook registration requires a mask provider.
Returning `None` means every routed token in that call is valid. The caller is
responsible for matching flattened routed-token order to its attention mask.
The adapter does not silently infer this mapping.

An isolated deterministic CPU check (PyTorch 2.11.0+cpu, seed `20261026`,
19-token toy inputs, hidden size 7, intermediate size 5, six experts, top-k 3)
compared a dense reference with routed-only scoring in float64. Maximum
absolute score error was `1.7763568394002505e-15`; routed expert evaluations
were 57 versus 114 dense expert/token pairs. This establishes only toy
mathematical agreement and fewer evaluated pairs for that fixture. It is not
a runtime, peak-memory, full-model, or quality benchmark.

A separate optional CPU integration test constructs Transformers 5.10.1's
actual `Qwen3_5MoeExperts` module with small random weights, runs its eager
forward path, observes its positional hook inputs, and checks score parity,
padding exclusion, route counts, and hook cleanup. This verifies that exact
library implementation's tensor and hook contract; it does not validate
pretrained weights, other Transformers versions, or model-level quality and
performance.

A second tiny contract test runs Transformers' real `Qwen3_5MoeTopKRouter`
and passes its returned indices and already-renormalized weights to the
observer. It checks the selected weights sum to one and that scores match a
direct expert-output calculation. This prevents confusing expert-boundary
weights with raw router logits during future integration.

The focused adapter suite passes on Python 3.12.14 and PyTorch 2.14.1+cu130
through its CPU path: 7 passed, no skips. This confirms the installed
Transformers Qwen3.5 modules load in the updated environment; it is not a CUDA
execution or model-quality result.

As a local checkpoint-weight smoke test, the adapter also scored expert 23 in
layer 0 using its real BF16 gate/up and down tensors from
[`Endy-Qwen3.6-CyberSec-35B-A3B` revision `eed558458ca9ff6972063c486c7f1ab9e0b0c93e`](https://huggingface.co/endystrike/Endy-Qwen3.6-CyberSec-35B-A3B/tree/eed558458ca9ff6972063c486c7f1ab9e0b0c93e).
That model card declares AGPL-3.0; no weights or tensor contents are
redistributed. The two locally read shard files matched their pinned LFS
sizes and SHA-256 values. Four valid, seeded synthetic BF16 hidden vectors
were routed to the expert (seed `20261026`, six rows total, two masked; CPU
PyTorch 2.13.0+cpu and Transformers 5.10.1); the observer's mean
router-weighted output norm was `0.37381422`, versus `0.37381423` from a
direct calculation (absolute error `7.45e-9`). This confirms BF16 operation
against real expert weights with synthetic inputs only. It provides no
calibration ranking, quality, deployment, or performance evidence.

After installing Python 3.12.14 and PyTorch 2.14.1+cu130, a second CPU-only
smoke loaded expert 23's real layer-0 BF16 tensors directly from the same two
pinned shards (gate/up shape `[1, 1024, 2048]`, down shape `[1, 2048, 512]`).
With four valid synthetic hidden vectors, the observer returned `0.710203558`
versus `0.710203528` from direct evaluation (absolute error `2.98e-8`). This
checks the updated environment against real tensor dtypes and dimensions; the
CUDA build was not exercised on the busy GPU, and this remains synthetic-input
parity rather than a calibration or quality result.

The pinned Hugging Face checkpoint lists 26 safetensors totaling
`71,903,776,776` bytes. This exceeds the target workstation's 64 GiB of physical
RAM, so a fully resident Transformers load is not a valid consumer-hardware
plan. The checkpoint was downloaded for shard and real-weight inspection, but
no full-model forward or calibration run was attempted. A model-level result
requires a measured layerwise/offload path or a smaller supported checkpoint;
the synthetic-input parity above must not be presented as model validation.

## Next evidence required

The first full held-out PPL comparison is recorded in
[`qwen35moe-reap-224-wikitext-fulltest-2026-10-27.md`](qwen35moe-reap-224-wikitext-fulltest-2026-10-27.md)
and machine-readable form in
[`records/qwen35moe-reap-224-wikitext-fulltest-2026-10-27.json`](records/qwen35moe-reap-224-wikitext-fulltest-2026-10-27.json).
It shows a substantial one-cell PPL improvement over router norm and two
random seeds, while missing the 98% gate. Replicate with independently sampled
calibration subsets and random seeds; include coding/reasoning/task quality,
artifact reload and runtime checks, decode/prefill speed, peak RAM/VRAM, and
calibration cost. Validate the first-party adapter's score ordering against
this external POE result before treating the result as evidence for PR #738's
implementation. Keep failures and unsupported cells. The adapter remains a
research instrument and cannot authorize pruning.

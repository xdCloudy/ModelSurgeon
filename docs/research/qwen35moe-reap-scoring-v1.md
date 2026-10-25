# Qwen3.5 REAP-style expert scoring adapter

Status: experimental calibration adapter. It is not connected to candidate
selection, structural surgery, or autonomous optimization. No model-quality,
latency, or resource benefit has been established.

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

## Implementation boundary

`Qwen35MoeReapObserver` accepts the tensors at the fused-expert boundary or
installs forward hooks on each layer's expert module. It recomputes only the
selected expert/token pairs to obtain per-expert outputs, accumulates weighted
L2 means and route counts, and reports `None` for experts not observed. It
does not change router decisions, prune or merge experts, normalize top-k
weights, choose an artifact, or publish model output. It computes the score
from the unweighted expert output before applying the selected router weight,
matching the pinned REAP metric's observation semantics.

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

## Next evidence required

Use the same checkpoint, calibration and held-out splits, retained expert
counts, and evaluation budget for REAP-style scores, router norm, and several
random seeds. Report artifact reload validity, disk/parameter reductions,
perplexity and representative task quality, calibration wall time, and peak
CPU/GPU memory. Keep all failures and unsupported architecture cells. Until
that study demonstrates repeatable value, this adapter remains a research
instrument and cannot authorize pruning.

# Competitive technology acquisition matrix (initial audit)

**Audit basis:** ModelSurgeon `main` at `72cf8017d48669157dd6b51b7c2ad0a1917320a1`;
upstream revisions and repository licence metadata were resolved on 2026-10-25.
This is the first implementation-focused tranche of the continuing review, not a
claim that every listed upstream has completed a line-by-line audit. No upstream
code is copied by this change.

## Current ModelSurgeon boundary

The source audit found a real HF path from calibration and per-channel feature
collection through candidate measurement, structural surgery, reload, quality
gates, deployment measurement, and evidence publication. A signed Meta-Surgeon
can order candidates, but current search physically measures every candidate
before selecting the measured feasible result. Existing comparison records
include Meta-Surgeon, weight-magnitude, random, and no-guidance baselines. The
retained transfer evidence is negative or neutral, so predictions do not have
measurement authority and are not yet an optimization advantage. This is the
critical integration gap for this pass.

GGUF remains a separate typed path with explicit family/layout/codec contracts.
The current evidence proves physically written and reloaded candidates with
negative quality-gate outcomes; it does not establish broad MoE or quantized
model coverage. ModelSurgeon's own [goal-to-reality matrix](goal-reality-gap-matrix.md)
and [first-surgeon proof protocol](../first-surgeon-proof.md) retain the wider
implementation and evidence limitations.

## Acquisition matrix

| Source, exact revision, licence | Component and current equivalent | Advantages and requirements | Decision and validation |
| --- | --- | --- | --- |
| [Torch-Pruning](https://github.com/VainF/Torch-Pruning/tree/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9), `e80127d7c6935c319a7dd6719b8a72e3c1fcbff9`, MIT; entry points: [`dependency/graph.py`](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/dependency/graph.py), [`pruner/importance.py`](https://github.com/VainF/Torch-Pruning/blob/e80127d7c6935c319a7dd6719b8a72e3c1fcbff9/torch_pruning/pruner/importance.py) | DepGraph groups coupled tensor dimensions by tracing an autograd graph; group-level importance aggregates scores over coupled operations. ModelSurgeon already uses explicit architecture adapters and canonical coupling constraints for supported HF/GGUF layouts. | The graph grouping is a useful cross-check for PyTorch architectures and non-obvious residual/concat dependencies. Requires PyTorch, gradient-enabled example forwards, representative inputs, and module-specific pruning handlers; a traced graph does not prove tokenizer/model-family semantics or safe artifact reload. | **ADAPT (research only).** Do not add it to the runtime dependency path yet. Prototype graph cross-checks on tiny HF fixtures; compare every inferred dependency to the adapter graph and physical reload, then add a new architecture only if it catches a demonstrated coverage gap. MIT permits reuse subject to notice preservation, but this pass reuses no code. |
| [NVIDIA Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer/tree/3c423a9f165b3744e38c5e1e9929b4f15081c413), `3c423a9f165b3744e38c5e1e9929b4f15081c413`, Apache-2.0; scoring entry point [`score_pruning_activations.py`](https://github.com/NVIDIA/Model-Optimizer/blob/3c423a9f165b3744e38c5e1e9929b4f15081c413/modelopt/torch/puzzletron/activation_scoring/score_pruning_activations.py), pruning API [`pruning.py`](https://github.com/NVIDIA/Model-Optimizer/blob/3c423a9f165b3744e38c5e1e9929b4f15081c413/modelopt/torch/prune/pruning.py) | Activation-ranked width/depth pruning and hardware-oriented pruning workflows. ModelSurgeon already stores per-channel activation statistics and physically measures structured HF candidates. | Published code exposes activation importance for neuron/head/layer selection, manual target dimensions and separate auto-pruning/search workflows. The larger end-to-end recipes include calibration, distillation, quantization, and inference-specific runtime setup; reported results are not directly comparable to ModelSurgeon's CPU acceptance cells. | **ADAPT.** Added an independently implemented lowest-activation-magnitude ranking baseline using existing `activation_abs_mean` records. Compare its held-out feasible rate and regret against weight magnitude, random, and Meta-Surgeon on identical physical candidate sets before considering it as a default policy. No NVIDIA code is copied. |
| [WANDA](https://github.com/locuslab/wanda/tree/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6), `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`, MIT | Unstructured weight pruning scored by weight magnitude times input activation norm. ModelSurgeon focuses on structural channel/head/layer edits and GGUF block-aware operations. | A simple calibration-aware pruning baseline with no weight reconstruction or retraining in the published dense-weight path. Requires PyTorch model execution and calibration activations; unstructured sparsity only improves deployable speed if the selected runtime/hardware uses the sparse representation. | **REIMPLEMENT for comparison, not surgery reuse.** Implement a separately named and validated score only if the optimization contract explicitly includes supported sparse artifacts; compare actual size/runtime and held-out quality, not zero counts. MIT permits reuse with notices, but no source copied. |
| [SparseGPT](https://github.com/IST-DASLab/sparsegpt/tree/147d2159dc4f3e9f73e47b32c04d7b3708f44436), `147d2159dc4f3e9f73e47b32c04d7b3708f44436`, Apache-2.0 | Calibration Hessian-based one-shot unstructured / N:M pruning and weight reconstruction. No equivalent in the current structural candidate path. | Adds local reconstruction using second-order statistics and evaluates perplexity on named corpora. The upstream scripts target older OPT/BLOOM/LLaMA stacks; reconstruction and Hessian memory/compute costs require explicit measurement, and sparse deployment is runtime dependent. | **REJECT as current P0 integration.** It changes the compression target from physical structured reduction to sparse-weight encoding and does not establish faster inference on the supported consumer path. Revisit as a separate sparse-artifact benchmark with real runtime support. |
| [LLM Compressor](https://github.com/vllm-project/llm-compressor/tree/e7a7016e70f904ae48825e749c8dd90193b81ffd), `e7a7016e70f904ae48825e749c8dd90193b81ffd`, Apache-2.0 | Quantization and sparsity workflows for vLLM deployment; ModelSurgeon has CPU dynamic-int8 HF quantization and explicit GGUF codec handling. | Mature deployment-oriented method/runtime pairing can ground future quantization and sparse-kernel comparisons. Its vLLM-centered stack and hardware/dependency needs do not directly satisfy the current Windows/llama.cpp consumer workflow. | **INTEGRATE later through a version-pinned adapter** only when a declared target runtime and a physical artifact acceptance cell are added. First measure a same-source/quality/runtime deployment comparison; no dependency added now. |
| [LLM-Pruner](https://github.com/horseee/LLM-Pruner/tree/128a07d977f9b205d60ab14cfbc6a78f8a8e39d2), `128a07d977f9b205d60ab14cfbc6a78f8a8e39d2`, Apache-2.0 | Structured LLM pruning and recovery workflows. ModelSurgeon already implements scoped structural surgery with optional repair as a separate measured stage. | Useful comparative reference for coupled substructure selection and recovery cost; full fine-tuning/data pipelines are outside a low-resource consumer acceptance run. | **ADAPT for an equal-budget competitor protocol**, not a new engine dependency. Pin a small supported checkpoint, dataset, pruning budget and downstream task before implementation claims. |
| [SliceGPT](https://github.com/microsoft/TransformerCompression/tree/183cd92d8469bf3f1e979585a08dc382692bfbed), `183cd92d8469bf3f1e979585a08dc382692bfbed`, MIT | Coordinate transformation plus row/column deletion for dense Transformer width reduction. Current ModelSurgeon low-rank and MLP-channel paths do not implement the same transformation. | Directly targets structured dimensions and can reduce dense checkpoint dimensions; transformations require architecture-specific tensor slicing and quality validation. | **ADAPT as a high-priority isolated study.** First reproduce on a tiny supported model and compare physical bytes, reload, calibration loss and task quality against MLP-channel pruning. MIT-compatible if code is later reused with notices; no source copied. |
| [Minitron repository](https://github.com/NVlabs/Minitron/tree/fed740c91886155e1f272159ef662536bb673593), `fed740c91886155e1f272159ef662536bb673593`, no SPDX licence declared by GitHub | Public model recipes and results; implementation is in the separately licensed NVIDIA Model-Optimizer. | Valuable methodology/results for pruning followed by substantial knowledge distillation, including layer/depth and width choices. The repository's model licence is NVIDIA Open Model License; that is not a code-reuse licence. | **REJECT direct code reuse; ADAPT published technical principles only.** Use the Apache-2.0 Model-Optimizer source when implementation details are needed, and record model/data licences separately. This change copies neither code nor model assets. |
| [MoE Surgeon](https://github.com/netsky-lab/moe-surgeon/tree/63fa2a96aef70f122546ad2d1fed5fd182b346f4), `63fa2a96aef70f122546ad2d1fed5fd182b346f4`, no SPDX licence declared by GitHub | MoE expert selection/optimization; ModelSurgeon has MoE descriptors but explicitly lacks a demonstrated expert surgery acceptance cell. | Direct relevance to expert pruning/merging, but architecture/router semantics and quantized tensor layouts must be proven per family. | **REJECT source incorporation pending licence clarification.** Independently prototype only against published methods and an explicit Llama.cpp/HF MoE test matrix. Do not infer support from component discovery. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/74bca3950054eed1795f525cb9cf06bfe796492d), `74bca3950054eed1795f525cb9cf06bfe796492d`, MIT | GGUF reader/writer, quantization and deployment runtime. ModelSurgeon already has typed GGUF parsing, codecs, selective block rewriting, and pinned external-runtime validation. | Canonical runtime/parser behavior and quantization metadata are essential compatibility oracles. Requires build/runtime-specific validation; llama.cpp support alone does not imply ModelSurgeon can safely transform every tensor. | **INTEGRATE as pinned validation/runtime oracle**, preserving exact revision and runtime geometry in evidence. Keep ModelSurgeon's codecs and fail-closed operation support until per-codec equivalence is demonstrated. |

## Change made from the research

The HF active-search comparison now records a separate `activation_magnitude`
baseline: it orders candidates by the lowest finite pre-mutation
`activation_abs_mean` and reports that candidate's measured quality-gate result
and regret beside the existing baselines. It is an observational baseline only;
it does not change which candidates are evaluated, the selected measured
candidate, the quality gate, or prediction authority. Missing activation
features leave the baseline explicitly unmeasured. This adapts the activation
importance idea from NVIDIA's Minitron workflow to a feature ModelSurgeon
already captures; it is not an implementation of NVIDIA's pruning algorithm and
does not claim better performance.

The acceptance criterion for any future policy use is positive, uncertainty-
bounded reduction in physical evaluations at an equal feasible quality/size
frontier across held-out model identities, with random, weight-magnitude,
activation-magnitude and Meta-Surgeon comparisons on identical candidate sets.
Current evidence does not meet that criterion.

## Next research tranche

The next source audit should trace the actual pruning/calibration code (not only
project documentation) for QLoRA/LoRA recovery, LLM-Surgeon, Crucible, and
llama.cpp's GGUF conversion and quantization paths; record exact file-level
licence coverage; and design pinned, consumer-hardware experiments for dense
structured pruning, quantization, and MoE separately. The backlog already has
open work across many branches; integration must be sequenced after those PRs
land or be rebased to avoid duplicating their implementation.

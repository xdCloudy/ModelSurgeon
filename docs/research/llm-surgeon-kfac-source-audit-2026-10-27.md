# LLM-Surgeon KFAC structured-pruning source audit

Assessment date: 2026-10-27  
Upstream: [Qualcomm-AI-research/llm-surgeon](https://github.com/Qualcomm-AI-research/llm-surgeon) at `726734425b6866782048877bac0c0d90e8cb36b3`.  
Local baseline: ModelSurgeon main `6820c77c6c090bbbd7c821b16a39041f8c39c980`. No upstream code was copied.

## Source trace

The CLI in [`surgeon.py`](https://github.com/Qualcomm-AI-research/llm-surgeon/blob/726734425b6866782048877bac0c0d90e8cb36b3/surgeon.py) loads OPT and LLaMA causal models and assembles calibration/evaluation and optional LoRA-repair options. The pruning path enters [`sparsify.py:sparsify_model`](https://github.com/Qualcomm-AI-research/llm-surgeon/blob/726734425b6866782048877bac0c0d90e8cb36b3/sparsify.py), which obtains supported transformer layers, collects a cost vector for each selected linear module, and derives one global threshold for a requested sparsity. [`curvature.py`](https://github.com/Qualcomm-AI-research/llm-surgeon/blob/726734425b6866782048877bac0c0d90e8cb36b3/curvature.py) provides identity, activation, and Kronecker-factored curvature paths. [`pruner.py`](https://github.com/Qualcomm-AI-research/llm-surgeon/blob/726734425b6866782048877bac0c0d90e8cb36b3/pruner.py) estimates row, column, element, or block loss and can update surviving weights using the estimated curvature. [`threshold.py`](https://github.com/Qualcomm-AI-research/llm-surgeon/blob/726734425b6866782048877bac0c0d90e8cb36b3/threshold.py) solves a global structured threshold while accounting for row/column overlap and layer sizes.

The KFAC path builds Kronecker factors from activation and output-gradient/Fisher statistics. In outline, for a linear weight matrix it uses an approximation of the loss curvature by a product of an output-side factor and an input-activation factor. It estimates the loss of candidate row/column removals, ranks costs across modules, and optionally compensates surviving weights through inverse/eigendecomposition-based updates. The CLI exposes approximations and cost controls such as KFAC rank, Fisher sample count, damping, maximum correlated coordinates, diagonal terms, and low-memory evaluation. This is more than a magnitude score: the second-order factors estimate interactions and the update step attempts to restore the layer's local response after deletion.

The inspected model dispatch is explicitly OPT and LLaMA; the source does not establish support for arbitrary Hugging Face graphs, fused modern MLPs, MoE, or GGUF. The implementation is coupled to model-specific linear-module discovery and directly updates parameters before its own save/evaluation path. Its repository tree contains no automated test suite. The pinned requirements include Python-era dependencies `torch==2.1.0` and `transformers==4.34.1`; compatibility with ModelSurgeon's current Python/Torch/Transformers stack is not established.

## Licence and acquisition decision

The exact upstream [`LICENSE`](https://github.com/Qualcomm-AI-research/llm-surgeon/blob/726734425b6866782048877bac0c0d90e8cb36b3/LICENSE) identifies BSD-3-Clause-Clear and requires retaining the copyright, conditions, and disclaimer in source redistributions, reproducing them in binary distributions, and not using the authors' names for endorsement. It states no express patent license. This is not a legal clearance for a particular distribution; any direct port must preserve these obligations and receive a file-level notice review.

**Decision: ADAPT the mathematical scoring/compensation principles behind ModelSurgeon's existing architecture adapter and orchestrator; do not adopt the standalone optimizer or dependency as-is.** No ModelSurgeon KFAC feature or physical comparison exists yet. The repository's `features/sensitivity_spike.py` is a bounded synthetic probe with supplied diagonal-Hessian values; it is not a KFAC collector or a real-model second-order pruning result. Existing Hugging Face channel proof code measures real activations/gradients and physical masks, but does not collect Kronecker activation/output-gradient factors or compensate coupled channel surgery with them.

## Testable hypothesis and validation plan

**Hypothesis:** a bounded, architecture-aware Kronecker-curvature score plus local compensation can select MLP-channel removals with lower held-out loss increase than random, magnitude, activation, and first-order gradient scores at the same physically removed parameter count. It may reduce the number of expensive physical candidate evaluations if it predicts measured damage consistently.

Validation should proceed in separate stages:

1. Verify the factorized quadratic cost against exact dense Hessian calculations on deterministic tiny linear layers, including PSD, singular, damping, and finite-value cases.
2. Add bounded factor collection for a specifically supported Hugging Face architecture; report factor storage, calibration time, and peak CPU/GPU memory. Preserve no-curvature and no-repair arms.
3. On a pinned small pretrained checkpoint, compare equal-size removals selected by random seeds, existing magnitude/activation/gradient methods, and KFAC. Use separate calibration and held-out data, reload every physically reduced artifact, and enforce the existing quality gate.
4. Only evaluate compensation after the uncompensated baseline is measured; compare quality recovered per optimization time and memory. Do not infer inference speed from parameter count.
5. Repeat calibration seeds and report uncertainty. A tiny algebraic test validates the formula only; it does not qualify as optimization evidence.

The pinned upstream CLI suggests calibration and matrix-factor work can be substantial. Consumer-hardware viability, current framework compatibility, modern architecture support, and any quality/performance gain remain unproven. Do not install its old dependency set into ModelSurgeon or claim the method is superior based on the paper or upstream results alone.

# ModelSurgeon Roadmap

The GitHub Project is the execution source of truth; this document explains its milestones and critical path. Every Project leaf is intended to fit roughly 0.5–2 focused engineering days and has objective acceptance criteria, tests, ownership fields and dependencies.

```mermaid
flowchart LR
  A[v0.1 Foundation] --> B[v0.2 Instrumentation]
  B --> C[v0.3 Mutation Lab]
  C --> D[v0.4 Experiment Dataset]
  D --> E[v0.5 First Surgeon]
  E --> F[v0.6 Active Surgeon]
  C --> G[v0.7 Structural + Native GGUF Surgery]
  F --> H[v0.8 Cross-Model Generalization]
  G --> H
  H --> I[v0.9 Automated Optimization]
  I --> J[v1.0 ModelSurgeon]
  J --> K[v1.1 Competitive Ground Truth]
  K --> L[v1.2 Physical Deployment Pareto]
  L --> M[v1.3 Hardware-Aware Surgery]
  M --> N[v1.4 Multi-Axis Optimization]
  N --> O[v1.5 State-Dependent Surgeon]
  O --> P[v1.6 Cross-Model Meta-Surgeon]
  P --> Q[v1.7 Learned Repair]
  Q --> R[v1.8 Scientific Rigor]
  R --> S[v1.9 Production Ecosystem]
  S --> T[v2.0 Autonomous Optimizer]
```

## v0.1 — Foundation

Establish repository quality, reproducible configuration/logging, safe hardware discovery, Hugging Face and safetensors loading, stable component identities, architecture discovery and the coupling graph. Specify GGUF container, quantization and architecture boundaries now so the low-hardware path is not bolted on later.

## v0.2 — Instrumentation

Add calibration datasets, lifecycle-safe activation and gradient collection, streaming aggregation, static/spectral/runtime/redundancy features, versioned records and quantization provenance. Demonstrate bounded collection on the 12 GB reference GPU and CPU.

## v0.3 — Mutation Lab

Implement the transactional mutation contract, masking for heads/channels/components, layer bypass, rollback/provenance and tier 0/1 evaluation. Deliver the first safe mask-and-measure loop before physical resizing.

## v0.4 — Experiment Dataset

Build migrated SQLite metadata, Parquet features/examples, content-addressed artifacts, resumable queues, hardware budgets, OOM recovery and automated mutation campaigns. Produce a leakage-checked training dataset with thousands of small-model examples.

## v0.5 — First Surgeon

Train heuristic, magnitude, random, linear/logistic, LightGBM and small-MLP baselines. Measure AUC, delta-perplexity MAE/RMSE and precision at top N on held-out components. Quantization and feature precision are explicit covariates.

## v0.6 — Active Surgeon

Calibrate probabilities and uncertainty, generate and deduplicate large candidate pools, combine uncertainty/utility/diversity, schedule within budgets and trigger iterative retraining. Test whether active learning reduces required experiments.

## v0.7 — Structural and Native GGUF Surgery

Physically resize HF/safetensors tensors and implement first-class out-of-core GGUF surgery. Deliver mmap/lazy/container I/O, exact quantization codecs, bounded decode/encode, transactional streaming output, architecture adapters, MLP/head/layer/low-rank mutations and llama.cpp validation. The milestone proof physically removes a Q4_K_M MLP channel group without a full floating-point model, then reports size, parameters, perplexity, quality, throughput and peak RAM.

## v0.8 — Cross-Model Generalization

Evaluate Llama, Qwen, Mistral and Gemma families across roughly 100M to 7B where practical. Add model- and architecture-held-out protocols, native GGUF compatibility matrices, quantization controls and systematic research experiments Q1–Q8.

## v0.9 — Automated Optimization

Implement constrained multi-objective search across sequences of graph-valid mutations, Pareto tracking, keep/rollback policies, LoRA/short-fine-tuning/distillation repair and iterative surgery comparisons. Support full/tensor/streaming memory-mode planning.

## v1.0 — ModelSurgeon

This is the current unfinished milestone. Stabilize public schemas and CLI workflows; complete explainability, reports, reproduce/export commands, performance/regression suites, security hardening, release docs and reproducible reference experiments on consumer hardware. Post-v1 work remains blocked by its declared open v1.0 prerequisites; it does not replace or imply completion of them.

## v1.1 — Competitive Benchmarking and Ground Truth

Implement strong Wanda, SparseGPT, structured-pruning and quantized baselines behind equal-budget contracts. Exit with reproducible, statistically defensible quality/throughput/memory/size/cost Pareto evidence on physically deployable artifacts. New search algorithms are out of scope.

## v1.2 — Physical Compression and Deployment Pareto Optimization

Make cumulative HF and GGUF surgery physically smaller, reloadable and measurable, with artifact-correctness gates and explicit surgery-versus-quantization loss. Exit with deployment frontiers and retained failure evidence; a learned hardware policy is deferred.

## v1.3 — Hardware-Aware Surgery

Profile target machines and learn measured kernel/alignment/resource costs rather than treating parameter count as deployment benefit. Exit when conditioned decisions improve or honestly fail to improve held-out consumer-hardware frontiers.

## v1.4 — Multi-Axis Architecture Optimization

Represent complete architecture states and search graph-valid combinations of depth, width, heads, MLP channels, rank and precision. Exit with bounded beam/evolutionary/surrogate search and physically validated Pareto candidates; interaction learning belongs to v1.5.

## v1.5 — Interaction-Aware and State-Dependent Surgeon

Collect ordered mutation interactions and condition predictions on cumulative architecture state. Exit when long-sequence studies quantify cumulative regret, violations, rollbacks and evaluation cost against stateless/additive baselines.

## v1.6 — Cross-Model Meta-Surgeon

Define lineage-safe model meta-features, compatibility rules and transfer confidence. Exit with four-family zero/few-shot evidence showing when transferable knowledge saves target evaluations and when target-specific retraining is required.

## v1.7 — Learned Recovery, Distillation and Repair

Predict recoverability and repair cost, then jointly choose surgery, repair and quantization under consumer budgets. Exit with measured no-repair, LoRA, fine-tuning, distillation and oracle comparisons; repair is never assumed to rescue an unsafe mutation.

## v1.8 — Robustness, Security, Validation and Scientific Rigor

Add hostile artifacts, mutation properties, crash consistency, cross-platform numerical validation, leakage controls, signed evidence and clean-environment reproduction. Exit only after independent audit of claims and retained negative/failed cells.

## v1.9 — Production UX, Ecosystem and Large-Scale Optimization

Deliver a resumable `optimize` workflow, worker scheduling, artifact registry, stable plugin/runtime interfaces, an evidence explorer and representative user workflows. Exit with practical local and multi-worker campaigns; autonomous policy selection remains v2.0 work.

## v2.0 — Autonomous Evidence-Driven Model Optimizer

Turn a constrained user objective and hardware profile into an auditable capability space, strategy, mutation/evaluation/repair/quantization workflow and deployable alternatives. Exit with human approval gates, deterministic replay, a competitive reference benchmark, signed release artifacts and a scientific report that clearly separates verified, experimental, unsupported and unknown capabilities.

## Central critical path

```text
configuration and safety contracts
  -> stable component IDs
  -> HF + GGUF architecture discovery
  -> coupling graph and constraint validation
  -> instrumentation + versioned features
  -> transactional masking + tiered evaluation
  -> experiment schema/store + resumable campaign runner
  -> leakage-safe dataset
  -> LightGBM surgeon + calibrated uncertainty
  -> active candidate scheduler
  -> physical mutation planner
  -> quantization codecs + streaming GGUF writer
  -> native Q4_K_M MLP proof
  -> cross-model/quantization evaluation
  -> constrained iterative search
  -> reproducible v1.0 workflow
  -> equal-budget competitor ground truth
  -> physical deployment Pareto evidence
  -> measured hardware-conditioned cost models
  -> multi-axis architecture-state search
  -> interaction-aware cumulative prediction
  -> cross-family meta-learning
  -> joint surgery + repair + quantization economics
  -> hostile-input and signed-evidence audit
  -> resumable production optimize workflow
  -> autonomous objective-to-artifact v2.0 release
```

## Scientific questions

The tracked research program asks whether static features predict safe pruning; how much activations and gradients add; whether learned rankings beat magnitude and random baselines; whether predictions transfer across models and families; whether active learning reduces experiments; whether iterative surgery beats one-shot pruning; which quantized features remain reliable; and how surgery loss separates from requantization loss.

Post-v1 research extends those questions without erasing negative results: whether ModelSurgeon is competitive under equal deployment budgets; which physical mutations create real hardware gains; whether cumulative interactions, meta-learning and recoverability estimates generalize; and whether an autonomous policy can produce a better feasible frontier with fewer evaluations while preserving provenance, safety and human control.

The [ModelSurgeon Roadmap Project](https://github.com/users/xdCloudy/projects/2) is the source of truth for individual issue status, priority, effort, risk, phase and native blocker relationships.


# Teacher-guided native weight understanding for the Meta-Surgeon

**Status: proposed research design, not an implemented capability or demonstrated result.**

## Objective

Train ModelSurgeon's own specialized Meta-Surgeon to interpret neural-network weights and their functional relationships, predict the consequences of interventions, and eventually propose useful edits **without depending on a general-purpose reasoning LLM at inference time**. A capable reasoning model may be used **offline as a teacher** to propose weight-space hypotheses, explanations, contrasts and candidate interventions. A reproducible measurement harness must independently test these proposals before they become trusted supervision or accepted model artifacts.

This extends the existing architecture-aware, outcome-trained Meta-Surgeon; it does not replace that system with an LLM agent, a prompt-driven optimizer or a new source of acceptance authority. The text LLM in the conversational product remains a separate, optional user-interface/control-plane component.

The existing learned surgeon currently focuses on engineered features, candidate ranking, outcome prediction and cross-model transfer. None of those capabilities alone proves that it can identify the *function* of a previously unseen circuit from weights. Native weight understanding is a **new, independently measurable research target**, not a synonym for a good pruning score.

## Motivation and source boundary

Takehi, Teoh, Zhang and Khattab, *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights* (MIT CSAIL, 2 November 2026), report early experiments in which a frontier reasoning model:

- inferred tasks and some circuit functions from weights of small controlled networks;
- proposed direct weight changes for a failing control policy;
- improved a looping-prone Qwen3.5-0.8B checkpoint through a targeted intervention with forward-pass feedback.

Source: https://rikiyat.github.io/blog/weight-space-programming/

These are external, experimental findings, **not ModelSurgeon results**. Important negative controls matter: weight-only classification of sycophancy in the reported Gemma 3 270M experiment failed until attention observations were allowed; editing another Gemma 3 270M checkpoint did not establish meaningful math improvement. The Qwen math result mainly removed a completion-blocking loop, not proof that arbitrary absent capabilities can be written into a model. Generalization to model families, sizes, distributions or hardware outside those experiments is unproven.

The proposal is to **distil testable weight-space knowledge into the student**, not to deploy the original frontier model as the optimizer.

## Role separation

1. **Reasoning teacher (optional, offline research):** inspect bounded tensors, architecture descriptions and authorized diagnostics; propose falsifiable functional hypotheses, candidate interventions and uncertainty estimates. The teacher's natural-language explanations are untrusted until checked.
2. **ModelSurgeon execution/evidence engine:** control tensor identity and access, verify the operation geometry, perform interventions transactionally, run independent probes/measurements, enforce budgets and quality gates, preserve both negative and positive evidence, and retain rollback and provenance.
3. **Learned Meta-Surgeon (student):** train on architecture-conditioned weight representations and verified outcomes, predict functions/intervention effects with uncertainty, abstain out of distribution and eventually rank/propose interventions independently.
4. **Optional conversational text LLM:** translate user intent into structured objectives and explain canonical results. It is not a weight oracle, training teacher by default, evaluator or artifact-promotion authority.

Offline teaching flow:

1. Select a pinned target checkpoint and an allowed, reproducible functional question.
2. Gather bounded weight views, topology, and separately labelled optional activations/probes.
3. Query the teacher for a typed hypothesis and/or a typed surgery proposal; retain raw teacher output strictly as untrusted annotation.
4. Independently test hypotheses with held-out functional probes and causal interventions. Compare proposed edits to unchanged, random and established method baselines under equal budgets.
5. Save a lineage-pinned record containing input views, candidate, teacher version, claimed rationale, measured outcome, failures, resource usage and final status.
6. Train the student against **verified, independently measured targets**, with teacher-derived annotations only as auxiliary supervision.
7. At deployment, use the **student alone** for predictions/proposals, followed by the existing physical validation and acceptance gates. No teacher request is required.

No teacher-produced Python, shell commands, arbitrary weight writes or measurement assertions may bypass typed capabilities or existing sandbox/approval boundaries.

## What the student must learn

### Inputs and representations

Preserve the current architecture/feature/lineage contracts, while researching additional representations that expose **weight structure itself**, such as:

- localized tensor blocks, weight-row/column sets and matched cross-layer projections;
- topology and coupling metadata, positional/layer/role information, and attention/MLP/expert structure;
- normalized raw-weight statistics together with bounded learned tensor embeddings and cross-layer relational features;
- controlled optional activation and attribution evidence, **explicitly distinguished** from pure weight-only inputs;
- quantization codec and uncertainty, so decoded approximations are never misrepresented as exact source weights;
- intervention history, previous outcomes, current architecture and hardware/resource objectives.

A successful representation should generalize across function-preserving neuron permutations, compatible rescalings, irrelevant parameterization changes and model sizes where appropriate. Do not assume raw float sequences are semantically sufficient, or that unlike model architectures have interchangeable channels/heads. Larger models need sampled/bounded, hierarchical views rather than dumping whole checkpoints into context or GPU memory.

### Training targets

Study multiple distinct tasks rather than treating edit success as proof of interpretability:

- **Functional identification:** classify independently established tasks/circuits or predict functional properties from specified weight-only views.
- **Causal localization:** rank components/directions whose controlled intervention changes a measured behavior.
- **Intervention effects:** predict delta quality, behavioral change, loop frequency, deployment costs and uncertainty under typed perturbations.
- **Useful edits:** propose graph-valid, budgeted corrective/simplifying operations and predict whether they will survive independent quality checks.
- **Transfer and abstention:** retain calibration on held-out checkpoints/families; explicitly refuse to infer unsupported semantics.

Teacher explanations are not ground-truth circuit labels. Use designed synthetic mechanisms, analytic networks, interventions, teacher-independent probes and blinded behavioral tests to establish labels. Separate **prediction quality**, **causal validity**, **artifact effectiveness**, and **explanation faithfulness**.

## Dataset, controls and leakage prevention

Each example should include immutable source and candidate hashes, architecture family and revision, tensor/view identities and exact precision, teacher identity/version/settings, available tools and feedback budget, prompt/template revision, candidate operation parameters, measured controls, probe/test revisions, random seeds, resource costs and disposition. Keep rejected, unsupported, contradictory and failed teacher suggestions.

Split by **complete model lineage** and task family where required; never let sibling checkpoints, teacher-generated paraphrases, equivalent permutations or identical evaluation episodes bridge train/test. Pre-register weight-only versus weight-plus-activations conditions and no-feedback versus bounded-feedback conditions. Keep final evaluation tasks hidden from both teacher and student. Audit teacher contamination and use controlled synthetic networks with known mechanisms where possible.

Compare, at matched compute/evaluation budget:

- existing Meta-Surgeon engineered-feature and learned-ranker baselines;
- random, magnitude, activation- and gradient-aware candidates where applicable;
- teacher-only intervention proposals (research reference, not production acceptance);
- student trained on measured outcomes **without** teacher supervision;
- student trained with teacher annotations, plus an ablation with shuffled or incorrect teacher annotations.

Report held-out circuit accuracy, localization precision, intervention effect prediction/calibration, failed/unsafe edit rates, evaluation cost, and final **physically deployed** Pareto results. Include three or more independent seeds where practical and uncertainty intervals grouped by model lineage. Do not label high annotation agreement as successful native understanding.

## Staged research program (proposed; not part of an already closed milestone)

1. **R0 — Task and protocol audit.** Reproduce small, inspectable weight-reading tasks and explicit failure cases on independent fixtures; pin sources and teacher/feedback budget. Exclude any production claims.
2. **R1 — Typed teacher evidence contract.** Define schema, lineage, authorized probes, cost accounting, untrusted-annotation storage, benchmark splits and refusal behavior. No unrestricted execution.
3. **R2 — Student weight encoder.** Compare engineered features versus bounded raw/local-weight and relational/graph/set representations on held-out functional and causal targets. Evaluate permutation and reparameterization robustness.
4. **R3 — Teacher-guided training.** Train paired students using identical measured labels with and without verified teacher annotations. Require evidence of incremental benefit, not merely teacher imitation.
5. **R4 — Independent proposal and validation.** Permit the student, with teacher disconnected, to suggest typed interventions that the existing surgery engine checks, runs and either rejects or accepts.
6. **R5 — Cross-family and consumer-hardware transfer.** Extend only after successful R0–R4 evidence. Test dense and MoE families/quantized views individually, retain unsupported cases, and measure actual size/quality/latency/VRAM frontiers on declared hardware.

The first small-scale proof should ask: **Can a teacher-supervised student infer an unseen network's function or circuit and predict a causal intervention more accurately than the same student trained only on measured outcomes, without consulting its teacher at test time?**

## Acceptance and stopping rules

- No claim of native weight understanding without independently held-out functional/circuit and intervention tests; do not infer semantic understanding from a low loss or a plausible rationale.
- No claim of better optimization without a reproducible **physical** frontier gain or evaluation-cost reduction over matched, teacher-free controls and all hard quality/resource constraints.
- No deployment dependence on a frontier teacher. If teacher supervision fails to add generalizable value, keep the simpler student and publish the negative finding.
- The learned student's output remains a **proposal/prediction**, never a measurement or automatic approval. Existing immutable inputs, validated operations, consent/approval, fail-closed compatibility, rollback, and evidence-grounded publication remain authoritative.
- No universal GGUF, MoE, interpretability, unseen-family or security support is implied by this design; each newly validated adapter/operation/codec requires separate evidence.

## Relationship to existing milestones

- The v0.4–v0.6 experiment dataset, candidate predictor and active-learning work provide the initial evidence infrastructure.
- The v1.5 cumulative architecture state and v1.6 cross-family Meta-Surgeon representations are prerequisites and controls, **not completed proof of teacher-guided native weight understanding**.
- The v1.7 teacher-cache/logit-distillation repair work repairs **target models**; it is not a teacher-to-Meta-Surgeon training pipeline.
- The v2.0/v3.0 engine, conversational product and artifact gates stay unchanged. An optional **post-v1.6/v3.1 research track** can develop the teacher–student mechanism without reopening frozen release contracts.
- The canonical status/progress artifacts must not count this proposal as implemented or scored until code, tests and reproducible evidence exist.

See also [long-term goal](../goal.md), [architecture](../../ARCHITECTURE.md), [roadmap](../../ROADMAP.md), [current Meta-Surgeon evidence](../progress/META_SURGEON.md) and [cross-family feature contract](meta-surgeon-features.md).

# ModelSurgeon progress and scoring

Assessment date: **2026-10-27** · assessed main commit: `589c609e1571493e2aeb5ea717a3febf0447cf83` · inventory scope: `1.2.9`.

Conservative evidence-bounded inventory. This revision adds method-specific differentiable-mask, layer-regularization, dependency-pruning, and Meta-Surgeon lifecycle questions; narrows the iterative-schedule record to its actual Q8 evidence; and refreshes the backlog after PR #774 merged. Research discovery and upstream claims do not imply ModelSurgeon support or effectiveness. Scope version 1.2.2 separates bounded Qwen3.5 REAP quality-gate and profile-aggregation results from the broader MoE selection milestone; the REAP evidence remains experimental and unintegrated. Scope version 1.2.3 adds a bounded Qwen3.5 leave-one-profile-out REAP ranking-stability study; its small descriptive stability gain does not establish improved model quality or optimization effectiveness. Scope version 1.2.4 adds a separate bounded Qwen2.5 ShortGPT physical depth-selection result from open PR #735; its negative quality finding does not imply effectiveness, integration, or generalization. Scope version 1.2.5 adds an independently tracked, bounded negative CPU pilot for first-order gradient-based transformer depth sensitivity from open PR #779, stacked on #735; the score neither improves the matched magnitude result nor establishes practical effectiveness. Scope version 1.2.6 adds the first bounded KFAC physical MLP-channel selection comparison in open PR #780, stacked on #775; KFAC beats magnitude on this single held-out PPL cell but all tested methods fail the declared quality gate, and random is better at some small cuts. Scope version 1.2.7 adds a separate CPU KFAC replication on one Qwen2.5-0.5B cell in open PR #780: KFAC ranks best among five candidates but still fails the quality gate, with substantial workspace and scoring costs. This is one bounded cross-model observation, not evidence of broad transfer. Scope version 1.2.8 adds a pinned MIT WANDA source audit and an independently implemented structured activation-score baseline. On Qwen2.5-0.5B, KFAC ranks best across two short train calibration windows, but all candidate artifacts fail the quality gate and channel overlap across windows is low; no reproduction of original unstructured WANDA is claimed. Scope version 1.2.9 separately tracks the bounded Q7 feature-diversity acquisition replay in open PR #771: its fixed mixture is not consistently better than uncertainty-only or random, and its raw source inputs are unavailable for independent reproduction. A Q7 report-integrity follow-up corrected its line-ending-dependent digest to the normalized Git-blob digest and added a regression test; this does not change the replay result or resolve missing raw campaign inputs.
Canonical source: [`modelsurgeon-progress-v1.json`](modelsurgeon-progress-v1.json) · [JSON Schema](schema-v1.json) · [maintenance workflow](CONTRIBUTING.md). The dashboard is generated locally; repository edits remain reviewable in the same PR.

## Current scores

Overall weighted completion: **29.0%**. This is the defined, versioned scope score, not a release certification.

| Dimension | Score | Dimension weight in overall | Meaning |
| --- | ---: | ---: | --- |
| Research coverage | 50.6% | 20% | Progress through prior art, hypotheses, physical experiments, matched baselines, and reproducible conclusions. Negative conclusions count as completed research. |
| Implementation maturity | 39.0% | 25% | Progress through tested and integrated software; integrated means present on assessed main. |
| Optimization effectiveness | 5.9% | 35% | Evidence that methods meet quality/resource gates; negative or unmeasured methods score zero. |
| Production readiness | 33.6% | 20% | Verification, integration, regression, deployment, and operating readiness within declared scope. |
| Representative compatibility coverage | 36.1% | Separate coverage measure | Weighted share of explicit model/format/operation/backend/runtime/recovery cells with evidence-backed support. It is not added a second time to the overall score. |

## Scoring method

Each workstream has a positive integer weight; workstream weights total 100. Each workstream's active milestone weights total 100. A milestone contributes only to dimensions listed in `scored_dimensions`. Within a workstream, milestone-weighted dimension scores are combined using its explicit dimension weights (renormalized over applicable dimensions); the overall completion is then the workstream-weighted mean. Research, implementation, effectiveness, and readiness use the ordered score maps defined in `tools/progress.py`. Research completion applies only to the milestone's explicit research question and scope; broader open questions must be separate. Research outcome is not itself a score: an evidence-backed negative conclusion can have 100% research completion and 0% effectiveness. Compatibility coverage is separately reported; `verified` requires a plausible scope, implementation on assessed main, and physical or independent reproduction evidence.

Newly discovered scope is added as milestones and a dated scope-history entry. It expands the denominator and may reduce scores. A milestone can be retired only with a permanent record in `retired_milestones`; research conclusions and identifiers are not deleted. No unmerged PR contributes to integrated or main-readiness stages.

## Workstreams

| Workstream | Weight | Overall | Research | Implementation | Effectiveness | Readiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Learned Meta-Surgeon](FEATURE_MATRIX.md#meta-surgeon) | 22% | 21.5% | 48.6% | 40.0% | 3.6% | 34.4% |
| [Pruning and sparsification](FEATURE_MATRIX.md#pruning) | 10% | 22.9% | 50.0% | 24.8% | 2.0% | 17.8% |
| [Quantization](FEATURE_MATRIX.md#quantization) | 7% | 15.8% | 37.5% | 8.0% | 5.5% | 14.0% |
| [Mixture-of-Experts optimization](FEATURE_MATRIX.md#moe) | 8% | 7.1% | 21.9% | 3.0% | 1.0% | 3.0% |
| [Model recovery and repair](FEATURE_MATRIX.md#recovery) | 6% | 18.0% | 46.0% | 10.0% | 6.7% | 10.0% |
| [Architecture optimization](FEATURE_MATRIX.md#architecture) | 5% | 32.6% | 62.5% | 40.0% | 10.0% | 20.0% |
| [Search and optimization](FEATURE_MATRIX.md#search) | 7% | 36.3% | 75.2% | 33.8% | 10.0% | 30.4% |
| [Model and format compatibility](FEATURE_MATRIX.md#compatibility) | 9% | 47.5% | 46.0% | 80.0% | 6.7% | 70.0% |
| [Hardware and inference optimization](FEATURE_MATRIX.md#hardware) | 7% | 23.1% | 62.5% | 10.0% | 10.0% | 10.0% |
| [Benchmarking and evaluation](FEATURE_MATRIX.md#benchmarking) | 9% | 52.6% | 62.5% | 80.0% | 10.0% | 70.0% |
| [Independent scientific research](FEATURE_MATRIX.md#scientific-research) | 5% | 35.6% | 41.4% | 56.9% | 3.3% | 50.2% |
| [Production readiness](FEATURE_MATRIX.md#production-readiness) | 5% | 52.6% | 62.5% | 80.0% | 10.0% | 70.0% |

## Priorities

1. **Feature-diversity acquisition replay** (`meta_surgeon.active_learning_diversity`): The replay inputs are not committed or linked to a versioned archive. Obtain the three exact source bundles by hash and independently reproduce the recorded curves; then run prospective acquisition studies across additional model families. Keep the fixed mixture unpromoted until prospective evidence demonstrates value.
2. **Learned decision effectiveness on physical optimization** (`meta_surgeon.prospective_effectiveness`): Run a prospective physical comparison on a CPU-feasible model first; GPU-scale tests remain conditional on user availability.
3. **Training data acquisition and provenance** (`meta_surgeon.dataset_generation`): Design a prospective collection protocol and evaluate coverage, missingness, duplicate/near-duplicate outcomes and leakage across campaigns.
4. **Predictor architecture and objective selection** (`meta_surgeon.predictor_selection`): Define compute-bounded tuning and a prospective held-out decision protocol.
5. **Differentiable structured masks** (`pruning.differentiable_masks`): Audit full training objective and reference paper, then run an isolated CPU correctness fixture before any model-scale experiment.
6. **Layer removal with pre-pruning regularization** (`pruning.depth_regularization`): Inspect the complete layer-scaling and removal path and reproduce against ShortGPT on a small supported checkpoint.
7. **Expert scoring: random, router, activation, sensitivity and balancing** (`moe.selection`): Complete matched comparisons for activation and sensitivity selectors, test further MoE cells, and measure deployment costs; bounded Qwen3.5 REAP quality and profile studies are tracked separately.
8. **Hardware-conditioned quality and resource prediction** (`meta_surgeon.resource_prediction`): Add pinned multi-device evidence, define deployment metrics and measure model/hardware transfer before training this predictor.

## Critical-path dependencies

A dependent milestone remains bounded by every listed prerequisite; opening a PR does not satisfy a dependency.

| Milestone | Depends on |
| --- | --- |
| `meta_surgeon.decision` — Q4 held-out single-channel learned ranking versus controls | `meta_surgeon.data`, `meta_surgeon.features_static`, `meta_surgeon.features_activation`, `meta_surgeon.features_gradient`, `search.controls`, `benchmarking.statistics` |
| `pruning.schedules` — Q8 iterative/progressive pruning schedule study | `pruning.physical`, `search.controls` |
| `pruning.unstructured_baselines` — Pinned baseline adapter availability: WANDA and SparseGPT | `pruning.physical` |
| `pruning.structured_baselines` — Pinned baseline adapter availability: structured pruning methods | `pruning.physical` |
| `pruning.dependency_methods` — Dependency-aware structured pruning libraries | `pruning.physical` |
| `pruning.differentiable_masks` — Differentiable structured masks | `pruning.physical` |
| `pruning.depth_regularization` — Layer removal with pre-pruning regularization | `pruning.physical` |
| `pruning.depth_shortgpt` — ShortGPT block-influence depth selection | `pruning.physical` |
| `pruning.depth_gradient_sensitivity` — First-order gradient sensitivity for transformer depth removal | `pruning.physical` |
| `quantization.joint` — Pruning/quantization order and repair interactions | `quantization.codecs`, `pruning.physical` |
| `quantization.gsq_rco` — GSQ and RCO quantization research | `quantization.codecs` |
| `moe.surgery` — Expert pruning/merge, redundancy, router, shared/routed and layerwise allocation | `moe.selection` |
| `moe.joint` — MoE pruning plus quantization/recovery and expert specialization | `moe.surgery`, `quantization.codecs`, `recovery.lora` |
| `moe.expert_merging` — MoE expert merging and redundancy reduction | `moe.surgery` |
| `recovery.joint` — Learned recovery prediction and joint structural/quantization repair | `recovery.lora`, `recovery.distill`, `quantization.codecs`, `pruning.physical` |
| `architecture.transforms` — Low rank, SliceGPT rotations, depth removal and multi-axis architecture search | `architecture.coupling` |
| `search.policies` — Bayesian/evolutionary/beam/surrogate and active-learning acquisition | `search.controls`, `meta_surgeon.decision` |
| `search.constraints` — Pareto, budget/multifidelity, hardware and state-dependent sequential search | `search.policies`, `hardware.resources`, `hardware.inference` |
| `compatibility.boundaries` — MoE, multimodal, unsupported configs, load and validation boundaries | `compatibility.families`, `compatibility.formats` |
| `hardware.inference` — Prefill/decode throughput, TTFT, latency, bandwidth and kernels | `hardware.resources` |
| `benchmarking.artifact` — Physical size, parameters, reload, runtime, reliability and rollback | `compatibility.formats` |
| `benchmarking.statistics` — Independent splits/replays, significance, uncertainty and contamination prevention | `benchmarking.artifact`, `benchmarking.quality` |
| `scientific_research.emerging` — GSQ/RCO, expert merging, recovery, meta-learning and novel methods | `scientific_research.competitors`, `scientific_research.negative` |
| `production_readiness.operations` — Reproducibility, observability, docs/tests, packaging, deployment and maintenance | `production_readiness.interfaces`, `production_readiness.safety`, `benchmarking.statistics` |
| `meta_surgeon.active_learning_diversity` — Feature-diversity acquisition replay | `meta_surgeon.active_learning` |
| `meta_surgeon.features_static` — Q1 static weight-feature baseline | `meta_surgeon.data` |
| `meta_surgeon.features_activation` — Q2 activation-feature ablation | `meta_surgeon.features_static` |
| `meta_surgeon.features_gradient` — Q3 gradient-feature ablation | `meta_surgeon.features_activation` |
| `pruning.kfac_physical_pilot` — KFAC singleton-additive MLP channel ranking physical comparison | `pruning.physical` |
| `pruning.kfac_qwen25_replication` — KFAC singleton-additive channel selection on Qwen2.5-0.5B | `pruning.kfac_physical_pilot` |

## Retired scope identifiers

Retired IDs remain traceable; their complete former records and histories stay in the canonical JSON.

| Former ID | Retired | Superseded by | Former research result | Reason |
| --- | --- | --- | --- | --- |
| `meta_surgeon.features` | 2026-10-27 | `meta_surgeon.features_static`, `meta_surgeon.features_activation`, `meta_surgeon.features_gradient` | Inconclusive | Superseded by separate Q1 static, Q2 activation, and Q3 gradient feature studies so their distinct outcomes are independently scored; full former record and history retained below. |

## Current blockers

| Blocker | Impact | Next action |
| --- | --- | --- |
| **meta-surgeon-utility** · Learned Meta-Surgeon has no demonstrated reliable physical advantage over strong random/heuristic baselines on unseen models. | Central scientific goal remains unproven. | Run preregistered equal-budget model-grouped physical comparisons with independent task and deployment measurements. |
| **moe-evidence** · Qwen MoE expert surgery/REAP comparisons remain in open PRs. | No main-integrated MoE support or accepted quality/performance result. | Complete matched methods and review dependency chain without counting branches as main. |
| **upstream-baselines** · Upstream sparse/structured executables are not available in current test environment. | Comparative superiority claims lack measured established baselines. | Audit licenses and implement pinned isolated adapters where appropriate. |
| **q7-feature-diversity-inputs** · The Q7 feature-diversity replay report contains hashes and curves, but its examples, split manifests, and gradient manifests are not available in the repository or a versioned archive. | The reported 45-curve acquisition result cannot be independently reproduced or checked against its input provenance. | Locate or archive the exact three source bundles by hash, rerun the study, and update the milestone without upgrading effectiveness unless the replay and prospective tests support it. |

## Commands

```bash
uv run python tools/progress.py validate
uv run python tools/progress.py generate
uv run python tools/progress.py check
```

`generate` writes derived README/matrices only when content changes. `check` is read-only and fails with file-specific stale-output diagnostics; CI runs `validate` and `check` and never commits generated updates.

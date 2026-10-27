# ModelSurgeon progress and scoring

Assessment date: **2026-10-27** · assessed main commit: `6820c77c6c090bbbd7c821b16a39041f8c39c980` · inventory scope: `1.0.0`.

Conservative initial evidence audit across the known ModelSurgeon research and engineering ambition. Research, implementation, effectiveness, and production readiness are independent dimensions. Infrastructure and negative research do not imply an effective optimizer. Scope expands append-only as avenues are discovered; that may lower percentages.
Canonical source: [`modelsurgeon-progress-v1.json`](modelsurgeon-progress-v1.json) · [JSON Schema](schema-v1.json) · [maintenance workflow](CONTRIBUTING.md). The dashboard is generated locally; repository edits remain reviewable in the same PR.

## Current scores

Overall weighted completion: **35.1%**. This is the defined, versioned scope score, not a release certification.

| Dimension | Score | Dimension weight in overall | Meaning |
| --- | ---: | ---: | --- |
| Research coverage | 63.4% | 20% | Progress through prior art, hypotheses, physical experiments, matched baselines, and reproducible conclusions. Negative conclusions count as completed research. |
| Implementation maturity | 50.8% | 25% | Progress through tested and integrated software; integrated means present on assessed main. |
| Optimization effectiveness | 6.4% | 35% | Evidence that methods meet quality/resource gates; negative or unmeasured methods score zero. |
| Production readiness | 43.5% | 20% | Verification, integration, regression, deployment, and operating readiness within declared scope. |
| Representative compatibility coverage | 48.5% | Separate coverage measure | Weighted share of explicit model/format/quantization/operation cells with evidence-backed support. It is not added a second time to the overall score. |

## Scoring method

Each workstream has a positive integer weight; workstream weights total 100. Each workstream's active milestone weights total 100. A milestone contributes only to dimensions listed in `scored_dimensions`. Within a workstream, milestone-weighted dimension scores are combined using its explicit dimension weights (renormalized over applicable dimensions); the overall completion is then the workstream-weighted mean. Research, implementation, effectiveness, and readiness use the ordered score maps defined in `tools/progress.py`. Research outcome is not itself a score: an evidence-backed negative conclusion can have 100% research completion and 0% effectiveness. Compatibility coverage is reported separately to avoid double counting.

Newly discovered scope is added as milestones and a dated scope-history entry. It expands the denominator and may reduce scores. A milestone can be retired only with a permanent record in `retired_milestones`; research conclusions and identifiers are not deleted. No unmerged PR contributes to integrated or main-readiness stages.

## Workstreams

| Workstream | Weight | Overall | Research | Implementation | Effectiveness | Readiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Learned Meta-Surgeon](FEATURE_MATRIX.md#meta-surgeon) | 22% | 41.7% | 100.0% | 80.0% | 3.4% | 70.0% |
| [Pruning and sparsification](FEATURE_MATRIX.md#pruning) | 10% | 39.4% | 66.6% | 53.5% | 6.7% | 36.6% |
| [Quantization](FEATURE_MATRIX.md#quantization) | 7% | 19.4% | 46.0% | 10.0% | 6.7% | 16.8% |
| [Mixture-of-Experts optimization](FEATURE_MATRIX.md#moe) | 8% | 4.7% | 12.6% | 3.4% | 0.0% | 3.4% |
| [Model recovery and repair](FEATURE_MATRIX.md#recovery) | 6% | 18.0% | 46.0% | 10.0% | 6.7% | 10.0% |
| [Architecture optimization](FEATURE_MATRIX.md#architecture) | 5% | 32.6% | 62.5% | 40.0% | 10.0% | 20.0% |
| [Search and optimization](FEATURE_MATRIX.md#search) | 7% | 36.3% | 75.2% | 33.8% | 10.0% | 30.4% |
| [Model and format compatibility](FEATURE_MATRIX.md#compatibility) | 9% | 47.5% | 46.0% | 80.0% | 6.7% | 70.0% |
| [Hardware and inference optimization](FEATURE_MATRIX.md#hardware) | 7% | 23.1% | 62.5% | 10.0% | 10.0% | 10.0% |
| [Benchmarking and evaluation](FEATURE_MATRIX.md#benchmarking) | 9% | 52.6% | 62.5% | 80.0% | 10.0% | 70.0% |
| [Independent scientific research](FEATURE_MATRIX.md#scientific-research) | 5% | 35.6% | 41.4% | 56.9% | 3.3% | 50.2% |
| [Production readiness](FEATURE_MATRIX.md#production-readiness) | 5% | 52.6% | 62.5% | 80.0% | 10.0% | 70.0% |

## Priorities

1. **Prediction calibration, uncertainty, active learning, transfer, OOD and autonomy** (`meta_surgeon.decision`): Establish cross-model/family/hardware transfer, calibrated abstention and incremental-learning benefit.
2. **Expert scoring: random/router/activation/sensitivity/REAP and balancing** (`moe.selection`): Complete pinned REAP source audit and fair expert-selection benchmarks.
3. **Independent splits/replays, significance, uncertainty and contamination prevention** (`benchmarking.statistics`): Add repeated independent optimizer superiority studies.
4. **Iterative/progressive pruning and WANDA/SparseGPT/Torch-Pruning/LLM-Pruner baselines** (`pruning.schedules`): Run pinned upstream adapters and repeated physical schedule comparisons.
5. **Mixed precision, activation-weighted allocation, GSQ/RCO and error estimation** (`quantization.allocation`): Resolve GSQ/RCO primary sources and test sensitivity-based allocation.

## Critical-path dependencies

A dependent milestone remains bounded by every listed prerequisite; opening a PR does not satisfy a dependency.

| Milestone | Depends on |
| --- | --- |
| `meta_surgeon.decision` — Prediction calibration, uncertainty, active learning, transfer, OOD and autonomy | `meta_surgeon.data`, `meta_surgeon.features`, `search.controls`, `benchmarking.statistics` |
| `pruning.schedules` — Iterative/progressive pruning and WANDA/SparseGPT/Torch-Pruning/LLM-Pruner baselines | `pruning.physical`, `search.controls` |
| `quantization.joint` — Pruning/quantization order and repair interactions | `quantization.codecs`, `pruning.physical` |
| `moe.surgery` — Expert pruning/merge, redundancy, router, shared/routed and layerwise allocation | `moe.selection` |
| `moe.joint` — MoE pruning plus quantization/recovery and expert specialization | `moe.surgery`, `quantization.codecs`, `recovery.lora` |
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

## Current blockers

| Blocker | Impact | Next action |
| --- | --- | --- |
| **meta-surgeon-utility** · Learned Meta-Surgeon has no demonstrated reliable physical advantage over strong random/heuristic baselines on unseen models. | Central scientific goal remains unproven. | Run preregistered equal-budget model-grouped physical comparisons with independent task and deployment measurements. |
| **moe-evidence** · Qwen MoE expert surgery/REAP comparisons remain in open PRs. | No main-integrated MoE support or accepted quality/performance result. | Complete matched methods and review dependency chain without counting branches as main. |
| **upstream-baselines** · Upstream sparse/structured executables are not available in current test environment. | Comparative superiority claims lack measured established baselines. | Audit licenses and implement pinned isolated adapters where appropriate. |

## Commands

```bash
uv run python tools/progress.py validate
uv run python tools/progress.py generate
uv run python tools/progress.py check
```

`generate` writes derived README/matrices only when content changes. `check` is read-only and fails with file-specific stale-output diagnostics; CI runs `validate` and `check` and never commits generated updates.

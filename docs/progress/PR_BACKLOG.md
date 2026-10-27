# Open pull-request and issue backlog snapshot

GitHub CLI snapshot: **2026-10-27** · assessed `main` `6820c77c6c090bbbd7c821b16a39041f8c39c980` · 0 open issues · 56 open PRs.

All 56 open PRs were enumerated; the following 21 optimization, evidence, and progress-system PRs are retained individually for prioritization. Counts and branch relationships are a dated snapshot, not live CI state. No PR was merged by this audit.

All open PRs were enumerated; notable optimizer/research and stacked-dependency changes are listed below. This is a dated snapshot, not live status. An open PR or green check is not integrated functionality.

## Stacked dependencies

Chains are listed in base-first order. Re-fetch branch bases and review conflicts before integration.

| Chain | Branch relationship |
| --- | --- |
| [#769](https://github.com/xdCloudy/ModelSurgeon/pull/769) → [#770](https://github.com/xdCloudy/ModelSurgeon/pull/770) → [#771](https://github.com/xdCloudy/ModelSurgeon/pull/771) | `codex/normalize-diversity-distance → codex/integrate-diversity-acquisition → codex/evaluate-feature-diversity-acquisition` |
| [#693](https://github.com/xdCloudy/ModelSurgeon/pull/693) → [#768](https://github.com/xdCloudy/ModelSurgeon/pull/768) | `codex/abstain-meta-unknown → codex/abstain-meta-numeric` |
| [#749](https://github.com/xdCloudy/ModelSurgeon/pull/749) → [#748](https://github.com/xdCloudy/ModelSurgeon/pull/748) | `codex/acceptance-repair-arms → codex/source-teacher-distillation` |
| [#703](https://github.com/xdCloudy/ModelSurgeon/pull/703) → [#723](https://github.com/xdCloudy/ModelSurgeon/pull/723) | `codex/stream-task-quality-dataset → codex/gguf-task-quality` |

## Notable open optimizer and evidence PRs

| PR | Base | Branch | Area / relationship | Integrated on main |
| --- | --- | --- | --- | --- |
| [#693: Abstain on out-of-support Meta-Surgeon categories](https://github.com/xdCloudy/ModelSurgeon/pull/693) | `main` | `codex/abstain-meta-unknown` | meta-surgeon support boundary | No |
| [#703: Stream task-quality dataset loading](https://github.com/xdCloudy/ModelSurgeon/pull/703) | `main` | `codex/stream-task-quality-dataset` | dataset dependency for #723 | No |
| [#723: Enable task-quality gates for native GGUF](https://github.com/xdCloudy/ModelSurgeon/pull/723) | `codex/stream-task-quality-dataset` | `codex/gguf-task-quality` | stacked on #703 | No |
| [#735: Add activation-aware search baselines](https://github.com/xdCloudy/ModelSurgeon/pull/735) | `main` | `codex/competitive-activation-baseline` | pruning baseline | No |
| [#736: Add bounded Qwen3.5 MoE GGUF expert slicing](https://github.com/xdCloudy/ModelSurgeon/pull/736) | `main` | `codex/qwen35moe-native-slice` | MoE physical surgery | No |
| [#737: Plan GGUF quantization under exact byte budgets](https://github.com/xdCloudy/ModelSurgeon/pull/737) | `main` | `codex/gguf-byte-budget-planner` | GGUF quantization planning | No |
| [#738: Add sparse Qwen3.5 REAP scoring adapter](https://github.com/xdCloudy/ModelSurgeon/pull/738) | `main` | `codex/qwen35moe-reap-scorer` | MoE expert scoring | No |
| [#746: Add reproducible CUDA PyTorch setup](https://github.com/xdCloudy/ModelSurgeon/pull/746) | `main` | `codex/hf-cuda-extra` | optional CUDA path | No |
| [#748: Use immutable source model for distillation repair](https://github.com/xdCloudy/ModelSurgeon/pull/748) | `codex/acceptance-repair-arms` | `codex/source-teacher-distillation` | stacked on #749 | No |
| [#749: Reject distillation repairs with no accepted update](https://github.com/xdCloudy/ModelSurgeon/pull/749) | `main` | `codex/acceptance-repair-arms` | repair acceptance | No |
| [#764: Weight gradient features by causal token count](https://github.com/xdCloudy/ModelSurgeon/pull/764) | `main` | `codex/token-weighted-gradient-sensitivity` | Meta-Surgeon feature quality | No |
| [#765: Calculate surrogate hypervolume improvement](https://github.com/xdCloudy/ModelSurgeon/pull/765) | `main` | `codex/exact-surrogate-hypervolume` | surrogate search | No |
| [#766: Cross-fit probability calibration selection](https://github.com/xdCloudy/ModelSurgeon/pull/766) | `main` | `codex/crossfit-safety-calibration` | Meta-Surgeon calibration | No |
| [#767: Prevent ranking study evidence leakage](https://github.com/xdCloudy/ModelSurgeon/pull/767) | `main` | `codex/fix-ranking-objective-leakage` | Meta-Surgeon evaluation integrity | No |
| [#768: Abstain on numeric extrapolation](https://github.com/xdCloudy/ModelSurgeon/pull/768) | `codex/abstain-meta-unknown` | `codex/abstain-meta-numeric` | stacked on #693 | No |
| [#769: Normalize diversity feature distances](https://github.com/xdCloudy/ModelSurgeon/pull/769) | `main` | `codex/normalize-diversity-distance` | diversity acquisition base | No |
| [#770: Integrate feature diversity into acquisition](https://github.com/xdCloudy/ModelSurgeon/pull/770) | `codex/normalize-diversity-distance` | `codex/integrate-diversity-acquisition` | stacked on #769 | No |
| [#771: Compare feature diversity in active learning](https://github.com/xdCloudy/ModelSurgeon/pull/771) | `codex/integrate-diversity-acquisition` | `codex/evaluate-feature-diversity-acquisition` | stacked on #770 | No |
| [#772: Enforce integer search budgets](https://github.com/xdCloudy/ModelSurgeon/pull/772) | `main` | `codex/strict-search-budget-integers` | search budget correctness | No |
| [#773: Separate calibration and held-out splits](https://github.com/xdCloudy/ModelSurgeon/pull/773) | `main` | `codex/fix-pruning-calibration-split` | pruning evaluation integrity | No |
| [#774: Add evidence-backed project progress system](https://github.com/xdCloudy/ModelSurgeon/pull/774) | `main` | `codex/model-progress-system` | canonical research and implementation progress tracking | No |

## Triage

Prioritize correctness/leakage/quality-gate fixes (#773, #767, #766, #772) before adopting new acquisition and experimental MoE work. #774 adds the canonical progress workflow but its unmerged content is not integrated. Recheck the 56-PR backlog for duplicates and required review before merging; stacked PRs require base-to-head integration.

Re-fetch and re-audit the live backlog before later review or branch integration. Respect normal CI and approval requirements; this inventory does not authorize merging or bypass review.

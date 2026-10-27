# Open pull-request and issue backlog snapshot

GitHub CLI snapshot: **2026-10-27** · assessed `main` `589c609e1571493e2aeb5ea717a3febf0447cf83` · 0 open issues · 61 open PRs.

All 61 currently open PRs were enumerated; notable optimization, evidence, and progress-system items are retained individually. PR #780 adds a bounded KFAC physical comparison stacked on #775. This dated snapshot is not live CI state. No PR was merged as part of this audit. PR #780 head e473388 adds tested evaluation-window offsets; all four CI checks passed and the physical repeat was deferred for host RAM, so prior negative/inconclusive effectiveness assessments remain unchanged.

All open PRs were enumerated; notable optimizer/research and stacked-dependency changes are listed below. This is a dated snapshot, not live status. An open PR or green check is not integrated functionality.

## Stacked dependencies

Chains are listed in base-first order. Re-fetch branch bases and review conflicts before integration.

| Chain | Branch relationship |
| --- | --- |
| [#769](https://github.com/xdCloudy/ModelSurgeon/pull/769) → [#770](https://github.com/xdCloudy/ModelSurgeon/pull/770) → [#771](https://github.com/xdCloudy/ModelSurgeon/pull/771) | `codex/normalize-diversity-distance → codex/integrate-diversity-acquisition → codex/evaluate-feature-diversity-acquisition` |
| [#693](https://github.com/xdCloudy/ModelSurgeon/pull/693) → [#768](https://github.com/xdCloudy/ModelSurgeon/pull/768) | `codex/abstain-meta-unknown → codex/abstain-meta-numeric` |
| [#749](https://github.com/xdCloudy/ModelSurgeon/pull/749) → [#748](https://github.com/xdCloudy/ModelSurgeon/pull/748) | `codex/acceptance-repair-arms → codex/source-teacher-distillation` |
| [#703](https://github.com/xdCloudy/ModelSurgeon/pull/703) → [#723](https://github.com/xdCloudy/ModelSurgeon/pull/723) | `codex/stream-task-quality-dataset → codex/gguf-task-quality` |
| [#774](https://github.com/xdCloudy/ModelSurgeon/pull/774) → [#775](https://github.com/xdCloudy/ModelSurgeon/pull/775) | `codex/model-progress-system → codex/kfac-structured-sensitivity` |
| [#776](https://github.com/xdCloudy/ModelSurgeon/pull/776) → [#777](https://github.com/xdCloudy/ModelSurgeon/pull/777) | `codex/progress-inventory-refinement-20261027 → codex/clarify-structural-surrogate-evidence` |
| [#735](https://github.com/xdCloudy/ModelSurgeon/pull/735) → [#779](https://github.com/xdCloudy/ModelSurgeon/pull/779) | `codex/competitive-activation-baseline → codex/gradient-depth-sensitivity` |
| [#775](https://github.com/xdCloudy/ModelSurgeon/pull/775) → [#780](https://github.com/xdCloudy/ModelSurgeon/pull/780) | `codex/kfac-structured-sensitivity → codex/kfac-physical-pilot` |

## Notable open and dependency PRs

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
| [#774: Add evidence-backed project progress system](https://github.com/xdCloudy/ModelSurgeon/pull/774) | `main` | `codex/model-progress-system` | canonical research and implementation progress tracking | Yes |
| [#775: Prototype bounded KFAC structured removal costs](https://github.com/xdCloudy/ModelSurgeon/pull/775) | `codex/model-progress-system` | `codex/kfac-structured-sensitivity` | stacked on progress system PR #774; CPU-only research primitive | No |
| [#776: Expand evidence-backed research milestones](https://github.com/xdCloudy/ModelSurgeon/pull/776) | `main` | `codex/progress-inventory-refinement-20261027` | expanded progress inventory foundation | No |
| [#777: Fix structural Meta-Surgeon test-set selection](https://github.com/xdCloudy/ModelSurgeon/pull/777) | `codex/progress-inventory-refinement-20261027` | `codex/clarify-structural-surrogate-evidence` | stacked on #776; structural feature-path selector evaluation guard | No |
| [#778: Track bounded Qwen3.5 REAP outcomes](https://github.com/xdCloudy/ModelSurgeon/pull/778) | `codex/clarify-structural-surrogate-evidence` | `codex/reap-progress-evidence` | stacked on #777; records bounded CPU-only REAP experiments | No |
| [#779: Add gradient sensitivity depth-pruning baseline](https://github.com/xdCloudy/ModelSurgeon/pull/779) | `codex/competitive-activation-baseline` | `codex/gradient-depth-sensitivity` | stacked on #735; CPU-only bounded gradient depth-sensitivity comparison | No |
| [#780: Compare KFAC channel costs against pruning baselines](https://github.com/xdCloudy/ModelSurgeon/pull/780) | `codex/kfac-structured-sensitivity` | `codex/kfac-physical-pilot` | stacked on #775; CPU physical MLP channel comparison and bounded adapter; latest head adds held-out evaluation offsets, with second-window physical validation pending | No |

## Triage

Prioritize correctness/leakage/quality-gate fixes (#773, #767, #766, #772) before adoption. PR #774 merged to main at 589c609; PR #775 remains open and stacked on its former base branch codex/model-progress-system. PR #777 is stacked on open PR #776 and remains unintegrated. Preserve these review dependencies and recheck PR state before integration. PR #779 adds a bounded CPU-only gradient depth-sensitivity comparison stacked on open PR #735; its pilot is negative and is not integrated. PR #780 records one bounded CPU KFAC channel-removal comparison stacked on #775; all tested cuts miss the PPL gate, so the result does not establish quality-preserving effectiveness or integration. PR #780 head e473388 adds held-out evaluation-window offsets and focused tests; the second-window physical run remains pending sufficient RAM.

Re-fetch and re-audit the live backlog before later review or branch integration. Respect normal CI and approval requirements; this inventory does not authorize merging or bypass review.

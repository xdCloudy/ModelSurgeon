# Bounded Qwen3.5 REAP results

Assessment recorded 2026-10-27 from the reproducibility records in [PR #738 at commit `30aaac1b75a503c1f21fc4608509557ead703a58`](https://github.com/xdCloudy/ModelSurgeon/tree/30aaac1b75a503c1f21fc4608509557ead703a58). The adapter and all full run records remain in that open PR; this note bounds the progress conclusions and keeps them visible in the progress branch without treating the adapter as integrated.

## Fixed 12.5% cut

The pinned `endystrike/Endy-Qwen3.6-CyberSec-35B-A3B` revision `eed558458ca9ff6972063c486c7f1ab9e0b0c93e` (source GGUF SHA-256 `ed118e7768330eb440ddc8ade42a04005ecab37f131d7e84bb08d14d1e315ae1`) has 41 layers, 256 experts per layer and top-k 8. The run removed 32 experts per layer, retaining 224. It used 8,192 WikiText-2 raw-train calibration tokens and a separate 296,960-token, 580-chunk WikiText-2 raw-test evaluation.

| Method | Held-out PPL | PPL retention | Artifact bytes | 98% PPL gate |
| --- | ---: | ---: | ---: | --- |
| Source | 6.7090 | 100.00% | 21,713,462,496 | Pass |
| Router column norm | 8.7952 | 76.28% | 19,199,501,568 | Fail |
| Random, seed 20261026 | 7.5839 | 88.47% | 19,199,501,568 | Fail |
| Random, seed 20261027 | 7.5997 | 88.28% | 19,199,501,568 | Fail |
| Single-profile REAP | 6.9050 | 97.16% | 19,199,501,664 | **Fail** |

A paired circular moving-block bootstrap over held-out chunks found lower REAP PPL than router norm (delta -1.8902, 95% interval [-2.1239, -1.6699]) and both random seeds (deltas -0.6789 and -0.6947; both intervals below zero). This is a favorable comparison on this one corpus, but REAP still missed the declared 98% retention threshold (maximum PPL 6.8459). The bounded research outcome is therefore **negative for passing the gate at this cut**, not a claim that REAP is generally ineffective.

## Two-profile score aggregation

POE combined two disjoint 8,192-token WikiText-2 train profiles using per-layer min-max normalization and equal weights. The candidate was structurally validated and reloaded by the evaluator. It reduced artifact bytes by 11.58%.

| Candidate | Held-out PPL | Retention | Paired delta vs single profile (95% interval) | 98% gate |
| --- | ---: | ---: | ---: | --- |
| Single-profile REAP | 6.9050 | 97.16% | — | Fail |
| Two-profile ensemble | 6.9008 | 97.22% | -0.00419 [-0.03598, 0.03031] | Fail |

The interval includes zero, so this single pair provides no detectable PPL improvement from the ensemble. The bounded outcome is **inconclusive** about profile aggregation, and the candidate still fails the quality gate.

## Reproduction, provenance and limits

- [Full single-profile report](https://github.com/xdCloudy/ModelSurgeon/blob/30aaac1b75a503c1f21fc4608509557ead703a58/docs/research/qwen35moe-reap-224-wikitext-fulltest-2026-10-27.md) and [machine-readable results](https://github.com/xdCloudy/ModelSurgeon/blob/30aaac1b75a503c1f21fc4608509557ead703a58/docs/research/records/qwen35moe-reap-224-wikitext-fulltest-2026-10-27.json).
- [Paired uncertainty record](https://github.com/xdCloudy/ModelSurgeon/blob/30aaac1b75a503c1f21fc4608509557ead703a58/docs/research/records/qwen35moe-reap-224-wikitext-paired-bootstrap-2026-10-27.json).
- [Full ensemble report](https://github.com/xdCloudy/ModelSurgeon/blob/30aaac1b75a503c1f21fc4608509557ead703a58/docs/research/qwen35moe-reap-multiprofile-outcome-2026-10-27.md) and [machine-readable results](https://github.com/xdCloudy/ModelSurgeon/blob/30aaac1b75a503c1f21fc4608509557ead703a58/docs/research/records/qwen35moe-reap-multiprofile-outcome-2026-10-27.json).
- Both physical evaluations used llama.cpp CPU execution (`ngl=0`, 12 threads); neither used the user's GPU. The reports record model/dataset revisions and hashes, commands, tool revisions, seeds, artifact hashes, and limitations.
- The tested model weights are identified as AGPL-3.0 in the report. The inspected REAP source is Apache-2.0 and the POE planner is MIT. No upstream implementation code or model weights were copied into this progress change.
- Coding/reasoning and task accuracy, prefill/decode throughput, latency, peak RAM/VRAM, loading time and energy were not measured. The report noted a Windows memory-prefetch warning in the ensemble run, which nevertheless completed.
- PR #738 remains open and its implementation is experimental. These findings do not establish main-integrated support, cross-model generalization, or deployment improvement.

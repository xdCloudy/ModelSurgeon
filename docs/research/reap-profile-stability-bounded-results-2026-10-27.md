# Bounded Qwen3.5 REAP profile-stability result

This record summarizes the CPU-only leave-one-profile-out analysis in [PR #738 at commit `cf4d6fc2d304d432add1d3b138c00e02a134bb36`](https://github.com/xdCloudy/ModelSurgeon/tree/cf4d6fc2d304d432add1d3b138c00e02a134bb36). The reproducible script, tests and per-layer output are in that PR. Full profile provenance is in the [study report](https://github.com/xdCloudy/ModelSurgeon/blob/cf4d6fc2d304d432add1d3b138c00e02a134bb36/docs/research/qwen35moe-reap-profile-aggregation-stability-2026-10-27.md).

With three 8,192-token profiles from one Qwen3.5 checkpoint, per-layer mean-minmax aggregation improved average held-out bottom-set Jaccard from 0.3129 to 0.3618 and the worst held-out fold from 0.2614 to 0.2736. This is a small descriptive result from one triplet, not a significance claim. The 41 layers are not independent observations.

The result is **positive only for this bounded selection-stability metric**. It does not establish better model quality: the separate physical two-profile PPL comparison had an interval including zero, and both candidates failed the 98% PPL-retention gate. No new candidate was evaluated here. More profiles, other models, and held-out quality/deployment measurements are required before aggregation should influence optimizer decisions.

# Qwen2.5-0.5B KFAC channel comparison (CPU)

This is a bounded second architecture-cell comparison implemented in [PR #780](https://github.com/xdCloudy/ModelSurgeon/pull/780), stacked on scorer prototype [PR #775](https://github.com/xdCloudy/ModelSurgeon/pull/775). It does not establish broad Qwen support for KFAC or overall optimization effectiveness.

## Protocol

Pinned model: `Qwen/Qwen2.5-0.5B`, snapshot revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, 494,032,768 parameters, 24 layers, hidden size 896, MLP width 4,864, BF16 safetensors. Dataset: Salesforce/WikiText revision `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1`; 256 train calibration target tokens and 1,024 held-out test target tokens. The [raw report at commit `98a70fa`](https://github.com/xdCloudy/ModelSurgeon/blob/98a70facbec556d2e3191c78443cf27e7f75d565/docs/research/kfac-qwen25-05b-channel-64-2026-10-27.json) includes model, dataset, input-token, script, artifact hashes, CPU/software environment and full timings.

FP32 arithmetic was used; pruned artifacts were serialized to BF16 and reloaded for measurement. The comparison removed 64 globally shared MLP channels across all layers using additive singleton KFAC, magnitude L2, and three random seeds (11, 29, 47). CUDA was disabled. The final run used a 16 GiB total KFAC workspace cap and 1 GiB per-collector cap. Two earlier attempts stopped at explicit limits: the default 256 MiB collector cap was below the per-layer workspace requirement; with 1 GiB per collector, the 8 GiB total cap was below the planned 14,965,814,784-byte workspace.

## Results

Baseline held-out perplexity: 27.8724.

| Method | Held-out PPL | Delta | Quality gate (absolute delta <= 0.05) |
|---|---:|---:|---|
| KFAC singleton additive | 28.4082 | +0.5358 | Fail |
| Magnitude L2 | 29.6346 | +1.7622 | Fail |
| Random seed 11 | 29.5349 | +1.6626 | Fail |
| Random seed 29 | 31.5902 | +3.7178 | Fail |
| Random seed 47 | 29.0834 | +1.2110 | Fail |

KFAC ranked best among these five candidates in this cell, but it still failed the quality gate by a wide margin. The surgery removed 4,128,768 parameters (0.8357%) and reduced artifact bytes from 999,586,188 to 991,266,896 (0.8323%). KFAC scoring took 140.36 seconds; process peak RSS was 11,653,767,168 bytes (about 10.85 GiB). No inference-speed or task-quality improvement is demonstrated.

## Conclusion and limits

The result is a promising but unreplicated relative-ranking signal on one Qwen2.5-0.5B cell; it is not a quality-preserving optimization result. The held-out sample is short, KFAC is one deterministic selection, random has only three seeds, and no uncertainty estimate is available. The workspace and runtime costs are substantial for a 0.5B model. Repeat with longer independent calibration/test data and multiple selection seeds; compare activation and first-order-gradient selection; measure task quality and deployment latency before promotion.

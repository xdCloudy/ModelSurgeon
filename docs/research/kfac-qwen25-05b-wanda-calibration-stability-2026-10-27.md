# Qwen2.5-0.5B KFAC vs WANDA-style activation channel scores

This CPU-only experiment adds an independently implemented, structured channel-level adaptation of the WANDA score to the KFAC comparison in [PR #780](https://github.com/xdCloudy/ModelSurgeon/pull/780). It compares two independent calibration windows on the same pinned Qwen2.5-0.5B model and held-out test tokens.

## Upstream method and provenance

WANDA ranks individual weights by absolute weight magnitude multiplied by the L2 norm of the corresponding calibration input activation, comparing weights per output. The pinned upstream entry point is [`lib/prune.py`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py) at revision `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`; its pinned [`LICENSE`](https://github.com/locuslab/wanda/blob/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/LICENSE) is MIT. ModelSurgeon independently sums `|down_proj_weight| * activation_L2` across output rows and layers to rank globally shared MLP channels. This is a channel-level adaptation, not the original per-weight WANDA pruning procedure; no upstream code was copied.

## Protocol

Model: `Qwen/Qwen2.5-0.5B` at snapshot `060db6499f32faf8b98477b0a26969ef7d8b9987` (494,032,768 parameters; Qwen2; BF16 safetensors). Dataset: Salesforce/WikiText revision `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1`. Each calibration window contains 256 target tokens from train. Window 0 starts at token offset 0; window 1 starts at offset 1,024. Both use the same 1,024-target-token test prefix. The raw reports include model/dataset/input hashes, source commit and script hashes, artifacts, reload-logit checks, software/hardware and measurements: [window 0 report](https://github.com/xdCloudy/ModelSurgeon/blob/6003b49306db43a77c8617155d2e85304ccc7d10/docs/research/kfac-qwen25-05b-wanda-window0-2026-10-27.json), [window 1 report](https://github.com/xdCloudy/ModelSurgeon/blob/6003b49306db43a77c8617155d2e85304ccc7d10/docs/research/kfac-qwen25-05b-wanda-window1-2026-10-27.json).

Each window compares 64 globally shared MLP channels selected by KFAC singleton-additive costs, the structured WANDA-style activation score, summed magnitude L2, and three fixed random masks (seeds 11, 29, 47). Candidates are physically modified, serialized in BF16, reloaded into FP32 for evaluation, and checked for matching logits. CUDA is disabled; scoring is CPU-only with a 16 GiB total workspace cap and 1 GiB per-collector cap.

## Results

Baseline held-out PPL is 27.8724. The declared quality gate is absolute PPL delta <= 0.05.

| Selector | Window 0 delta | Window 1 delta | Quality gate |
|---|---:|---:|---|
| KFAC singleton additive | +0.5358 | +0.5100 | Fail / Fail |
| WANDA-style activation channel | +0.7958 | +1.2324 | Fail / Fail |
| Magnitude L2 | +1.7622 | +1.7622 | Fail / Fail |
| Random seed 11 | +1.6626 | +1.6626 | Fail / Fail |
| Random seed 29 | +3.7178 | +3.7178 | Fail / Fail |
| Random seed 47 | +1.2110 | +1.2110 | Fail / Fail |

KFAC ranked best in both windows and was better than the structured WANDA-style baseline by 0.2600 and 0.7223 PPL respectively. However, KFAC's 64-channel selections shared only 7 of 64 channels across calibration windows; WANDA-style selections shared 10 of 64. The outcome ranking is consistent on this fixed short test prefix, but the selected channels are not. This is a relative ranking signal, not a statistically supported or quality-preserving result.

The structural change removes 4,128,768 parameters (0.8357%) and 8,319,292 artifact bytes (0.8323%). KFAC scoring took 121.01 seconds for window 0 and 88.15 seconds for window 1; process peak RSS was 11.62 and 12.14 GB respectively. No task quality, inference-speed or memory-use improvement is demonstrated.

## Conclusion and limits

The bounded experiment gives KFAC the strongest held-out score of these selectors on both sampled Qwen2 calibration windows, but every artifact misses the declared quality gate by at least +0.51 PPL. Low channel-set overlap across windows indicates calibration sensitivity even though aggregate held-out deltas are similar. The test prefix is short and reused across both windows; results have no statistical uncertainty and cover one model, one removal budget and one metric. The WANDA baseline is a structured score adaptation rather than a reproduction of upstream unstructured pruning. Larger independent held-out data, more calibration windows, task metrics and deployment measurements are needed before claiming reliable selection or optimization effectiveness.

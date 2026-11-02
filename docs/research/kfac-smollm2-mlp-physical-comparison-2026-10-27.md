# Bounded KFAC physical MLP-channel comparison (CPU)

This assessment records the single-model physical comparison implemented in [PR #780](https://github.com/xdCloudy/ModelSurgeon/pull/780), stacked on [KFAC scorer PR #775](https://github.com/xdCloudy/ModelSurgeon/pull/775). Neither PR is integrated into `main`.

## Protocol and provenance

The model is HuggingFaceTB/SmolLM2-135M at revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` (134,515,008 parameters, Llama architecture, FP32 safetensors). The data is Salesforce/WikiText revision `b08601e04326c79dfdd32d625aee71d232d685c3`, configuration `wikitext-2-raw-v1`. Calibration uses 256 target tokens from train; evaluation uses 2,048 target tokens from test. Token-ID SHA-256 hashes, source checkpoint hashes, source commit, benchmark script hash, environment, candidate file hashes, and complete raw measurements are recorded in the [native-storage sweep report at PR commit `e9246f1`](https://github.com/xdCloudy/ModelSurgeon/blob/e9246f19d7503c85c33e6ae2e867304bf72ebf3a/docs/research/kfac-smollm2-mlp-channel-sweep-native-storage-2026-10-27.json). Arithmetic and perplexity evaluation used FP32; candidates were serialized to the source checkpoint's BF16 storage dtype before reload into FP32. CUDA was disabled; the run used CPU and two Torch threads. Earlier FP32-export pilot JSON files remain in PR #780 as an instrumentation error record and must not be used for disk-size comparisons; the native-storage report supersedes them.

At 8, 16, 32 and 64 globally shared MLP channels removed, selection compared additive singleton KFAC costs, summed magnitude L2, and three nested random masks (seeds 11, 29, 47). Each candidate was applied to a fresh checkpoint, saved, reloaded, checked for matching logits, then evaluated on the held-out token prefix. No compensation or recovery was applied.

## Results

| Channels removed | Parameter reduction | Artifact bytes (reduction from 272,445,324) | KFAC PPL (delta) | Magnitude PPL (delta) | Random PPL deltas (seeds 11 / 29 / 47) |
|---:|---:|---:|---:|---:|
| 8 | 0.3083% | 271,755,724 (0.25%) | 35.215 (+0.534) | 36.002 (+1.321) | +0.806 / +0.174 / +0.284 |
| 16 | 0.6166% | 270,926,284 (0.56%) | 35.479 (+0.798) | 36.585 (+1.904) | +1.569 / +2.844 / +0.515 |
| 32 | 1.2332% | 269,267,404 (1.17%) | 37.474 (+2.793) | 38.188 (+3.507) | +3.978 / +5.977 / +2.396 |
| 64 | 2.4665% | 265,949,636 (2.38%) | 40.757 (+6.076) | 41.597 (+6.916) | +12.203 / +10.514 / +21.670 |

Baseline held-out perplexity was 34.6812. The predeclared quality gate was absolute PPL delta <= 0.05; all 20 candidates failed. KFAC was better than magnitude at every tested cut, while a random mask was better than KFAC for some 8- and 16-channel cuts. With native BF16 serialization, 64 channels reduce exact artifact bytes from 272,445,324 to 265,949,636 (2.38%), alongside 2.4665% fewer parameters. No latency, decode throughput, task quality, or memory benefit is established.

## Conclusion and limitations

This is an inconclusive, bounded scorer comparison with a negative quality-gate outcome. It justifies further controlled comparison, not promotion as a quality-preserving optimizer. The run covers one small dense model, one short calibration prefix, one held-out prefix, and one selection per method/seed; it has no confidence intervals or cross-family replication. Additive singleton costs omit channel interactions and compensation. The benchmark scores all Llama-family layers on CPU; only this exact SmolLM2 cell has physical evidence. KFAC scoring plus surgery remain experimental and open-PR work.

The source review is separately recorded in [the pinned Qualcomm LLM-Surgeon audit](llm-surgeon-kfac-source-audit-2026-10-27.md): revision `726734425b6866782048877bac0c0d90e8cb36b3`, BSD-3-Clause-Clear. PR #780 copies no upstream source code.

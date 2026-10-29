# Bounded ShortGPT depth-pruning result: Qwen2.5-0.5B

Assessment date: 2026-10-27. This note records a bounded physical experiment from open PR [#735](https://github.com/xdCloudy/ModelSurgeon/pull/735), pinned to head `61451985128b3c72a433300f69aa5e7a471d79c6`. The result is experimental work in an unmerged PR and does not establish main-branch support.

## Protocol

- Model: `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`, weight SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
- Calibration: WikiText-2 `train.txt`, SHA-256 `6929d7c17de108ee68800c120da694fbced4a0bf6affcb8c6fb9e3a9de4ea0f6`; 2,048 tokens.
- Held-out quality: WikiText-2 `test.txt`, SHA-256 `a6f23df4148d59ae891819f5b155171aed6295bbd97a77216418155696f83c9a`; 2,032 CPU tokens and 2,016 CUDA tokens in the separate runs.
- One of 24 transformer layers was removed, reducing parameters by 14,912,384 (3.02%). The ShortGPT score was mean block influence `1 - cosine(block_input, block_output)` over 2,048 calibration tokens. Weight-magnitude and three predeclared random layer selections were compared at the same depth reduction.
- The PR includes CPU and RTX 3060 CUDA runs. The CUDA results are transcribed from the already completed PR evidence; this assessment did not run or access the GPU.

## Results

| Run | Baseline PPL | ShortGPT PPL | Delta | Magnitude PPL delta | Random PPL deltas |
|---|---:|---:|---:|---:|---|
| CPU | 28.9815 | 34.9985 | +6.0170 | +2.4173 | +5.5118, +2.3001, +2.4173 |
| RTX 3060 CUDA | 29.0411 | 35.0840 | +6.0429 | +2.4979 | +5.4955, +2.3290, +2.4979 |

Both ShortGPT candidates reloaded and produced finite logits. The CPU candidate occupied 969,698,404 bytes versus 999,586,188 source bytes. The CUDA run reported short-context prefill medians of 7,220 versus 6,602 tokens/s over three repeats, but did not measure autoregressive decode, loading, or surgery time; this is not evidence of a dependable deployment speedup. The declared quality gate was not met by any candidate.

## Conclusion and limits

For this single model, single removed layer, and short WikiText-2 evaluation, the selected ShortGPT layer had worse held-out perplexity than the matched magnitude candidate and two of three random candidates. The physical size reduction and reload succeeded, but did not preserve quality. This is a negative result for this bounded selection cell, not a general rejection of depth pruning or ShortGPT. Three random seeds are descriptive rather than powered uncertainty estimates; cross-model transfer, longer context, task quality, repeated timing robustness, decode speed, and recovery remain untested.

## Immutable evidence

- [CPU result JSON at PR head](https://github.com/xdCloudy/ModelSurgeon/blob/61451985128b3c72a433300f69aa5e7a471d79c6/docs/research/shortgpt-depth-qwen25-0.5b-2026-10-26.json)
- [CUDA result JSON at PR head](https://github.com/xdCloudy/ModelSurgeon/blob/61451985128b3c72a433300f69aa5e7a471d79c6/docs/research/shortgpt-depth-qwen25-0.5b-cuda-2026-10-26.json)
- [Benchmark implementation at PR head](https://github.com/xdCloudy/ModelSurgeon/blob/61451985128b3c72a433300f69aa5e7a471d79c6/tools/benchmark_shortgpt_depth.py)
- Independent method reference: [`sramshetty/ShortGPT`](https://github.com/sramshetty/ShortGPT), commit `78d9615fdcae6d90368832bd0a86c49c323549b9` (MIT). No upstream code was copied into ModelSurgeon by this progress update.

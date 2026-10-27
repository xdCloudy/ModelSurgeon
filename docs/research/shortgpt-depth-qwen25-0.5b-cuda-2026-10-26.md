# Qwen2.5-0.5B ShortGPT pruning on the RTX 3060

Date: 2026-10-26

Purpose: measure physical depth pruning, held-out quality, and prefill throughput on the consumer GPU targeted by ModelSurgeon.

## Result

The ShortGPT block-influence score selected layer 18. Removing it reduced the model by 14,912,384 parameters (3.02%) and the reloadable artifact directory by 29,887,784 bytes (2.99%). Reloaded logits were finite and the reloaded artifact reproduced the candidate perplexity.

| Candidate | Removed layer | Held-out PPL | Delta from source | Median prefill tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Source | — | 29.0411 | — | 6,602 |
| ShortGPT block influence | 18 | 35.0840 | +6.0429 | 7,220 |
| Weight magnitude | 11 | 31.5390 | +2.4979 | 7,086 |
| Random, seed 11 | 14 | 34.5366 | +5.4955 | 6,782 |
| Random, seed 23 | 9 | 31.3701 | +2.3290 | 6,904 |
| Random, seed 47 | 11 | 31.5390 | +2.4979 | 5,994 |

ShortGPT had the worst held-out perplexity of the five candidates. Its median prefill throughput was higher than the source in this run, but timings were collected in fixed candidate order and were not interleaved. Three repeats are too few to separate candidate effects from GPU clock and execution-order effects, so this is not evidence of a stable speedup. The benchmark covers prefill on short contexts; it does not measure autoregressive decode, load time, or peak deployment memory. No candidate passed a quality-retention gate or was promoted.

## Protocol and provenance

- Model: `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Source parameters: 494,032,768; source directory: 999,586,188 bytes; source `model.safetensors` SHA-256: `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
- Calibration: 2,048 tokens from the pinned WikiText-2 train split. Evaluation: 2,016 causal target tokens from its pinned test split, in 128-token contexts. The raw report records the exact input SHA-256 values.
- The evaluator warmed each model on one calibration chunk, then made three synchronized evaluation passes. Each pass scored the same held-out chunks; the report retains individual wall-time and throughput samples as well as their medians.
- Each chunk's summed loss is copied back to the host during scoring, and that synchronization is inside the timer. The reported tokens/s is therefore a harness-level prefill measurement, not an optimized serving benchmark.
- Environment: Windows 11, Intel Core i5-14600K, RTX 3060 12 GB, Python 3.14.7, PyTorch 2.14.1+cu130, CUDA 13.0, Transformers 5.10.1, BF16 weights, `device_map=auto`.
- Benchmark script SHA-256: `6a8758177752a95203413dfb2914ec9ed6a425a388ecfc6245849becb0fa1cc5`.
- Machine-readable report: [`shortgpt-depth-qwen25-0.5b-cuda-2026-10-26.json`](shortgpt-depth-qwen25-0.5b-cuda-2026-10-26.json), SHA-256 `16981b97cc3153e62bce40d8c13d3095e110b1d197315e97830231c4f0bc72f7`.

Reproduction uses the same local, revision-pinned model snapshot and dataset files:

```powershell
uv run --extra dev --extra hf-cuda python tools/benchmark_shortgpt_depth.py `
  --model <local-Qwen2.5-0.5B-snapshot> `
  --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 `
  --calibration-text <wikitext-train.txt> `
  --evaluation-text <wikitext-test.txt> `
  --output-directory <new-output-directory> `
  --calibration-tokens 2048 --evaluation-tokens 2032 `
  --sequence-length 128 --cpu-threads 4 `
  --random-seeds 11 23 47 --device-map auto --measurement-repeats 3
```

This one-model, one-removal study is a negative quality result for ShortGPT on this cell. It does not establish cross-model ranking quality, task-accuracy preservation, or deployment speed improvement.

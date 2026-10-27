# Qwen3.5 REAP calibration-split stability

Date: 2026-10-27
Status: ranking-stability experiment only; no model was modified or evaluated.

## Result

I profiled the same pinned Qwen3.5-family GGUF with 8,192 calibration tokens from WikiText-2 train and, independently, 8,192 tokens from WikiText-2 validation. At the same 12.5% per-layer cut (32 of 256 experts), REAP's selected prune sets had **0.257 mean per-layer Jaccard** (minimum 0.103, maximum 1.000). The mean per-layer Spearman correlation over all REAP scores was 0.634. The hottest 12.5% router-frequency sets had 0.507 mean Jaccard across the same profiles.

The train-profile REAP bottom-set stability between its final two 128-token windows was 0.978, and the validation profile's was 0.978; their final rank correlations were 0.998 in each run. Thus, high within-run convergence did not imply cross-split prune-set stability. The result is evidence of sensitivity to calibration distribution, not proof that REAP is intrinsically unstable: train and validation differ in domain, and this is only one checkpoint and one split pair.

| Metric | Train vs. validation |
| --- | ---: |
| Tokens per profile | 8,192 each |
| REAP score mean Spearman | 0.633525 |
| REAP bottom-32 prune-set mean Jaccard | 0.257138 |
| REAP bottom-32 prune-set min / max Jaccard | 0.103448 / 1.000000 |
| Layers with no REAP prune-set overlap | 0 / 41 |
| Router-frequency top-32 mean Jaccard | 0.506755 |

## Reproduction and provenance

- Model: `endystrike/Endy-Qwen3.6-CyberSec-35B-A3B`, revision `eed558458ca9ff6972063c486c7f1ab9e0b0c93e`, GGUF SHA-256 `ed118e7768330eb440ddc8ade42a04005ecab37f131d7e84bb08d14d1e315ae1`, fingerprint `poe1:a0952c3c50425919`; architecture `qwen35moe`, 41 layers, 256 experts/layer, top-k 8. The checkpoint declares AGPL-3.0; no weights are included.
- Calibration datasets: WikiText-2 raw train text, 10,962,377 bytes, SHA-256 `fb23ad9643a34514eec5cb85ec2a6f49d1a33e6a3d5077dff5a403e1d18f5047`; and WikiText-2 raw validation, 1,146,846 bytes, SHA-256 `4cd0f6876d07a413aa911261ff6d363c72d757d47f0fdd6015702014c89cb9c7`. The text is not redistributed.
- Profiler and plan/comparison source: POE commit `3908873326bcdbe097f4184729db7c33fff1cf60`, MIT; profiler linked against llama.cpp `c7fc5094179bf8a9c73c18593db714f8da41a402`, CPU, `GGML_CPU_REPACK=OFF`. Both profiles used `--metric reap --max-tokens 8192 --batch 128 --ngl 0 --threads 12`. The train run took 545 seconds; the validation run took 596 seconds. Each captured 12,800 MoE tensors with zero invalid expert IDs and zero skipped REAP captures.
- Profile SHA-256: train `4a068990bcaf4a4dd3e9bb75d650cb1d3e0de56c7dfd955630f1ea7667b78a3b`; validation `105dcf436c40455474389b732bc06c518a72b907582894c7f21aa3b6ba2ab04e`. Aggregate POE profiles and losslessly compressed profiler logs are retained in `records/`; the profiles contain per-expert aggregate scores/counts, not weights or token text. Decompressed log SHA-256 values: train `67fbe83541c7c038ca47679384f1b36a5ec5c85c04e4925463f3b63f8cfbb1e9`; validation `6226ad0b8fedc242723bff9bfb46f7bdb8a45a1a621c99f08600d5e3eda684e1`.
- [`tools/compare_reap_profiles.py`](../tools/compare_reap_profiles.py) independently implements the pinned POE `profile.c` comparison semantics for average-tie Spearman and deterministic expert-ID tie breaking. It reads both profile files and the corresponding local dataset files, verifies the profile byte counts, and emits the machine-readable comparison [`records/qwen35moe-reap-calibration-stability-2026-10-27.json`](records/qwen35moe-reap-calibration-stability-2026-10-27.json). No upstream code was copied.

No validation perplexity was computed from the validation-calibrated profile, so this report does not reuse calibration text as quality evidence. No candidate artifact, task quality, inference speed, memory use, or calibration-subset variance was measured.

## Decision

Keep the Qwen REAP adapter experimental. Its strong result against the earlier random/router-norm PPL point estimates is tempered by a low cross-split prune-set overlap, and that earlier 12.5% artifact failed the 98% quality-retention gate. The next model-level test should compare independently sampled train calibration subsets and a predeclared validation/test suite, with the first-party ModelSurgeon scorer and same-runtime candidate writer included. Measure held-out quality and deployment costs before any optimizer integration.

# Mixed-codec GGUF deployment throughput check

This order-balanced llama.cpp run checks whether byte-budget candidates that
were physically written and reloaded change prompt or decode throughput. It
extends the earlier three-artifact comparison with the mixed Q2_K/Q3_K/Q4_K
activation-weighted candidate. Results apply to this model and runtime only.

## Protocol

- Source: SmolLM2-135M Q4_K_M, SHA-256
  `ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747`.
- Candidates: weight-MSE Q3_K, activation-weighted Q3_K, and
  activation-weighted mixed Q2_K/Q3_K/Q4_K. Artifact hashes and exact byte
  counts are in the machine-readable record.
- Runtime: llama.cpp `llama-bench` 0.7.0-dev, build 11804, commit
  `c7fc50941`, on Intel Core i5-14600K / RTX 3060 12 GB / 64 GB RAM.
- Workload: 512-token prompt, 128 generated tokens, batch and microbatch 512,
  8 threads. GPU mode offloaded 99 layers; CPU mode offloaded zero.
- All 24 artifact order permutations ran once per mode, with a fixed-seed
  shuffle. Every artifact appeared in each position six times. Each process
  used the normal warmup and one measured repetition, yielding 24 samples per
  artifact per mode.
- Changes are paired against the source within each permutation. Intervals are
  descriptive paired Student t 95% intervals (24 pairs, 23 degrees of freedom).

## Results

Percent change in throughput against the source; all intervals include zero.

| Mode | Candidate | Prompt tokens/s change | Generation tokens/s change |
| --- | --- | ---: | ---: |
| CPU, 8 threads | Weight-MSE Q3_K | +0.33% [−5.99%, +6.64%] | +11.68% [−3.70%, +27.05%] |
| CPU, 8 threads | Activation-weighted Q3_K | −1.78% [−7.68%, +4.12%] | +8.90% [−4.34%, +22.14%] |
| CPU, 8 threads | Activation-weighted mixed Q2_K/Q3_K/Q4_K | −1.85% [−8.02%, +4.32%] | +5.11% [−10.31%, +20.53%] |
| RTX 3060, 99 layers | Weight-MSE Q3_K | +1.82% [−2.30%, +5.93%] | −0.94% [−2.78%, +0.91%] |
| RTX 3060, 99 layers | Activation-weighted Q3_K | +0.86% [−3.31%, +5.03%] | +0.28% [−1.07%, +1.64%] |
| RTX 3060, 99 layers | Activation-weighted mixed Q2_K/Q3_K/Q4_K | +1.05% [−2.07%, +4.18%] | +0.48% [−0.69%, +1.65%] |

The mixed candidate shows no reliable prompt or decode improvement in this
small-model run. The CPU generation point estimate is positive, but its
interval is wide and crosses zero. The GPU results are close to zero. This
does not support a speedup claim or a hardware-aware codec-selection policy.

## Interpretation and limits

Each sample is one measured repetition after llama-bench warmup, so intervals
reflect paired run-to-run variation across balanced model orders rather than
within-process repeated timing. This is a single 135M model, one llama.cpp
build, and one workstation; it does not characterize larger-model memory
bandwidth, load time, time to first token, peak RAM/VRAM, or other runtimes.
The planner's quality gate remains authoritative, and reduced file size is not
treated as evidence of faster inference.

The full 192-run data set, commands, artifact hashes, output rows, and paired
summaries are retained in
[`gguf-mixed-codec-throughput-2026-10-25.json`](gguf-mixed-codec-throughput-2026-10-25.json)
(SHA-256 `2b92a3dcda09d48980b62b3f469256023b3ef232e67dcfe0a4a77eb32d8e9f71`).

# Qwen3.6-derived MoE GGUF deployment throughput

This follow-up measures the source and exact router-norm 252-expert artifact from the [full-split quality comparison](qwen35moe-gguf-expert-slicing-fulltest-router-norm-2026-10-26.json). The candidate was regenerated with PR #736's current surgery code and its SHA-256 matched that quality record before benchmarking.

## Protocol

- Model: `Endy-Qwen3.6-CyberSec-35B-A3B-Q4_K_M`, source revision `eed558458ca9ff6972063c486c7f1ab9e0b0c93e`.
- Source SHA-256: `ed118e7768330eb440ddc8ade42a04005ecab37f131d7e84bb08d14d1e315ae1`.
- Router-norm candidate SHA-256: `06969df1ce0505ad2f7d506497c9f5f9731c3a3887d07351ccafbd3edd7d3fe3`.
- Hardware: Intel Core i5-14600K, RTX 3060 12 GB, 64 GB RAM, Windows 11.
- Runtime: llama.cpp 0.7.0-dev, build 11804, commit `c7fc50941`, CUDA backend.
- Each invocation used `llama-bench -p 512 -n 128 -b 512 -ub 512 -t 8 -ngl 8 -r 1 -o json`, leaving the normal warmup enabled. Eight of the model's 41 layers were GPU-offloaded; this is a mixed CPU/GPU profile because the full model does not fit in 12 GB VRAM.
- Twelve paired rounds were run, with six source-first and six candidate-first pairs in the deterministic shuffled order recorded in the JSON. Each invocation recorded 512-token prompt processing and 128-token generation.
- The reported percent change is the paired candidate/source throughput ratio minus one. The interval is a descriptive 95% paired t interval over 12 rounds (11 degrees of freedom, `t*=2.201`).

## Results

| Metric | Source mean | Candidate mean | Candidate change vs. source (95% paired t interval) |
| --- | ---: | ---: | ---: |
| 512-token prompt processing | 289.85 tok/s | 293.91 tok/s | +1.50% [−1.35%, +4.35%] |
| 128-token generation | 18.018 tok/s | 18.032 tok/s | +0.16% [−2.05%, +2.38%] |

Both intervals include zero. This run does not establish an inference speed improvement. The file reduction remains 314,245,088 bytes (1.45%) for this artifact; storage reduction alone is not evidence of faster inference.

The preceding CPU full-test report measured source PPL 6.7090 and candidate PPL 6.8094 on 296,960 WikiText-2 targets. Its propagated uncertainty is not a formal confidence interval. This throughput follow-up does not remeasure quality, apply a promotion gate, or establish task-quality retention.

## Limits and raw evidence

This is one artifact pair, one workstation, and one partial-GPU-offload configuration. llama.cpp's benchmark workloads are synthetic. The study did not measure TTFT, model loading time, peak RAM/VRAM, energy, or additional serving contexts. The intervals describe these paired runs only and do not support generalization to other hardware or models.

The machine-readable record includes all 12 source/candidate pairs and their sample rates: [raw paired results](qwen35moe-gguf-expert-slicing-gpu-throughput-2026-10-26.json), SHA-256 `8bfa7bf4df9dbfce93f64b06bbd658122578079cde997c390eecf41f5990e6bb`.

# GGUF byte-budget throughput comparison

This follow-up measures deployment throughput for the source, weight-MSE, and
activation-weighted artifacts from the
[activation-weighted allocation experiment](gguf-activation-weighted-allocation-2026-10-25.md).
The 135M Llama artifact is small; conclusions are limited to this model and
llama.cpp build.

## Protocol

The source was `unsloth/SmolLM2-135M-Instruct-GGUF` revision
`9e6855bc4be717fca1ef21360a1db4b29d5c559a` (SHA-256
`ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747`). The
weight-MSE candidate SHA-256 is
`ab390de77e83d183c2fe725578e08e8420fab67f394133bd22752018a4a10d8f`; the
activation-weighted candidate SHA-256 is
`905a45028be5b2f99d7a297952ae28fe6bd68f368057bc1955e5c4e53664be24`.

The benchmark used llama.cpp `llama-bench` 0.7.0-dev, build 11804, commit
`c7fc50941`, on an Intel Core i5-14600K / RTX 3060 12 GB / 64 GB RAM system.
For each deployment mode, all six orderings of the three artifacts were run
twice in a fixed-seed shuffle, giving 12 matched rounds and 12 measured
throughput samples per artifact. Each `llama-bench` process used its normal
warmup and one measured repetition. Prompt processing used 512 tokens;
generation used 128 tokens; batch/microbatch were 512 and CPU thread count was
8. CPU mode used `--n-gpu-layers 0`; GPU mode used `--n-gpu-layers 99`.

This ordering design was necessary: an earlier grouped run showed that the
first artifact measured in a process could be markedly faster, so its grouped
means confounded artifact order with throughput. Those pilot values are not
used below. The final comparison pairs each candidate's rate with the source
rate from the same balanced round. The approximate 95% intervals below use a
paired Student t interval over 12 ratios (11 degrees of freedom, `t*=2.201`).

## Results

Percentage throughput change against the source; positive values mean higher
tokens per second. The intervals include zero in every case.

| Mode | Candidate | 512-token prompt processing | 128-token generation |
| --- | --- | ---: | ---: |
| CPU, 8 threads | Weight-MSE | −1.64% [−8.52%, +5.24%] | −4.54% [−21.28%, +12.19%] |
| CPU, 8 threads | Activation-weighted | −5.06% [−11.23%, +1.12%] | −0.97% [−23.70%, +21.77%] |
| RTX 3060, 99 layers | Weight-MSE | +2.29% [−0.41%, +5.00%] | −1.16% [−3.54%, +1.21%] |
| RTX 3060, 99 layers | Activation-weighted | +3.12% [−0.45%, +6.70%] | −0.01% [−1.80%, +1.78%] |

The point estimates and intervals do not establish a reliable throughput
improvement or regression for either candidate. The GPU prompt point estimates
are slightly positive; the confidence intervals still include no change. GPU
decode rates were steady in the order-balanced design, and both candidate
intervals overlap zero. No time-to-first-token, model-load latency, RAM, or VRAM
comparison is claimed. The reported llama.cpp guessed dominant codec name is
not the complete mixed tensor-type description.

The machine-readable record contains all 72 order-balanced runs, their prompt
and generation samples, per-artifact means and standard deviations, model
hashes, configuration, ordering, and paired comparisons:
[`gguf-byte-budget-throughput-2026-10-25.json`](gguf-byte-budget-throughput-2026-10-25.json)
(SHA-256 `45e587b025a1e57b7e2e8730642288b1a944fc09378f31e0281b2a357e38ea09`).

To reproduce a single measured process, use the pinned llama.cpp build and
replace `<model.gguf>` with one of the three hash-identified artifacts:

```text
llama-bench --model <model.gguf> --n-prompt 512 --n-gen 128 \
  --batch-size 512 --ubatch-size 512 --threads 8 \
  --n-gpu-layers <0-for-CPU-or-99-for-GPU> --repetitions 1 --output json
```

Leave warmup enabled. Repeat under the recorded six-order schedule twice; do
not compare candidates measured only before or after the source.

## Decision

The byte-budget candidates retain the measured file-size reduction and the
activation-weighted candidate has a smaller WikiText perplexity penalty, but
this benchmark resolves no deployment throughput benefit. Quantization
allocation should therefore remain quality-gated and should not treat storage
reduction as a latency prediction. Larger models, longer generation runs, and
additional CPU/GPU systems are needed to characterize mixed-codec kernel
behavior.

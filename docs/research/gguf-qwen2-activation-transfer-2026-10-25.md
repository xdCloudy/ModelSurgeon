# Qwen2 activation-weighted GGUF transfer experiment

## Question and decision

Does the diagonal activation-weighted codec-allocation objective transfer from
the earlier small Llama experiment to a dense Qwen2 model? It does not show an
improvement on this Qwen2 cell. At a 5% file-byte target, the activation
weighted candidate had the highest held-out perplexity of the three tested
allocations. The reported chunk intervals overlap, so this small run does not
establish that the differences are statistically reliable. No candidate passes
a quality gate, and none is promoted.

## Provenance and protocol

- Source: [Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF),
  repository revision
  [`9217f5db79a29953eb74d5343926648285ec7e67`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/tree/9217f5db79a29953eb74d5343926648285ec7e67),
  `qwen2.5-0.5b-instruct-q4_k_m.gguf`, 491,400,032 bytes, SHA-256
  `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
  The model repository declares Apache-2.0. Its matching Transformers checkpoint
  is [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
  at revision `7ae557604adf67be50417f59c2c2f167def9a775`, also declared
  Apache-2.0.
- Calibration: eight 512-token contexts from the raw WikiText-2 train split
  (4,096 tokens total), tokenized with the pinned Qwen checkpoint. Train text
  SHA-256: `6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57`.
  The source-bound activation manifest contains 168 projection tensors.
- Evaluation: raw WikiText-2 test text, one held-out file evaluated as eight
  llama.cpp chunks, context/batch/microbatch 512. Test text SHA-256:
  `a1ac9f35d8d3014f89b8d5cb4cfd5e55516c0126b3a8de75447b7d3084fc5011`.
- Runtime: `llama-perplexity` from llama.cpp commit `c7fc50941`, CPU-only,
  eight threads, no warmup. Host: Intel Core i5-14600K, RTX 3060 12 GB, 64 GB
  RAM. The GPU was not used for collection or evaluation.
- Allocation: exact candidate file ceiling 466,830,030 bytes (95% of source),
  Q2_K/Q3_K/Q4_K alternatives, 16 MiB planner working-memory budget. The
  activation plan scored 24 eligible tensors, changed 19, and retained source
  codecs for unsupported layouts. It wrote a 466,749,312-byte GGUF (5.015%
  smaller), SHA-256
  `e05d8aa78e1d480fa65351eebceefda44d34d52a91add59d933c4e57ee6381b1`.
- The collector verified dense `qwen2` configuration, tensor geometry,
  tokenizer tokens and merges, and source hash. It rejects `qwen2_moe`; this
  evidence does not support MoE. Transformers dequantizes the full checkpoint
  to float32 for collection (2,520,669,696 dense weight bytes); the configured
  3 GiB weight limit is not a peak-RAM guarantee.
- The retained local activation-moment manifest has SHA-256
  `0d6ec6a1e7ff7fd91c2a561a8a573e8e5d812af40b685976794cea5ba6c91821`; the
  local planner/evaluation JSON has SHA-256
  `932de665347c6d81a7f233a934f42c689b143d42112d5c6f02b27b0bf8d97943`.

## Results

| Allocation | Candidate bytes | Reduction | Changed tensors | Test PPL ± reported chunk uncertainty | Delta vs source |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source Q4_K_M | 491,400,032 | — | — | 16.1400 ± 1.06307 | — |
| Weight-MSE, Q2_K/Q3_K/Q4_K | 466,613,120 | 5.044% | 20 | 17.7862 ± 1.16915 | +1.6462 |
| Weight-MSE, Q3_K only | 466,340,736 | 5.100% | 20 | 17.7242 ± 1.16568 | +1.5842 |
| Activation-weighted, Q2_K/Q3_K/Q4_K | 466,749,312 | 5.015% | 19 | 17.9591 ± 1.18121 | +1.8191 |

All source and candidate files loaded successfully in the same pinned
llama.cpp evaluator. The activation plan's held-out PPL was numerically worse
than both weight-MSE candidates at a similar byte budget. The intervals overlap
and this is one corpus, one model, one calibration sample set, and one runtime
configuration; do not interpret the ranking as a statistically established
regression or as evidence for a general method ranking. All three candidates
show a material PPL increase over the source, and quality retention remains
unproven.

## Acquisition matrix and next step

| Field | Assessment |
| --- | --- |
| Source component | WANDA's activation-aware importance principle, MIT, upstream commit [`8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`](https://github.com/locuslab/wanda/tree/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py); Transformers GGUF loader, Apache-2.0. No WANDA source was copied. |
| Current equivalent | ModelSurgeon computes a diagonal second-moment weighted reconstruction score and uses bounded GGUF allocation; prior physical calibration evidence covered a Llama model. |
| Integration | **REIMPLEMENT** the published mathematical principle; **INTEGRATE** the optional Transformers loader behind the research collector. Dense Qwen2 is now verified; MoE is excluded. |
| Benefit hypothesis | Activation-weighted allocation may improve held-out quality at equal file bytes. This Qwen2 result does not support that hypothesis; the activation candidate was numerically worse than weight-MSE alternatives. |
| Hardware and cost | CPU calibration dequantizes 2.52 GB of float32 weights; 16 MiB allocation working-set bound; CPU-only evaluation. Collector duration was about six minutes and plan/write about 431 seconds. Peak RAM was not measured reliably. |
| Risks and validation | Diagonal covariance omits feature correlations and downstream error accumulation. Eight calibration contexts and eight evaluation chunks are insufficient for a quality claim. Transformers/loader version was not captured in the moment manifest. |
| Priority | Keep as an experimental objective. Collect repeat runs with independent calibration text, multiple dense Qwen2 sizes, and more held-out data before using it to guide autonomous decisions. |

The exact candidate and JSON report were retained locally, outside this source
repository; model and dataset files are not redistributed here. The candidate
and report hashes above bind the key output to that local evaluation. The
earlier Llama result and its protocol are recorded in
[`gguf-activation-weighted-allocation-2026-10-25.md`](gguf-activation-weighted-allocation-2026-10-25.md).

## Reproduction commands

With the pinned model and local WikiText-2 files available, collect moments and
produce the same allocation/evaluation report with the optional HF extra:

```powershell
uv run --frozen --extra hf python tools/collect_gguf_activation_moments.py `
  --source qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775 `
  --calibration-text wiki.train.raw.txt `
  --dataset Salesforce/wikitext:wikitext-2-raw-v1 `
  --dataset-revision sha256:6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57 `
  --split train --context-size 512 --chunks 8 --threads 1 `
  --max-dense-weight-bytes 3221225472 --output activation-moments.json

uv run --frozen python tools/benchmark_gguf_quantization_budget.py `
  --source qwen2.5-0.5b-instruct-q4_k_m.gguf `
  --candidate activation-candidate.gguf `
  --target-file-bytes 466830030 `
  --candidate-types Q2_K Q3_K Q4_K `
  --max-working-bytes 16777216 `
  --activation-moments activation-moments.json --report activation-report.json `
  --evaluation-text wiki.test.raw.txt `
  --dataset Salesforce/wikitext:wikitext-2-raw-v1 `
  --dataset-revision sha256:a1ac9f35d8d3014f89b8d5cb4cfd5e55516c0126b3a8de75447b7d3084fc5011 `
  --split test `
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct `
  --tokenizer-revision 7ae557604adf67be50417f59c2c2f167def9a775 `
  --llama-perplexity llama-perplexity.exe --llama-revision c7fc50941 `
  --context-size 512 --threads 8 --chunks 8 --timeout-seconds 900
```

The local report retains evaluator commands, outputs, return codes, hashes and
reported uncertainties. The Transformers version used for this collection
was not captured in the manifest; pin it explicitly for a bit-for-bit software
environment replay.

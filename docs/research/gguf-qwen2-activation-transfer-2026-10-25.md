# Qwen2 activation-weighted GGUF transfer experiment

## Question and decision

Does the diagonal activation-weighted codec-allocation objective transfer from
the earlier small Llama experiment to a dense Qwen2 model? It does not show an
improvement on this Qwen2 cell. At a 5% file-byte target, the activation
weighted candidate had the highest held-out perplexity of the three tested
allocations. An initial eight-window comparison was extended to 64 windows and
then to every full-size context window in WikiText-2 test. All three compressed
artifacts have higher PPL than the source on the full test corpus. Activation
weighting is also numerically worse than both weight-MSE plans on that corpus.
llama.cpp's reported `+/-` uncertainty is propagated from a Gaussian assumption
on correct logits, not a formal confidence interval. These values therefore
describe the measured runs; they do not establish statistical significance or
general method ranking. No candidate passes a quality gate, and none is
promoted.

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
- Evaluation: raw WikiText-2 test text, first evaluated as eight and then as 64
  chunks; a final run evaluated all 508 available 512-token llama.cpp chunks
  (260,096 tokens). The pinned Transformers tokenizer counted 260,573 tokens,
  leaving 477 tokens outside the full-size chunk count. Context/batch/
  microbatch were 512. Test text SHA-256:
  `a1ac9f35d8d3014f89b8d5cb4cfd5e55516c0126b3a8de75447b7d3084fc5011`.
- Runtime: `llama-perplexity` from llama.cpp commit `c7fc50941`, CPU-only,
  eight threads, no warmup. Host: Intel Core i5-14600K, RTX 3060 12 GB, 64 GB
  RAM. The GPU was not used for collection or evaluation.
- The `+/-` values below are llama.cpp's Gaussian-logit error-propagation
  uncertainty, as documented at the exact runtime revision in its
  [perplexity README](https://github.com/ggml-org/llama.cpp/blob/c7fc50941/tools/perplexity/README.md).
  They are not formal confidence intervals, and no p-values or significance
  tests were computed.
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
llama.cpp evaluator. The eight-window and 64-window `+/-` bands overlap between
the candidate allocations, so those shorter runs do not establish method
ranking. The 64-window run evaluates the first 64 contexts, not a random sample
of the full test corpus.

### Extended held-out evaluation

The same four unchanged artifacts were evaluated on the first 64 WikiText-2
test windows (32,768 tokens) with the same CPU llama.cpp build and settings.
The source PPL differed from the full-corpus result, confirming that this
prefix is not a substitute for evaluating the whole test file. Candidate `+/-`
bands overlapped one another in this 64-window run.

| Artifact | Bytes | Reduction | Test PPL ± reported 64-window uncertainty | Delta vs source |
| --- | ---: | ---: | ---: | ---: |
| Source Q4_K_M | 491,400,032 | — | 15.6328 ± 0.35296 | — |
| Weight-MSE, Q2_K/Q3_K/Q4_K | 466,613,120 | 5.044% | 16.9625 ± 0.38405 | +1.3297 |
| Weight-MSE, Q3_K only | 466,340,736 | 5.100% | 16.9100 ± 0.38301 | +1.2772 |
| Activation-weighted, Q2_K/Q3_K/Q4_K | 466,749,312 | 5.015% | 17.2917 ± 0.39335 | +1.6589 |

All four 64-window runs returned successfully, and the test text digest was
rechecked. Full command-log hashes and per-artifact identities are in
[`gguf-qwen2-activation-transfer-64chunk-2026-10-25.json`](gguf-qwen2-activation-transfer-64chunk-2026-10-25.json).

### Full WikiText-2 test evaluation

The final comparison used `--chunks -1` (all available chunks), which processed
508 complete 512-token contexts for each artifact. This is 260,096 of the
260,573 tokens counted by the pinned Transformers tokenizer; the final 477
tokens do not form another complete context. Each command returned zero and
reported 508 progress rows.

| Artifact | Bytes | Reduction | Full-test PPL ± llama.cpp reported uncertainty | Delta vs source |
| --- | ---: | ---: | ---: | ---: |
| Source Q4_K_M | 491,400,032 | — | 16.1229 ± 0.13134 | — |
| Weight-MSE, Q2_K/Q3_K/Q4_K | 466,613,120 | 5.044% | 17.4498 ± 0.14221 | +1.3269 |
| Weight-MSE, Q3_K only | 466,340,736 | 5.100% | 17.3874 ± 0.14194 | +1.2645 |
| Activation-weighted, Q2_K/Q3_K/Q4_K | 466,749,312 | 5.015% | 17.7475 ± 0.14537 | +1.6246 |

The source and each candidate have disjoint reported `+/-` bands. The
activation-weighted candidate also has disjoint bands from both weight-MSE
candidates, although the activation vs. mixed weight-MSE gap is small. Since
these bands are not confidence intervals and this is still one model, one
corpus, one runtime configuration, and no independent-seed repeat, interpret
that as a full-corpus measured ranking rather than formal inference. All three
artifacts incur a clear PPL increase and fail the quality-retention objective.
No task accuracy or throughput benefit was established. The immutable
full-test manifest with model, dataset, evaluator, and log digests is in
[`gguf-qwen2-activation-transfer-fulltest-2026-10-25.json`](gguf-qwen2-activation-transfer-fulltest-2026-10-25.json).

## Acquisition matrix and next step

| Field | Assessment |
| --- | --- |
| Source component | WANDA's activation-aware importance principle, MIT, upstream commit [`8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`](https://github.com/locuslab/wanda/tree/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py); Transformers GGUF loader, Apache-2.0. No WANDA source was copied. |
| Current equivalent | ModelSurgeon computes a diagonal second-moment weighted reconstruction score and uses bounded GGUF allocation; prior physical calibration evidence covered a Llama model. |
| Integration | **REIMPLEMENT** the published mathematical principle; **INTEGRATE** the optional Transformers loader behind the research collector. Dense Qwen2 is now verified; MoE is excluded. |
| Benefit hypothesis | Activation-weighted allocation may improve held-out quality at equal file bytes. This Qwen2 result does not support that hypothesis; the activation candidate was numerically worse than weight-MSE alternatives. |
| Hardware and cost | CPU calibration dequantizes 2.52 GB of float32 weights; 16 MiB allocation working-set bound; CPU-only evaluation. Collector duration was about six minutes and plan/write about 431 seconds. Peak RAM was not measured reliably. |
| Risks and validation | Diagonal covariance omits feature correlations and downstream error accumulation. Calibration used eight contexts; the full held-out test was evaluated once on one corpus with no independent seeds. llama.cpp's reported uncertainty is not a formal confidence interval. Transformers/loader version was not captured in the moment manifest. |
| Priority | Keep as an experimental objective. Collect repeat runs with independent calibration text, multiple dense Qwen2 sizes, and more held-out data before using it to guide autonomous decisions. |

The candidate binaries and tensor-moment manifest (about 7 MB) remain in the external evaluation directory. The compact full-test JSON summary is checked in here; larger local command logs remain outside the source repository. Model and dataset files are not redistributed. The
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
  --context-size 512 --threads 8 --chunks 64 --timeout-seconds 900
```

For the full-test follow-up, evaluate the four unchanged artifacts with the
pinned `llama-perplexity` executable using the same flags as above and
`--chunks -1` (all available chunks). The exact argument template, per-file
hashes, return codes, progress counts, and raw log digests are recorded in the
full-test JSON manifest.

The local report retains evaluator commands, outputs, return codes, hashes and
reported uncertainties. The Transformers version used for this collection
was not captured in the manifest; pin it explicitly for a bit-for-bit software
environment replay.

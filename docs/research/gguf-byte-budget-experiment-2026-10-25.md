# GGUF byte-budget physical smoke experiment

## Result

The exact-byte planner and transactional writer produced a smaller, loadable
GGUF, but its measured perplexity increased. This does not support promoting
the candidate or claiming quality retention.

| Measurement | Source | Candidate |
| --- | ---: | ---: |
| File bytes | 105,454,144 | 100,145,760 |
| File reduction | — | 5.034% |
| WikiText-2 raw test PPL | 19.8433 ± 1.28446 | 24.7233 ± 1.59424 |

The measured delta was **+4.8800 PPL** over eight llama.cpp chunks. The
candidate changed 18 `ffn_down` tensors to Q3_K; other tensors were preserved.
Both source and candidate completed the same llama.cpp run successfully. The
PPL intervals do not overlap, but this short run is a smoke experiment, not a
publication-quality evaluation. No throughput, latency, peak memory, or task
accuracy comparison was made.

## Inputs and provenance

- Model: [`unsloth/SmolLM2-135M-Instruct-GGUF`](https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF/tree/9e6855bc4be717fca1ef21360a1db4b29d5c559a), revision
  `9e6855bc4be717fca1ef21360a1db4b29d5c559a`, file
  `SmolLM2-135M-Instruct-Q4_K_M.gguf`; SHA-256
  `ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747`.
  The model repository declares Apache-2.0.
- Evaluation text: [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-2-raw-v1`, `test` split;
  local text SHA-256
  `a1ac9f35d8d3014f89b8d5cb4cfd5e55516c0126b3a8de75447b7d3084fc5011`.
  The Hub dataset card lists CC-BY-SA-3.0 and GFDL. The text is not included
  in this repository.
- Tokenizer metadata: embedded in the source GGUF; digest
  `b582add311bdcbe1053aa488a7c36274a2f066b2e108345a2bd45c378c6621f4`.
- Runtime: llama.cpp `c7fc50941`, Windows x86_64, CPU-only, 8 threads,
  context/batch/microbatch 512, 8 chunks, no warmup.
- Host CPU: Intel Core i5-14600K, 14 cores / 20 logical processors. The GPU
  was not used.
- Candidate SHA-256:
  `ab390de77e83d183c2fe725578e08e8420fab67f394133bd22752018a4a10d8f`.

The candidate is an experimental artifact outside this repository. The source
GGUF remained unchanged. The planner's objective was the sum of codec-local
weight reconstruction MSE under a 100,181,436-byte ceiling (95% of source
size); that proxy did not protect perplexity here.

## Reproduction

With the model, test text, and llama.cpp build above available locally, run:

```powershell
uv run --frozen --extra dev python tools/benchmark_gguf_quantization_budget.py `
  --source source.gguf `
  --candidate candidate.gguf `
  --target-file-bytes 100181436 `
  --target-type Q3_K `
  --max-working-bytes 16777216 `
  --report report.json `
  --evaluation-text wiki.test.raw.txt `
  --dataset Salesforce/wikitext:wikitext-2-raw-v1 `
  --dataset-revision sha256:a1ac9f35d8d3014f89b8d5cb4cfd5e55516c0126b3a8de75447b7d3084fc5011 `
  --split test `
  --tokenizer unsloth/SmolLM2-135M-Instruct-GGUF `
  --tokenizer-revision 9e6855bc4be717fca1ef21360a1db4b29d5c559a `
  --llama-perplexity llama-perplexity.exe `
  --llama-revision c7fc50941 `
  --context-size 512 --threads 8 --chunks 8
```

The dataset revision value above is a content identifier for the exact local
test text, not a Hub commit identifier. The run used one test split and one
runtime build, so the evidence does not establish generalization, statistical
significance, or superiority over llama.cpp's standard quantization recipes.

# Activation-weighted GGUF codec allocation experiment

## Question

Does a calibration-derived activation-weighted tensor reconstruction objective
choose a better mixed-codec GGUF than weight-only MSE at the same physical file
budget? This experiment evaluates quantization allocation only. It does not
prune model structure, repair quality, or claim that smaller artifacts are
faster.

## Method and provenance

The planner minimizes an additive per-tensor objective subject to an exact
GGUF file-size ceiling. For a linear weight matrix `W`, quantized reconstruction
`Wq`, and diagonal input second moments `d_j = E[x_j^2]`, the activation score
is `sum_ij((W_ij - Wq_ij)^2 * d_j)`. This is the expected squared layer-output
error under a diagonal input covariance approximation. It uses input
activations collected from the source model; it does not model downstream
nonlinearities, correlated features, or accumulated error across layers.

The upstream algorithmic inspiration is WANDA's activation-aware importance
score, inspected at commit
[`8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`](https://github.com/locuslab/wanda/tree/8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6/lib/prune.py)
(MIT). No WANDA code was copied. The GGUF tensors were loaded through
Transformers 5.10.1's official GGUF loader, whose implementation and Apache-2.0
license are at
[`modeling_gguf_pytorch_utils.py`](https://github.com/huggingface/transformers/blob/v5.10.1/src/transformers/modeling_gguf_pytorch_utils.py)
and [`LICENSE`](https://github.com/huggingface/transformers/blob/v5.10.1/LICENSE).
This loader dequantizes the complete model to float32; the collector is limited
to Llama GGUFs with matching tokenizer/config metadata and refuses models whose
estimated dense float32 weights exceed 1 GiB. That bound is a model-size gate,
not a peak-RAM guarantee. It is an optional-HF research utility and adds no
runtime dependency to ModelSurgeon.

The measured artifact is
[`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
at revision `12fd25f77366fa6b3b4b768ec3050bf629380bac` (Apache-2.0), in GGUF
form from `unsloth/SmolLM2-135M-Instruct-GGUF` revision
`9e6855bc4be717fca1ef21360a1db4b29d5c559a`. The source GGUF SHA-256 is
`ed5fa30c487b282ec156c29062f1222e5c20875a944ac98289dbd242e947f747`.
Calibration and held-out evaluation use raw train/test text from
[Salesforce WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext).
The dataset card records CC-BY-SA-3.0 and GFDL terms;
dataset files are not distributed here. Train and test text SHA-256 values are
respectively
`6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57` and
`a1ac9f35d8d3014f89b8d5cb4cfd5e55516c0126b3a8de75447b7d3084fc5011`.
The calibration environment used Python 3.12.13, PyTorch 2.13.0+cpu, and
Transformers 5.10.1. The collector defaults to a fixed PyTorch seed and one
thread; two independent runs produced byte-identical moment manifests.

Activation moments came from eight 512-token training contexts (4,096 tokens),
with 210 Llama projection inputs hooked in the source model. Tokenizer
vocabulary, BPE merges, special tokens, BOS behavior, key model dimensions,
RoPE base, and normalization epsilon were checked against the GGUF metadata.
The calibration manifest is bound to the source hash. Only 30 projection
tensors have shapes eligible for Q3_K in this model; incompatible tensor
layouts retain their source codecs. The byte ceiling was 100,181,436 bytes,
95% of the 105,454,144-byte source. The independent planner used the same
ceiling and codecs for activation-weighted and weight-only allocation.

## Results

| Artifact | Bytes | Size reduction | Changed tensors | WikiText-2 test PPL ± reported uncertainty |
| --- | ---: | ---: | ---: | ---: |
| Source GGUF | 105,454,144 | — | — | 19.7622 ± 0.15952 |
| Weight-MSE allocation | 100,145,760 | 5.034% | 18 | 23.9716 ± 0.19275 |
| Activation-weighted allocation | 100,125,024 | 5.053% | 24 | 21.9718 ± 0.17756 |

Both candidates were reopened and evaluated with CPU-only llama.cpp at commit
`c7fc50941` (`llama-perplexity`, 8 threads, context/batch/microbatch 512,
514 windows over the same held-out text). The activation-weighted candidate
reduced the PPL penalty from `+4.2094` to `+2.2096`, a 1.9998 PPL-point or
47.5% smaller penalty relative to weight-only allocation. Its reported
uncertainty interval still does not overlap the source interval. **Both
candidates regress materially in quality; neither passes a quality gate or is
promoted.** This is evidence that the local activation proxy ranks codec
allocations better on this model and dataset, not evidence of quality
preservation or generalization.

The immutable result JSON is stored outside the repository with SHA-256
`1d89187f86474c5bf49b340189a9f989b53e8db65ab7a9110a2a3fa89b0b41c3`. Source,
candidate and calibration files also remain local. Candidate hashes are
`ab390de77e83d183c2fe725578e08e8420fab67f394133bd22752018a4a10d8f` for
weight-only and `905a45028be5b2f99d7a297952ae28fe6bd68f368057bc1955e5c4e53664be24`
for activation-weighted allocation. A fixed-seed, deterministic, single-thread
rerun of the checked-in collector generated calibration moments with SHA-256
`ebb5cc96b64c1958e9222c83d9dd635b968360d39045a6258b856b8c2f8451d8`; its
planner output matched the evaluated candidate hash above.

## Reproduction

Use the same pinned model revision, original GGUF, and WikiText-2 train text.
The collector requires the optional HF dependencies and keeps all calibration
artifacts local:

```powershell
uv run --frozen --extra hf python tools/collect_gguf_activation_moments.py `
  --source source.gguf `
  --model HuggingFaceTB/SmolLM2-135M-Instruct `
  --model-revision 12fd25f77366fa6b3b4b768ec3050bf629380bac `
  --calibration-text wiki.train.raw.txt `
  --dataset Salesforce/wikitext:wikitext-2-raw-v1 `
  --dataset-revision sha256:6707892fa3788b5ab9ed78ab5ff37d9fe825f6011a2ad4fcd6a6d467f0e7da57 `
  --split train --output activation-moments.json
```

Plan and write a candidate at the recorded 95% byte ceiling:

```powershell
uv run --frozen python tools/benchmark_gguf_quantization_budget.py `
  --source source.gguf --candidate activation-candidate.gguf `
  --target-file-bytes 100181436 `
  --activation-moments activation-moments.json --report activation-plan.json
```

The full comparison used `benchmark_gguf_perplexity` with llama.cpp
`c7fc50941`, WikiText-2 raw test text SHA above, 514 chunks and context size
512. The local report retains the commands, return codes, per-window output,
hashes, and uncertainty. Reproduction is CPU-only and does not require the
reference RTX 3060.

## Acquisition decision

| Field | Assessment |
| --- | --- |
| Source and component | WANDA activation-aware importance, MIT, commit `8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6`; Transformers GGUF loader, Apache-2.0, tag `v5.10.1`. |
| Current equivalent | The planner already estimates codec-local weight MSE and exact file bytes; it previously had no calibration-derived layer-output proxy or source-bound activation collector. |
| Approach | **REIMPLEMENT** the diagonal-covariance loss from the mathematical principle; **INTEGRATE** the optional official Transformers GGUF loading API for bounded research calibration. No upstream source code is copied. |
| Technical advantage | Ranks quantization error by observed input-feature energy; on this fixed-size candidate pair it reduced the held-out WikiText PPL penalty by 47.5% versus weight-only allocation. |
| Hardware and architectures | Calibration dequantizes a 538 MB dense float32 Llama model on CPU; the configured 1 GiB dense-weight guard does not guarantee peak RAM. This exact Llama/BPE GGUF was verified. Planner writes in bounded chunks; no GPU needed. |
| Expected benefit and criteria | Hypothesis: improve quality at equal byte budget. Pass only after repeated model-family/corpus results show a consistent, quality-gated Pareto improvement against weight-MSE and activation-aware baselines. Current single-pair result is a partial signal, with both candidates failing quality retention. |
| Risk and validation | Diagonal covariance ignores feature correlation and error propagation; Transformers private tokenizer details may evolve; only one small model and one held-out dataset were measured. Validate other Llama variants, independent calibration/evaluation corpora, reload, task quality, deployment speed, and peak resource use. |
| Priority | P0 research; keep experimental until broader evidence supports promotion. |

## Decision

Keep activation-weighted allocation as an experimental planner objective, but
do not use its score as an optimizer quality prediction or promotion signal.
Next, test independent calibration and evaluation corpora, repeated model
families, and quality-gated candidate selection; compare against a properly
matched activation-aware baseline before connecting this objective to
autonomous search. The current full-model PPL regression is retained as a
negative result.

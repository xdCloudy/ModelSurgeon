# ShortGPT depth pruning on Qwen2.5-0.5B

Date: 2026-10-26

Purpose: check whether activation-aware ShortGPT layer selection transfers to a second model family and verify that the pruned Hugging Face artifact reloads.

## Result

One of 24 transformer layers was removed from each candidate. All methods therefore removed 14,912,384 parameters (3.02% of the 494,032,768-parameter source model).

| Candidate | Removed layer | WikiText-2 test perplexity | Delta from source |
|---|---:|---:|---:|
| Source model | — | 28.9815 | — |
| ShortGPT block influence | 18 | 34.9985 | +6.0170 |
| Weight magnitude | 11 | 31.3988 | +2.4173 |
| Random, seed 11 | 14 | 34.4933 | +5.5118 |
| Random, seed 23 | 9 | 31.2816 | +2.3001 |
| Random, seed 47 | 11 | 31.3988 | +2.4173 |

The ShortGPT-selected removal had the largest perplexity increase among the measured candidates. It did not beat the magnitude baseline or any of the three sampled random choices. This is negative, descriptive evidence for this model and evaluation slice; three random samples and 2,032 test tokens are not sufficient to establish a general ranking.

The ShortGPT artifact reloaded under Transformers 5.10.1 and produced finite logits. It retained 479,120,384 parameters and occupied 969,698,404 bytes including tokenizer/config files, compared with 999,586,188 bytes for the source directory. The candidate CPU scoring pass took 48.97 seconds versus 48.46 seconds for the source pass. This single CPU measurement does not establish an inference speed change.

## Reproduction and provenance

- Model: [`Qwen/Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B), revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- Source `model.safetensors` SHA-256: `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
- Calibration and evaluation data: [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext), revision `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1` train/test splits.
- Train Parquet SHA-256: `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`.
- Test Parquet SHA-256: `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
- The Parquet `text` values were joined with newline separators and UTF-8 encoded. Resulting train/test text SHA-256 values are recorded in the JSON report alongside byte counts.
- Procedure: 2,048 train tokens for block-influence calibration; 2,032 scored test tokens in 128-token chunks; seeds 11, 23, and 47; CPU inference with four Torch threads.
- Environment: Windows 11, Intel Core i5-14600K, Python 3.14.7, Torch 2.14.1+cu130, Transformers 5.10.1. Torch ran on CPU.
- Raw machine-readable report: [`shortgpt-depth-qwen25-0.5b-2026-10-26.json`](shortgpt-depth-qwen25-0.5b-2026-10-26.json), SHA-256 `b599004d71c970add400d379229155c7b1d883c66300a9cd9d922b58ea8d5f87`.

No upstream source code was copied for this change. The ShortGPT-style scorer and benchmark harness are ModelSurgeon code already present on the PR branch. The Qwen model and WikiText files were used as local benchmark inputs and are not redistributed here.

## Compatibility finding

The first physical run exposed a save failure with newer Transformers configs: Qwen's `layer_types` list still had 24 entries after `num_hidden_layers` and the module list dropped to 23. The layer-removal adapter now validates this optional per-layer config field before mutation and filters it using the same retained-layer mapping. Unit tests cover aligned metadata and reject mismatched metadata without changing the model. The successful rerun saved, reloaded, and scored the artifact.

## Limits and next step

This is one model, one removed layer, one calibration slice, one short held-out slice, and three random draws. It does not measure GPU deployment, task accuracy, memory peak, decoding throughput, or cross-model generalization. The result suggests testing other held-out samples and a second Qwen size before treating activation-aware depth scores as competitive with magnitude selection.

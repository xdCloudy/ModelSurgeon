# Qwen3.5 REAP expert-pruning full held-out test

Date: 2026-10-27  
Status: one checkpoint and one language-modeling evaluation cell; exploratory, not a promotion result.

## Result

At the same 12.5% expert cut (224/256 experts retained in each of 41 MoE layers), the upstream POE REAP score had the best held-out WikiText-2 PPL among the tested pruning methods. Its PPL was 6.9050, versus 7.5839 and 7.5997 for two random seeds and 8.7952 for router-column norm. However, relative to the unpruned source PPL of 6.7090, REAP retained 97.16% under the project's `source PPL / candidate PPL` convention and therefore **failed the 98% gate** (candidate PPL must be at most 6.8459). It is not eligible for promotion or autonomous selection.

| Method | Artifact bytes | Held-out PPL ± llama.cpp reported uncertainty | Retention (`source / candidate`) | 98% gate |
| --- | ---: | ---: | ---: | --- |
| Unpruned source | 21,713,462,496 | 6.7090 ± 0.04243 | 100.00% | pass |
| Router-column norm, 224/layer | 19,199,501,568 | 8.7952 ± 0.05828 | 76.28% | fail |
| Random, seed 20261026, 224/layer | 19,199,501,568 | 7.5839 ± 0.04910 | 88.46% | fail |
| Random, seed 20261027, 224/layer | 19,199,501,568 | 7.5997 ± 0.04912 | 88.28% | fail |
| POE REAP, 224/layer | 19,199,501,664 | 6.9050 ± 0.04387 | 97.16% | **fail** |

The REAP point estimate is 0.6789 and 0.6947 PPL lower than the two random runs, and 1.8902 lower than router norm. The reported `±` values are llama.cpp's per-run uncertainty output, not paired confidence intervals or a statistical test. The random seed difference is 0.0158. This single model/dataset/cut is evidence that the ranking is worth investigating; it does not establish significance, generalization, or a quality-preserving operating point.

### Paired held-out chunk analysis

As a post-hoc analysis of the preserved, aligned 580-row logs, I recovered each 512-token chunk's mean NLL from llama.cpp's cumulative mean NLL, then bootstrapped paired chunks in circular contiguous blocks of 10 chunks (5,120 tokens). With 20,000 resamples and seed `20261027`, REAP's PPL delta and percentile 95% block-bootstrap interval were `-1.8902 [-2.1239, -1.6699]` against router norm, `-0.6789 [-0.7704, -0.5962]` against random seed `20261026`, and `-0.6947 [-0.7630, -0.6336]` against random seed `20261027`. Negative values favor REAP. All three intervals are below zero for this ordered WikiText-2 test corpus.

These intervals quantify chunk-sampling variation conditional on these fixed artifacts and this single corpus; they do not represent variation across calibration samples, model checkpoints, datasets, or tasks. The analysis is post-hoc and does not turn the result into a generalization or promotion claim. The exact raw logs, hashes, result JSON, and standard-library-only calculation script are preserved in `records/`; run `tools/paired_ppl_block_bootstrap.py` with the four `--reap`, `--router-norm`, `--random-20261026`, and `--random-20261027` log paths to reproduce it.

The REAP artifact is 2,513,960,832 bytes (11.58%) smaller than the source, and has 31.38B parameters after removing 32 experts from each layer. The 96-byte excess over the ModelSurgeon-produced random artifacts is POE provenance metadata and is negligible relative to model size.

A follow-up train-versus-validation calibration comparison found mean Jaccard 0.257 between the two REAP bottom-12.5% expert sets, despite high within-run convergence. This strengthens the need to test calibration sensitivity before generalizing the PPL result; details and raw score profiles are in the [calibration-stability record](qwen35moe-reap-calibration-stability-2026-10-27.md).

## Reproduction inputs and procedure

- Source GGUF SHA-256: `ed118e7768330eb440ddc8ade42a04005ecab37f131d7e84bb08d14d1e315ae1`; source model revision `eed558458ca9ff6972063c486c7f1ab9e0b0c93e`.
- Calibration used 8,192 tokens from the WikiText-2 **training** split; raw training text SHA-256 `fb23ad9643a34514eec5cb85ec2a6f49d1a33e6a3d5077dff5a403e1d18f5047`. The held-out test split was not used for scoring or selection.
- Held-out WikiText-2 raw test SHA-256: `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`; 580 chunks / 296,960 scored tokens.
- Upstream scorer/writer: POE `3908873326bcdbe097f4184729db7c33fff1cf60` (MIT). The REAP score originates from CerebrasResearch/REAP `1970473c51ca3caeb98c10392f15b3a08a672974` (Apache-2.0). The ModelSurgeon PR #738 adapter is an independent implementation; no upstream source was copied into this experiment or repository. The upstream POE tool wrote the candidate artifact.
- Calibration runtime: Linux/WSL CPU build of llama.cpp `c7fc5094179bf8a9c73c18593db714f8da41a402`, `GGML_CPU_REPACK=OFF`, 12 threads, 128 batch, `ngl=0`; 545 seconds. The resulting profile had 12,800 MoE tensors, 0 invalid expert IDs and 0 skipped REAP captures. This is one run; independent calibration-subset stability is not measured.
- Candidate POE plan SHA-256: `e5ba3580e9e4d58bbd55c58fe34463f9fb3efa02e642645a5ac7da5c547d94a6`; REAP profile SHA-256: `a6b2ab903ce995ab6fd54d125f28f6d3a96efa5790c27e308791bd5738299314`; REAP output GGUF SHA-256: `bd39618202c097173ecb675418db26aa39807dbfdb450ab9177a149d491d25c6`.
- PPL evaluation: llama.cpp `0.7.0-dev`, build 11804, commit `c7fc50941`, Windows binary, CPU (`ngl=0`), 12 threads, `-c 512 -b 512 -ub 256 --chunks -1 --no-warmup --ppl-output-type 1`; exit code 0 for the complete held-out set. The candidate passed POE plan validation and was loadable by llama.cpp for full PPL evaluation.
- Raw stdout and stderr are preserved in [`records/qwen35moe-reap-224-wikitext-ppl-output.txt`](records/qwen35moe-reap-224-wikitext-ppl-output.txt) and [`records/qwen35moe-reap-224-wikitext-ppl-stderr.log`](records/qwen35moe-reap-224-wikitext-ppl-stderr.log). Their SHA-256 values are, respectively, `11f7e46de10bd70fc77f2a3d228d4510c20897ff41864a2ff1d9cf3059374d6e` and `4c6c95e563edfd0d70a00840bcebb321c770208744918d191eab882a861dfefe`.

The POE calibration was run on WSL while held-out PPL was run by the native Windows binary. They used the same pinned llama.cpp source revision, but different builds/settings for CPU repacking; the profiling-to-evaluation deployment path therefore needs a same-runtime replication. The 545 seconds is the REAP profiling time; plan construction/application and held-out PPL wall time were not recorded here.

## Interpretation and limitations

REAP's signal is the mean, over routed calibration tokens, of selected gate weight multiplied by the norm of that expert's output. This tests output contribution, not a complete estimate of loss increase after coordinated expert removal and router renormalization. Random and router-norm methods also failed the gate, although REAP substantially reduced the observed PPL penalty. It is unknown whether more calibration tokens, a different calibration distribution, preserving high-impact experts, or iterative prune-and-repair would meet the gate.

This result does **not** measure coding/reasoning/task accuracy, prefill/decode throughput, latency, peak RAM/VRAM, load time, energy, or user-facing quality. Full evaluation success proves artifact parsing/inference for this PPL run, not broad runtime compatibility. No significance claim is made from overlapping per-run uncertainty values; use paired chunk-level comparisons and multiple independent calibration subsets/seeds in replication.

The checkpoint declares AGPL-3.0; the dataset terms are those of WikiText-2 (CC BY-SA/GFDL as stated by its dataset card). Neither weights nor raw training/evaluation text are included here. Third-party algorithm licenses and exact revisions are listed above. The JSON record contains metadata and hashes only.

## Decision and next experiment

Keep REAP research-only. Before promoting the PR #738 adapter or integrating it into pruning decisions, compare its score ordering and generated cut against the POE result on this pinned checkpoint; then run at least three independent, fixed-size calibration samples, paired with multiple random seeds and router norm, at the same retained-expert count. Apply the existing 98% quality gate without changing it. For candidates that pass, measure task quality, reload/runtime validity, artifact bytes/parameters, prefill and decode throughput, peak RAM/VRAM, and total optimization time on the same runtime/hardware. If 12.5% does not pass, test smaller cuts without substituting them into this result. Retain failed artifacts and outcomes.

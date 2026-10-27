# Qwen3.5 REAP multi-profile selection outcome

Date: 2026-10-27  
Status: full held-out PPL run completed; exploratory; no promotion.

## Result

I used POE's MIT-licensed multi-profile planner at commit `3908873326bcdbe097f4184729db7c33fff1cf60` to create a 12.5% expert cut from two equal-weight, disjoint 8,192-token WikiText-2 train profiles. POE min-max normalizes each profile's layer scores, then sums the normalized scores with the supplied profile weights. The plan kept 224 of 256 experts in each of 41 layers. The resulting Q4_K_M GGUF was 19,199,501,664 bytes (11.58% smaller than the source), passed POE structural/provenance validation, and loaded for the complete held-out test run.

The ensemble profile cut produced PPL **6.9008**, compared with **6.9050** for the single train profile. The paired difference is only `-0.00419` PPL, with a 95% moving-block bootstrap interval of `[-0.03598, 0.03031]`; the interval includes zero. Both candidates fail the unchanged 98% retention gate: the ensemble retained `97.22%` (`6.7090 / 6.9008`) against a maximum allowed PPL of `6.8459`.

| Method | Experts retained/layer | Artifact bytes | Test PPL | Retention | 98% gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Unpruned source | 256/256 | 21,713,462,496 | 6.7090 | 100.00% | pass |
| REAP, one train profile | 224/256 | 19,199,501,664 | 6.9050 | 97.16% | fail |
| REAP, two disjoint train profiles | 224/256 | 19,199,501,664 | 6.9008 | 97.22% | fail |
| Router-column norm | 224/256 | 19,199,501,568 | 8.7952 | 76.28% | fail |
| Random, seed 20261026 | 224/256 | 19,199,501,568 | 7.5839 | 88.47% | fail |
| Random, seed 20261027 | 224/256 | 19,199,501,568 | 7.5997 | 88.28% | fail |

The multi-profile candidate remains substantially better than the random and router-norm point estimates on this one checkpoint and test corpus, but it did not materially improve on single-profile REAP and did not reach the project's quality gate. The paired intervals against the two random seeds and router norm are below zero; these intervals are conditional on the same 580 chunks and do not establish generalization across models, tasks, or datasets. The candidate therefore remains research-only.

## Reproduction and provenance

- Model: `endystrike/Endy-Qwen3.6-CyberSec-35B-A3B`, revision `eed558458ca9ff6972063c486c7f1ab9e0b0c93e`, source GGUF SHA-256 `ed118e7768330eb440ddc8ade42a04005ecab37f131d7e84bb08d14d1e315ae1`, fingerprint `poe1:a0952c3c50425919`; Qwen3.5 MoE, 41 layers, 256 experts/layer, top-k 8. The checkpoint declares AGPL-3.0; no model weights are included.
- Calibration profiles: the first uses the initial 8,192 tokens of WikiText-2 train; the second uses the disjoint raw-text suffix starting at byte 1,048,576. Their profile SHA-256 values are `4a068990bcaf4a4dd3e9bb75d650cb1d3e0de56c7dfd955630f1ea7667b78a3b` and `4ef7e6033b0363c5a35f21a7c61407cf96ecea91768a241975bef5eece77edd4`. The profile and text identities are in the [calibration-sample stability report](qwen35moe-reap-calibration-stability-2026-10-27.md); raw calibration text is not redistributed.
- Planner and structural writer: [POE](https://github.com/peonist-ai/crucible/tree/3908873326bcdbe097f4184729db7c33fff1cf60), MIT, exact commit `3908873326bcdbe097f4184729db7c33fff1cf60`; llama.cpp `c7fc5094179bf8a9c73c18593db714f8da41a402`. The POE planner and writer were run as upstream tools; no POE source was copied. Plan SHA-256 `041d5369b7f14aac1dee3861c03b202843864377397e40cc058b4ded7f5d5832`; candidate GGUF SHA-256 `1e775265f9ddb09f4dfd5715fd7208b69e73eebaa06dcc9b9630c8e663dad8bc`.
- Evaluation: WikiText-2 raw test, 296,960 tokens / 580 aligned 512-token chunks, SHA-256 `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`. It was untouched by calibration. Windows llama.cpp 0.7.0-dev build 11804, CPU, 12 threads, `ngl=0`, context 512, batch 512, ubatch 256, all chunks, no warmup, `--ppl-output-type 1`. The run exited successfully; its logs warn that `PrefetchVirtualMemory` failed for lack of memory resources, then show all 580 chunks and final PPL. Output timestamps imply about 41 minutes wall time. No peak-memory sampler, task accuracy, latency, prefill/decode throughput, energy, or load-time benchmark was collected.
- The first POE output attempt to the Windows DrvFS mount failed after writing approximately 7.9 GB and removed its partial file. Free space was available; the cause was not determined. Applying the same plan on the WSL ext4 filesystem succeeded and passed structural validation. The Windows inference process then evaluated the WSL-hosted artifact successfully.
- The exact raw stdout and stderr are retained in `records/` with SHA-256 `0946241e0a1fc4be4841f2f2dccd78658dd4a182a076f441dc254b5565f81b99` and `30c4fdccdb2cf6f51485f87767ebcf98bf0f62d742388c39e9e03df14911115a`. The machine-readable [paired bootstrap record](records/qwen35moe-reap-ensemble-train-ab-paired-bootstrap-2026-10-27.json) retains all input log hashes and the 20,000-resample results. Its calculation uses [`tools/paired_ppl_block_bootstrap.py`](../tools/paired_ppl_block_bootstrap.py), with named candidate/baseline inputs.

## Decision

Do not add multi-profile REAP as an accepted selection policy from this result. Equal-weight aggregation of two disjoint profiles changed the selected cut but showed no measurable held-out PPL gain over a single profile, and both cuts failed the quality gate. The next optimization experiment should change the compression point or add recovery, while keeping this 12.5% result as a negative/neutral control; it should not silently lower the quality threshold. No inference-speed or memory-efficiency benefit is claimed from the smaller artifact.

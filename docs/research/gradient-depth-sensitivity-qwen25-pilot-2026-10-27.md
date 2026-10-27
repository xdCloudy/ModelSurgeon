# Bounded first-order gradient depth-sensitivity pilot

Assessment date: 2026-10-27. This is a CPU-only, single-model pilot implemented in open PR [#779](https://github.com/xdCloudy/ModelSurgeon/pull/779), stacked on the activation-baseline PR [#735](https://github.com/xdCloudy/ModelSurgeon/pull/735). The code is experimental and is not integrated on main.

## Protocol

- Model: `Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`; weight SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`.
- Dataset: `Salesforce/wikitext`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The serialized training text SHA-256 is `c58d555a94ada0882d1e8c0b3dc7ba68e1c269da5ad9e7a057300245d130c4e8` (256 calibration tokens); held-out text SHA-256 is `aca2f46735043bcfd0a44eca981d04627b9cdf74c4c9a04bf0856d04066f58fc` (508 target tokens).
- Baseline and four one-layer deletions were evaluated on the same CPU test split: signed first-order block-output directional sensitivity, absolute per-token sensitivity, ShortGPT block influence, magnitude, and one random choice (seed 11). Each removed 14,912,384 parameters (3.02%). Only one test repeat and one random seed were run.
- Environment: Windows 11, Python 3.12.14, Torch 2.13.0+cpu, Transformers 5.10.1, CPU, two Torch threads. The installed Torch build has no CUDA runtime; no GPU was used.

## Results

| Method | Removed layer | Held-out PPL | Delta from baseline | Save/reload |
|---|---:|---:|---:|---|
| Baseline | — | 23.2089 | — | Source checkpoint |
| Signed first-order sensitivity | 0 | 56,977.3954 | +56,954.1865 | Passed; PPL matched |
| Absolute per-token sensitivity | 11 | 25.1388 | +1.9298 | Passed; PPL matched |
| Magnitude | 11 | 25.1388 | +1.9298 | Reference selection; same candidate |
| ShortGPT block influence | 18 | 27.3489 | +4.1400 | Passed; PPL matched |
| Random, seed 11 | 14 | 27.3267 | +4.1178 | Physical evaluation |

The absolute sensitivity variant exactly tied magnitude on this candidate. The signed variant was catastrophically harmful. Absolute scoring took 623.36 seconds for 256 calibration tokens and the whole pilot took 915.08 seconds. A larger CPU attempt was stopped after 56 minutes without producing results. The parameter and disk reductions were physically realized, and the evaluated gradient artifacts reloaded with finite logits. This pilot did not measure a task-quality gate, uncertainty, autoregressive decode speed, or a deployment benefit.

## Conclusion and scope

This is negative evidence for the two tested first-order scoring variants in this one short-context Qwen2.5 cell. It does not refute Taylor-based structured pruning generally or reproduce the CNN filter-pruning setting in [Molchanov et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf). Further estimator choices, models, longer contexts, multiple seeds, task evaluation, and cost reductions remain open. No upstream code was copied.

## Immutable evidence and reproduction

- [Raw measurements and artifact SHA-256 manifest at PR #779 commit `ad254dc980d80260449f25e7c27e50915fc9940f`](https://github.com/xdCloudy/ModelSurgeon/blob/ad254dc980d80260449f25e7c27e50915fc9940f/docs/research/gradient-depth-sensitivity-qwen25-pilot-2026-10-27.json)
- [Benchmark implementation at the same commit](https://github.com/xdCloudy/ModelSurgeon/blob/ad254dc980d80260449f25e7c27e50915fc9940f/tools/benchmark_shortgpt_depth.py)
- Literature record: Molchanov et al., *Importance Estimation for Neural Network Pruning*, arXiv `1906.10771v1`. The paper reports first- and second-order Taylor estimates for CNN filter removal; the upstream implementation and its licensing have not been audited, and nothing from it was reused.
- Reproduction requires PR #779’s benchmark script, the pinned Hugging Face model and dataset revisions, the listed serialized split hashes, Python/Torch/Transformers versions, CPU device, and 2 Torch threads. The benchmark writes the full protocol, outcomes, artifact hashes, and runtime to the linked JSON report.

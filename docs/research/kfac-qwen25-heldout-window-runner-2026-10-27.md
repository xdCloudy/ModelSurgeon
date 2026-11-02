# KFAC held-out window runner follow-up

The bounded KFAC physical benchmark now accepts `--evaluation-offset-tokens` in addition to its calibration offset. The selected KFAC, WANDA-style activation, magnitude, and fixed random masks can therefore be checked against another non-overlapping target window from the pinned held-out split. The report records the evaluation offset and hashes the resulting input tokens. The follow-up does not change the selection method or the declared quality gate.

Two focused tests cover offset chunk construction and rejection when the requested targets run past the available tokens. Ruff, formatting, and `git diff --check` passed locally. The code is in open PR [#780](https://github.com/xdCloudy/ModelSurgeon/pull/780), commit [`e473388`](https://github.com/xdCloudy/ModelSurgeon/commit/e4733886721c724d5135f88700febe14c03d1c88); its quality, Hugging Face smoke, LightGBM smoke, and CPU performance CI checks all passed.

No physical follow-up was run. Only 1.3 GiB of host RAM was free, while the preceding Qwen scorer used 11.62–12.14 GB peak RSS and planned a 14.97 GB workspace. CUDA was explicitly disabled. This records an experiment-enabling change, not new optimization evidence or a positive result.

## Next experiment

With sufficient free host RAM, keep the existing 256-token train calibration window and compare the exact same 64-channel candidate masks over 1,024 held-out targets beginning at offset 2,048. Report baseline and candidate perplexity, absolute gate outcomes, reload checks, and resource costs. Keep the method experimental if candidates continue to fail the predeclared quality gate.

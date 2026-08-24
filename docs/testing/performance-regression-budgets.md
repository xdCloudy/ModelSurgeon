# Performance regression budgets

`tools/run_performance_regressions.py` executes deterministic CPU, GGUF-streaming,
and CUDA fixtures against the versioned profiles in
`tests/fixtures/performance_budgets_v1.json`. It is a regression alert system, not a
cross-hardware benchmark leaderboard.

## Lanes and fixtures

| Profile | Lane | Repetitions | Fixed work | Automation |
| --- | --- | ---: | --- | --- |
| `pr-cpu` | `pr_cpu` | 3 | 8 MiB SHA-256 stream and 8 MiB transactional F32 GGUF write | Every pull request on `ubuntu-latest` |
| `consumer-cpu` | `large_cpu` | 3 | 64 MiB SHA-256 stream and 64 MiB transactional F32 GGUF write | Separate self-hosted CPU workflow |
| `consumer-gpu` | `gpu` | 5 | 1024×1024 FP32 CUDA matrix multiplication | Separate self-hosted GPU workflow |

The pull-request lane deliberately remains small and never imports or invokes the CUDA
fixture. Larger CPU/GGUF work and real GPU work live in
`.github/workflows/performance-regression.yml`; they are manually dispatchable and can
be scheduled by enabling `PERFORMANCE_CPU_RUNNER_ENABLED` and
`PERFORMANCE_GPU_RUNNER_ENABLED` on runners carrying the matching `performance,cpu`
or `performance,gpu` labels.

## Decision contract

Every budget fixes a case, stage, metric, unit, direction, baseline, relative
tolerance, and absolute tolerance. The effective allowance is:

```text
max(absolute_tolerance, baseline × relative_tolerance)
```

Maximum metrics fail when the median is above `baseline + allowance`. Minimum metrics
fail when it is below `max(0, baseline - allowance)`. Repetition indices must be unique,
all declared budgets must have enough measurements, fixture identities and units must
match, and one report cannot mix hardware contexts.

The runner records:

- wall and process CPU time;
- absolute peak process RSS;
- process read/write byte deltas when the OS exposes them;
- PyTorch peak allocated and reserved CUDA memory for the GPU lane; and
- processed bytes per second for the streaming GGUF case.

Every report contains both the checked-in reference hardware record and the complete
actual `HardwareNormalizationContext`, including its content-derived context ID. A
passing number on different hardware is therefore never silently presented as a
same-host comparison.

## Run locally

```text
python tools/run_performance_regressions.py \
  --profile pr-cpu \
  --output work/performance-pr-cpu.json

python tools/run_performance_regressions.py \
  --profile consumer-cpu \
  --output work/performance-consumer-cpu.json

python tools/run_performance_regressions.py \
  --profile consumer-gpu \
  --output work/performance-consumer-gpu.json
```

Outputs are canonical, non-overwriting JSON. Exit code `0` means every budget passed,
`1` means at least one measured regression, and `2` means the run or configuration was
invalid. A failure report is still retained for diagnosis.

## Baseline update policy

Budget updates are reviewed source changes; automation never rewrites them. To update a
baseline:

1. Keep the fixture identity and work size unchanged, or introduce a new fixture version.
2. Run the profile at least twice without competing workloads on the declared reference
   hardware.
3. Compare every repetition, median, actual hardware context, and process/software change.
4. Explain why the movement is intentional and choose the smallest tolerance that still
   covers normal runner noise.
5. Retain the before/after reports in the PR or a research evidence record.

Do not raise a threshold merely to make an unexplained failure green. Timings from
GitHub-hosted runners intentionally have more headroom than the fixed consumer workstation;
they should be recalibrated only from multiple hosted runs.

# Physical deployment benchmark CLI

`modelsurgeon benchmark deploy` is the physical-runtime companion to the
equal-budget benchmark matrix. It plans and resumes measurements for real
Hugging Face/safetensors and native GGUF artifacts without treating a missing
runtime as a zero-valued measurement.

## Contract

Every artifact is identified by format, SHA-256 digest, byte size, and an
optional local path. Every cell retains the runtime name, CPU thread count,
GPU-offload setting, device, hardware context, timeout, drift tolerance,
warmups, and repetitions. The frozen minimum is two warmups and seven
repetitions.

The common metric names and units are:

| Metric | Unit |
| --- | --- |
| `load_time_seconds` | `s` |
| `prefill_tokens_per_second` | `tokens/s` |
| `decode_tokens_per_second` | `tokens/s` |
| `latency_seconds` | `s` |
| `peak_ram_bytes` | `bytes` |
| `peak_vram_bytes` | `bytes` |
| `disk_bytes` | `bytes` |

Measured records retain all samples plus median, rank-based p95, and population
standard deviation. Unsupported, failed, timeout, OOM, drift, invalid-artifact,
and interrupted results remain distinct terminal outcomes.

## Execution boundary

The plan command is local and deterministic. Runtime-specific execution is an
explicit JSON-lines runner boundary: it receives one cell on stdin and must
emit one versioned deployment record on stdout. The safe default records
`unsupported` rather than pretending that a runtime is available. A runner may
use the existing llama.cpp throughput implementation for GGUF and a bounded
Hugging Face child process for HF, but both must emit the same shared metric
names and units.

Example:

```text
modelsurgeon benchmark deploy plan \
  --artifact hf=baseline.safetensors \
  --artifact gguf=baseline.gguf \
  --output deployment-plan.json
modelsurgeon benchmark deploy run \
  --plan deployment-plan.json \
  --state deployment-state.json \
  --runner my_runner:factory
modelsurgeon benchmark deploy audit \
  --plan deployment-plan.json \
  --state deployment-state.json
```

Plans are immutable and states are atomically replaced. Completed cells are
never re-run during resume, and imported records are checked against the
content-addressed artifact/profile identity before they enter state.

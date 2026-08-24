# ModelSurgeon

[![CI](https://github.com/xdCloudy/ModelSurgeon/actions/workflows/ci.yml/badge.svg)](https://github.com/xdCloudy/ModelSurgeon/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**Learned, safety-first structural optimization and pruning for neural networks.**

ModelSurgeon is experimental research software for measuring the effect of structural changes to neural networks, learning which mutations are likely to be safe, and eventually searching for smaller or faster models under explicit quality constraints.

It combines model inspection, component graphs, feature extraction, transactional mutation, tiered evaluation, reproducible experiment data, learned ranking models, and low-memory native GGUF surgery in one local-first system.

> [!WARNING]
> Model surgery can permanently damage checkpoints. ModelSurgeon treats source models as immutable and does not overwrite them by default. Keep verified backups and treat generated checkpoints as untrusted until they pass validation.

## How it works

```text
model
  -> inspect + map components
  -> measure static/runtime features
  -> propose mutation
  -> apply transactionally
  -> evaluate
  -> keep or rollback
  -> record result
  -> train/update surgeon
  -> repeat
```

ModelSurgeon has two complementary execution paths:

```text
HIGH-PRECISION PATH                     LOW-HARDWARE PATH
HF / safetensors                        existing quantized GGUF
      |                                        |
inspect + instrument                    mmap + inspect in place
      |                                        |
mask / structural surgery               selective decode
      |                                        |
new checkpoint                          mutate + requantize
      |                                        |
optional quantization                   streaming new GGUF
```

The GGUF path is designed so ordinary surgery does not require a complete floating-point model in RAM or on disk.

## Current status

**Pre-alpha (`0.0.1`).** The codebase is substantially beyond the original scaffold, but the project is not yet a stable end-user pruning tool.

Implemented today includes:

- Hugging Face causal-LM loading, immutable revision capture, architecture-family detection, and canonical component discovery;
- stable component identities, dependency/coupling graphs, mutation constraints, graph persistence, and post-surgery identity remapping;
- static weight, spectral, activation, gradient, redundancy, latency, memory, and runtime telemetry features;
- deterministic calibration ingestion and bounded streaming aggregation;
- transactional mutation planning, masking, rollback, provenance, and single-mutation experiment execution;
- leakage-safe mutation datasets, grouped train/validation/test splits, and leakage audits;
- random, magnitude, heuristic, linear, logistic, LightGBM, and small-MLP surgeon baselines;
- immutable surgeon model bundles with preprocessing/schema/provenance metadata;
- First Surgeon proof orchestration and a real Hugging Face MLP-channel masking runtime;
- bounded memory-mapped GGUF parsing, lazy tensor access, exact quantization codecs, copy-on-surgery writing, resumability, and atomic publication;
- native GGUF structural planning/execution for supported MLP-channel and attention-head edits;
- pinned llama.cpp validation plumbing for generated GGUFs.

Still experimental or incomplete:

- the full real-world First Surgeon empirical campaign and benchmark evidence;
- real-hardware acceptance for some CUDA and external llama.cpp paths;
- physical Hugging Face tensor resizing/save-reload surgery;
- broader architecture/quantization coverage, MoE surgery, active learning, cross-model generalization, automated search, and repair.

The roadmap is dependency-driven rather than strictly sequential, so some native GGUF work has landed ahead of later empirical milestones. See [ROADMAP.md](ROADMAP.md) for the complete v1.0 plan.

## Quick start

### Requirements

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- optional: CUDA-capable PyTorch environment for GPU workflows

Clone and install the development environment with Hugging Face support:

```bash
git clone https://github.com/xdCloudy/ModelSurgeon.git
cd ModelSurgeon
uv sync --extra dev --extra hf --locked
```

Show the CLI:

```bash
uv run modelsurgeon --help
```

Inspect a Hugging Face causal LM:

```bash
uv run modelsurgeon inspect <model-id-or-local-path> \
  --device-map cpu \
  --dtype auto
```

Use `--json` for newline-delimited machine-readable output.

## CLI

| Command | Purpose |
|---|---|
| `inspect` | Load a Hugging Face causal LM and emit canonical model/component records. |
| `experiment` | Resolve, execute, evaluate, and roll back one transactional mutation through a runtime adapter. |
| `first-surgeon-proof` | Build a leakage-safe proof dataset through a generic experiment runtime. |
| `first-surgeon-hf-proof` | Run real Hugging Face MLP-channel masks and create the First Surgeon dataset. |
| `first-surgeon-evidence` | Train/evaluate the proof LightGBMs and compare them with random/magnitude baselines. |
| `train-surgeon` | Train and publish a baseline surgeon model from canonical mutation examples. |
| `predict-surgeon` | Score compatible candidates with an immutable persisted surgeon bundle. |

Global logging options:

```text
--log-level LEVEL
--log-format human|json
```

## First Surgeon workflow

The first empirical target is deliberately narrow: predict the effect of masking individual MLP channels using static + activation features, then compare learned LightGBM models against magnitude and seeded-random rankings on held-out components.

### 1. Generate a real mutation dataset

For Hugging Face causal LMs with the standard gated `gate_proj` / `up_proj` / `down_proj` MLP layout:

```bash
uv run modelsurgeon first-surgeon-hf-proof <model-id-or-local-path> ./calibration.txt \
  --output ./proof-data \
  --max-candidates 5000 \
  --sequence-length 256 \
  --max-tokens 4096 \
  --safe-perplexity-delta 0.25 \
  --seed 42 \
  --split-seed 43 \
  --tool-revision "$(git rev-parse HEAD)"
```

The command writes:

```text
proof-data/
├── examples.jsonl
├── split.json
├── leakage.json
└── campaign.json
```

It fails rather than publishing a dataset if a split is empty or the leakage audit finds contamination.

### 2. Train and compare the proof models

LightGBM is intentionally optional:

```bash
uv pip install "lightgbm>=4,<5"
```

Run the full held-out evidence pipeline:

```bash
uv run modelsurgeon first-surgeon-evidence ./proof-data/examples.jsonl \
  --split ./proof-data/split.json \
  --registry ./proof-data/registry \
  --output ./proof-data/first-surgeon-evidence.json \
  --safe-perplexity-delta 0.25 \
  --threads 4 \
  --seed 42 \
  --top-n 50 \
  --bootstrap-repetitions 1000
```

The report includes grouped held-out metrics, random/magnitude comparisons, artifact hashes, model/dataset/tool revisions, immutable-bundle inference smoke tests, and bounded training resource telemetry.

See [docs/first-surgeon-proof.md](docs/first-surgeon-proof.md) for the full protocol.

## Train your own surgeon baseline

Supported training backends:

```text
linear
logistic
lightgbm-regressor
lightgbm-classifier
mlp-regressor
mlp-classifier
```

Example regression run:

```bash
uv run modelsurgeon train-surgeon ./examples.jsonl \
  --split ./split.json \
  --registry ./artifacts/surgeons \
  --target perplexity \
  --baseline lightgbm-regressor \
  --threads 4 \
  --seed 42
```

Score compatible candidates with the emitted bundle digest:

```bash
uv run modelsurgeon predict-surgeon ./candidate.json \
  --registry ./artifacts/surgeons \
  --bundle sha256:<digest> \
  --json
```

Inference fails closed when required features are missing or the stored preprocessing/schema contract is incompatible.

## Native GGUF surgery

GGUF is a first-class format rather than an export-only target.

The current native stack includes:

- bounded read-only mmap parsing for GGUF v2/v3;
- lazy tensor handles and block/range reads;
- dense F32/F16/BF16 plus Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K and prioritized IQ4 codec support;
- architecture-aware physical mappings for supported Llama, Qwen, Mistral, and Gemma variants;
- exact block-alignment and axis validation;
- selective dequantize -> mutate -> requantize paths;
- copy-on-surgery for untouched encoded tensor ranges;
- resumable transactional output with checksums and atomic publication;
- matched no-surgery requantization controls;
- native quantized MLP-channel and attention-head removal paths;
- external llama.cpp validation with pinned tool provenance.

Unsupported architectures, layouts, codecs, or unsafe mutation geometries fail closed rather than being guessed. Dense-model support is ahead of MoE support; some newer family variants remain intentionally rejected until their semantics are implemented.

For the detailed subsystem design, see [ARCHITECTURE.md](ARCHITECTURE.md) and `docs/design/`.

## Design principles

- **Source checkpoints are immutable.** Physical output is staged separately and published atomically.
- **Fail closed.** Unknown architectures, tensor layouts, quantization types, and mutation constraints are not guessed.
- **Stable identities.** Components keep canonical IDs across inspection, features, experiments, datasets, predictions, and remapping.
- **Bound memory explicitly.** Large loops use streaming aggregation, mmap, chunking, disk-backed intermediates, and resource preflight.
- **Prefer reversible experiments first.** Masking and bypass experiments precede destructive structural edits.
- **Keep evidence reproducible.** Model, dataset, tokenizer, tool, schema, seed, hardware, and configuration provenance travel with results.
- **Prevent ML leakage.** Dataset splits are grouped around mutation/component identity rather than row-randomized after generation.
- **Local first.** The architecture targets consumer hardware and does not require distributed infrastructure.

## Reference hardware

The baseline architecture is designed around a consumer workstation class machine: roughly a 12 GB NVIDIA GPU and 64 GB system RAM, with CPU-only paths where practical.

This is a design constraint, not a promise that every future model or surgery operation will fit that hardware.

## Project layout

```text
src/modelsurgeon/
├── adapters/         # HF, safetensors, GGUF and architecture boundaries
├── graph/            # component IDs, topology, coupling and constraints
├── features/         # static, spectral, activation, gradient and runtime features
├── instrumentation/  # calibration, hooks and bounded aggregation
├── surgery/          # mutation contracts, masks, physical edits and rollback
├── evaluation/       # structural, perplexity, latency and external validation
├── experiments/      # identity, persistence, budgets, queues and reproducibility
├── datasets/         # mutation examples, stores, splits and leakage audits
├── surgeon/          # heuristics and learned baseline models
├── search/           # candidate/search infrastructure
├── explain/          # explanation/report infrastructure
└── cli/              # user-facing workflows
```

## Development

Install the locked development environment:

```bash
uv sync --extra dev --extra hf --locked
```

Run the main quality gates:

```bash
uv run ruff check .
uv run mypy src
uv run pytest --cov=modelsurgeon --cov-report=term-missing
```

CI additionally runs a real local Transformers First Surgeon smoke test and an optional-dependency LightGBM evidence smoke test.

## Documentation

- [Architecture](ARCHITECTURE.md)
- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)
- [First Surgeon proof protocol](docs/first-surgeon-proof.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)

## License

Apache-2.0. Model, dataset, tokenizer, and upstream tool licenses remain the responsibility of their users.

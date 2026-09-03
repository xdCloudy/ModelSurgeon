<div align="center">

# ModelSurgeon

### Learn what a neural network can lose — and measure what it costs.

[![CI](https://github.com/xdCloudy/ModelSurgeon/actions/workflows/ci.yml/badge.svg)](https://github.com/xdCloudy/ModelSurgeon/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-F59E0B)](https://github.com/xdCloudy/ModelSurgeon/milestones)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-4C7CBF)](LICENSE)

**Local-first, evidence-driven structural optimization for Hugging Face and GGUF models.**

[Get started](#quick-start) · [See what works](#project-status) · [Use the CLI](#cli-workflows) · [Read the architecture](ARCHITECTURE.md) · [Follow the roadmap](ROADMAP.md)

</div>

ModelSurgeon is experimental research software for learning which parts of a neural network can be removed, measuring the result, and preserving enough evidence to reproduce every decision. It combines canonical model structure, bounded feature collection, reversible experiments, learned surgeon models, constrained search, and transactional physical mutation.

It supports two complementary paths:

- **Hugging Face / safetensors** for inspection, calibration, masking experiments, learned outcome models, search, and physical tensor surgery.
- **Native GGUF** for bounded, copy-on-surgery edits to quantized models without materializing a full floating-point checkpoint.

> [!WARNING]
> ModelSurgeon is pre-alpha research software, not a production optimizer. Surgery can damage model quality or produce unusable checkpoints. Inputs are treated as immutable, outputs are staged separately, and unsupported layouts fail closed.

## Why ModelSurgeon?

| Principle | What it means in practice |
| --- | --- |
| **Evidence over intuition** | Static, activation, gradient, quality, latency, memory, and hardware evidence travel with each result. |
| **Learn from experiments** | Mutation outcomes become leakage-audited data for heuristic, linear, tree, and neural surgeon models. |
| **Reversible before destructive** | Mask, bypass, evaluate, and roll back before publishing a physically modified artifact. |
| **Quantized models are first-class** | GGUF edits selectively decode affected blocks and copy untouched encoded ranges byte-for-byte. |
| **Consumer hardware matters** | Work is bounded, streamed, resumable, and tested around a 12 GB GPU / 64 GB RAM workstation. |

## Project status

Current package version: **`0.0.1` (pre-alpha)**. The v0.5–v0.9 research path has reproducible repository evidence; **v1.0 stabilization is in progress**.

| Area | State | Current capability |
| --- | --- | --- |
| Inspection and component graph | **Implemented** | HF loading, revision provenance, architecture detection, stable component IDs, coupling, and mutation constraints. |
| Instrumentation and evaluation | **Implemented** | Static, spectral, activation, gradient, redundancy, perplexity, latency, memory, and runtime telemetry. |
| Mutation lab and datasets | **Implemented** | Transactional masks/bypasses, rollback, tiered evaluation, resumable campaigns, grouped splits, and leakage audits. |
| Learned surgeons | **Validated baseline** | Heuristic, linear/logistic, LightGBM, and MLP bundles with held-out evidence and honest negative results. |
| Active learning and search | **Experimental** | Calibrated uncertainty, bounded candidate pools, acquisition policies, resumable scheduling, Pareto archives, and repair arms. |
| Physical HF surgery | **Experimental** | Layer, attention-head, gated-MLP, and low-rank edits with shape, parameter, save, and reload checks. |
| Native GGUF surgery | **Experimental** | Exact codecs, MLP/head/layer/low-rank edits, streaming output, requantization controls, and `llama.cpp` validation. |
| Public/release surface | **In progress** | v1.0 schemas, CLI workflows, reports, performance gates, security hardening, and release documentation. |

Measured evidence currently includes:

- a leakage-free **3,000-mutation First Surgeon** campaign on SmolLM2-135M;
- active-learning, uncertainty, transfer, pruning-baseline, and iterative-search studies;
- real HF and native-GGUF physical mutation and repair runs; and
- a **134.5M–7.25B** consumer-hardware ladder on Windows with an RTX 3060 12 GB and 64 GB RAM.

The results include negative findings where a learned policy or repair did not beat the declared baseline. Start with the [research index](docs/research/README.md), [First Surgeon evidence](docs/research/v0.5-first-surgeon-evidence.md), and [consumer scale evidence](docs/research/v0.8-consumer-scale-evidence.md).

## Quick start

### Requirements

- Python **3.12+**
- [`uv`](https://docs.astral.sh/uv/)
- optional CUDA-capable PyTorch environment for GPU workflows
- optional external `llama.cpp` tools for native GGUF validation and benchmarking

```bash
git clone https://github.com/xdCloudy/ModelSurgeon.git
cd ModelSurgeon
uv sync --extra dev --extra hf --locked
uv run modelsurgeon --help
```

Inspect a Hugging Face causal language model without mutating it:

```bash
uv run modelsurgeon inspect HuggingFaceTB/SmolLM2-135M \
  --device-map cpu \
  --dtype auto
```

Add `--json` for machine-readable model and component records.

## CLI workflows

The public CLI exposes the stable orchestration boundary. Lower-level HF and GGUF surgery APIs remain library-level while their end-user contracts are stabilized for v1.0.

| Command | Purpose |
| --- | --- |
| `inspect` | Load and enumerate a Hugging Face causal language model. |
| `experiment` | Resolve, evaluate, and roll back one transactional mutation. |
| `first-surgeon-proof` | Build a leakage-safe proof dataset through a runtime adapter. |
| `first-surgeon-hf-proof` | Run real HF MLP-channel masks and create the proof dataset. |
| `first-surgeon-evidence` | Train proof LightGBMs and publish held-out baseline comparisons. |
| `train-surgeon` | Train and publish an immutable baseline surgeon bundle. |
| `predict-surgeon` | Score compatible candidates with a persisted bundle. |
| `search` | Start or resume one constrained greedy, beam, or uncertainty-aware search decision. |
| `features` | Extract bounded, cacheable model features through a trusted runtime. |
| `generate-dataset` | Run or resume a campaign and emit leakage-safe JSONL splits. |
| `reproduce` | Verify and optionally replay an immutable persisted experiment recipe. |
| `report` | Render deterministic JSON or offline HTML evidence reports. |

Global logging is available through `--log-level` and `--log-format human|json`. Run any command with `--help` for its complete contract. Generate shell-specific completion instructions with `modelsurgeon --show-completion`; use `--install-completion` only when you intend to modify the current user's shell configuration.

<details>
<summary><strong>Run the First Surgeon workflow</strong></summary>

Generate real MLP-channel mutation examples:

```bash
uv run modelsurgeon first-surgeon-hf-proof \
  HuggingFaceTB/SmolLM2-135M ./calibration.txt \
  --output ./proof-data \
  --max-candidates 3000 \
  --sequence-length 32 \
  --max-tokens 64 \
  --safe-perplexity-delta 0 \
  --seed 42 \
  --split-seed 43 \
  --tool-revision "$(git rev-parse HEAD)"
```

Train and compare the held-out LightGBM models:

```bash
uv pip install "lightgbm>=4,<5"
uv run modelsurgeon first-surgeon-evidence ./proof-data/examples.jsonl \
  --split ./proof-data/split.json \
  --registry ./proof-data/registry \
  --output ./proof-data/evidence.json \
  --safe-perplexity-delta 0 \
  --threads 4 \
  --seed 42 \
  --top-n 50 \
  --bootstrap-repetitions 1000
```

The campaign refuses to publish when a split is empty or the leakage audit finds contamination. See the [proof protocol](docs/first-surgeon-proof.md) and [evidence contract](docs/first-surgeon-evidence.md).

</details>

<details>
<summary><strong>Train and use a surgeon bundle</strong></summary>

```bash
uv run modelsurgeon train-surgeon ./examples.jsonl \
  --split ./split.json \
  --registry ./artifacts/surgeons \
  --target perplexity \
  --baseline lightgbm-regressor \
  --threads 4 \
  --seed 42

uv run modelsurgeon predict-surgeon ./candidate.json \
  --registry ./artifacts/surgeons \
  --bundle sha256:<digest> \
  --json
```

Supported baselines are `linear`, `logistic`, `lightgbm-regressor`, `lightgbm-classifier`, `mlp-regressor`, and `mlp-classifier`. Inference rejects missing features and incompatible schema or preprocessing contracts.

</details>

<details>
<summary><strong>Start or resume constrained search</strong></summary>

```bash
uv run modelsurgeon search search.json --dry-run
uv run modelsurgeon search search.json --state search.sqlite3
uv run modelsurgeon search search.json --state search.sqlite3 --resume
```

Search reserves candidates for an evaluator; predictions never silently become accepted checkpoints. See the [search CLI contract](docs/design/search-cli.md).

</details>

<details>
<summary><strong>Inspect or replay a persisted run</strong></summary>

```bash
uv run modelsurgeon reproduce run_<sha256> \
  --metadata ./artifacts/experiments.sqlite3 \
  --artifacts ./artifacts/store \
  --repository . \
  --lock ./uv.lock \
  --dry-run
```

Execution additionally requires an explicitly trusted local replay adapter. Environment drift, corrupt artifacts, missing metrics, and tolerance failures are reported rather than guessed around. See [reproducing persisted runs](docs/experiments/reproduce-run.md).

</details>

## How it fits together

```mermaid
flowchart LR
    INPUT["HF / safetensors<br/>or quantized GGUF"] --> GRAPH["Canonical graph<br/>identity · coupling · constraints"]
    GRAPH --> FEATURES["Bounded evidence<br/>static · runtime · activation · gradient"]
    FEATURES --> SURGEON["Surgeon models<br/>heuristic · linear · tree · neural"]
    SURGEON --> SEARCH["Constrained search<br/>budget · uncertainty · Pareto state"]
    SEARCH --> MUTATE["Transactional mutation<br/>mask · bypass · physical edit"]
    MUTATE --> EVAL["Tiered evaluation<br/>quality · latency · memory · size"]
    EVAL -->|accept| OUTPUT["Checkpoint + report"]
    EVAL -->|reject| ROLLBACK["Rollback + retained evidence"]
    EVAL --> DATA["Experiment dataset"]
    DATA --> SURGEON
```

Both model paths share component identities, mutation plans, provenance, evaluation records, and training examples. The GGUF path adds mmap inspection, block-aligned selective decoding, exact-codec requantization, direct copying of unchanged ranges, resumable writes, and external validation.

See [ARCHITECTURE.md](ARCHITECTURE.md), the [architecture compatibility matrix](docs/architecture-compatibility.md), and the [design records](docs/design/).

## Safety and reproducibility

- Source checkpoints are not overwritten by default; new artifacts stage and publish atomically.
- Unknown architectures, tensor axes, codecs, constraints, or unsafe geometries fail closed.
- Stable component identities and explicit old-to-new mappings survive physical edits.
- Dataset splits group structural identities and run a leakage audit before publication.
- Feature and metric records retain precision, quantization, revision, seed, and hardware context.
- Large work is preflighted against RAM, VRAM, disk, and scratch budgets, then chunked or streamed.
- Persisted runs bind content-addressed artifacts to immutable recipes and explicit replay tolerances.

## Repository map

```text
src/modelsurgeon/
├── adapters/         # Hugging Face, safetensors, GGUF, architecture boundaries
├── graph/            # canonical components, topology, coupling, constraints
├── features/         # static, spectral, activation, gradient, runtime evidence
├── instrumentation/  # calibration, hooks, bounded aggregation
├── surgery/          # mutation contracts, masks, physical edits, rollback
├── evaluation/       # structure, quality, latency, memory, external validation
├── experiments/      # identity, persistence, budgets, queues, reproducibility
├── datasets/         # examples, stores, splits, leakage audits
├── surgeon/          # heuristic and learned decision models
├── search/           # constrained policies, state, Pareto infrastructure
├── explain/          # decision summaries and reproducible reports
└── cli/              # user-facing orchestration
```

## Development

```bash
uv sync --extra dev --extra hf --locked
uv run ruff check .
uv run mypy src
uv run pytest --cov=modelsurgeon --cov-report=term-missing
```

Pull-request CI runs linting, strict typing, the CPU test suite, a real local Transformers smoke path, and optional LightGBM integration coverage. GPU, large-model, `llama.cpp`, and long-running evidence workflows are kept separate from the small PR gate.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a change. Security reports follow [SECURITY.md](SECURITY.md).

## Documentation

- [Architecture](ARCHITECTURE.md) and [compatibility matrix](docs/architecture-compatibility.md)
- [Public API and compatibility policy](docs/api-compatibility.md)
- [Roadmap](ROADMAP.md) and [GitHub milestones](https://github.com/xdCloudy/ModelSurgeon/milestones)
- [Research evidence](docs/research/README.md) and [experiment guides](docs/experiments/README.md)
- [Design records](docs/design/)
- [Changelog](CHANGELOG.md) and [citation metadata](CITATION.cff)

## License

ModelSurgeon is licensed under [Apache-2.0](LICENSE). Model, dataset, tokenizer, and upstream-tool licenses remain the responsibility of their users.

# ModelSurgeon

**Learned automated structural optimization and pruning for neural networks.**

ModelSurgeon is experimental research software for learning which neural-network structures can be removed, merged, or approximated while retaining specified capabilities. It couples structural and statistical inspection with controlled mutation, tiered evaluation, experiment persistence, and progressively learned decision models.

> [!WARNING]
> Model surgery can permanently damage checkpoints. ModelSurgeon never overwrites an input model by default. Keep verified backups and treat every generated checkpoint as untrusted until it passes evaluation.

## Research loop

```text
target model -> inspect -> measure -> represent components -> predict mutation outcome
             -> mutate -> evaluate -> keep/rollback -> record -> retrain -> repeat
```

The first scientific milestone is intentionally modest: use static and activation features from small Hugging Face models to predict the perplexity effect of masking individual components, then compare a LightGBM baseline with magnitude and random selection.

## Hardware target

The primary development target is Windows 11/WSL2 with an RTX 3060 12 GB, i5-14600K, and 64 GB system RAM. CPU offload, bounded VRAM, streaming aggregation, disk-backed artifacts, resumability, mixed precision, and OOM recovery are architectural requirements rather than later optimizations.

## Current status

The repository is at the v0.1 scaffold stage. The CLI can parse stable component identifiers, load supported Hugging Face causal language models when the `hf` extra is installed, and walk their named module architecture. See [ROADMAP.md](ROADMAP.md) for the complete v1.0 plan.

## Installation

```bash
uv sync --extra dev --extra hf
uv run modelsurgeon inspect MODEL_ID_OR_PATH
```

The base install deliberately excludes heavyweight ML dependencies. Install the `hf` extra for PyTorch and Transformers support.

## CLI

```text
modelsurgeon inspect MODEL [--revision REVISION] [--trust-remote-code]
```

All commands accept `--log-level LEVEL` and `--log-format human|json`. JSON logs carry bound run, model, and component context for automation; normal command output remains on stdout.

Planned command groups include `calibrate`, `features`, `experiment`, `generate-dataset`, `train-surgeon`, `predict`, `search`, `report`, and `reproduce`.

## Principles

- Start with heuristics and classical models; add neural surgeon architectures only when evidence warrants them.
- Prefer safe masking experiments before destructive structural resizing.
- Escalate evaluation cost only for promising candidates.
- Use stable component identities and a coupling graph to prevent invalid surgery.
- Record sufficient provenance to reproduce every result.
- Keep operation local-first; do not require distributed infrastructure or datacenter GPUs.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run mypy src
uv run pytest
```

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), [ARCHITECTURE.md](ARCHITECTURE.md), and [SECURITY.md](SECURITY.md) first.

## License

Apache-2.0. Model and dataset licenses remain the responsibility of their users.

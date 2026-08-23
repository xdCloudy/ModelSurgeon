# Contributing

ModelSurgeon is research software with safety-sensitive checkpoint mutation. Before starting work, select a `Ready` issue, confirm its dependencies are closed, and discuss changes that alter component identity, graph semantics, experiment schemas, or checkpoint formats.

## Workflow

1. Branch from `main` and keep the change scoped to one finite issue.
2. Add tests and documentation required by the issue acceptance criteria.
3. Run `uv run ruff check .`, `uv run mypy src`, and `uv run pytest`.
4. Open a pull request using the template and link the issue.

Do not commit model weights, datasets, generated checkpoints, credentials, or experiment artifacts. New dependencies require a rationale covering maintenance, license, CPU behavior, and memory impact.

## Research changes

Record hypotheses, evaluation protocol, seeds, model/dataset revisions, hardware, and negative results. Comparisons must include appropriate random and simple-heuristic baselines. Do not claim cross-model generalization without a model- or architecture-held-out split.


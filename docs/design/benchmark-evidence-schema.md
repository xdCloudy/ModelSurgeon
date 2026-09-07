# Deployable benchmark evidence schema

`modelsurgeon.evaluation.benchmark_schema` defines the versioned evidence record
for competitive and reference benchmark cells. It is deliberately framework
neutral: a runner may use Transformers, native GGUF, a subprocess evaluator, or
an external competitor adapter while producing the same auditable record.

## Record boundary

Each `BenchmarkEvidenceRecord` carries immutable identities for the model,
corpus/task/split, method implementation, hardware/runtime, equal-budget limits,
random seeds, tool/evaluator/configuration, and physical output artifact. Its
`cell_id` is a SHA-256 identity over those inputs. Changing any one of those
inputs changes the cell identity; measured metrics and terminal outcome do not
silently change the cell being compared.

The result states are explicit: `measured`, `unsupported`, `failed`, and
`incomplete`. A missing metric is represented as `unavailable` with a reason,
never as zero, infinity, or another numeric sentinel. Measured metrics retain
their unit and may carry lower/upper uncertainty bounds. Reliability retains
repetition counts, successful repetitions, and confidence intervals.

Physical artifacts have their own state and, when available, must carry a
lowercase SHA-256 digest and positive byte size. This prevents a quality number
from being presented as evidence of a published or reloadable artifact.

## Serialization and migration

The canonical JSON record is schema version 1. `migrate_benchmark_record` keeps
the checked-in v0 flat fixture shape reproducible by moving `status` to
`outcome` and `metrics` to `quality_metrics`, while adding explicit empty groups
for runtime and optimization cost. Unknown versions fail closed.

`render_benchmark_report` emits canonical JSON or Markdown. Both retain metric
units, uncertainty, outcome reasons, tool/evaluator revisions, and artifact
lineage. Markdown is a view, not a new source of truth.

## Protocol responsibilities

This schema does not select competitors, datasets, or statistical tests. A
protocol owner must preregister those choices, licenses, applicability,
contamination checks, budgets, seeds, and decision thresholds, then retain all
unsupported, failed, and incomplete cells alongside successful measurements.

# Physical compression and quality-loss Pareto study

The v1.2 physical study is represented by `modelsurgeon.evaluation.physical_pareto`.
It is the evidence boundary between cumulative physical surgery, deployment benchmarking,
and a conservative Pareto claim.

## Matrix and provenance

Every cell is matched by source artifact digest, model family, physical format, method,
compression target, quality-loss limit, held-out corpus, evaluator, runtime, checkpoint
revision, tool revision, and random seed. The default preregistered matrix contains two
families, five compression targets, three quality-loss limits, five methods, and three
seeds. Unsupported, failed, and unknown cells remain in the serialized study.

## Measured-cell gates

A measured cell must bind a complete `PhysicalArtifactOutcome` to a measured
`DeploymentBenchmarkRecord`. The digest, byte size, and HF/GGUF format must agree across
both records. The output must be physically smaller than the source, and its active
parameter, artifact-byte, quality-loss, runtime, memory, disk, and optimization metrics
must be present. Deployment intervals use the retained repeated-run dispersion; quality
loss carries the held-out bootstrap interval.

The frontier compares worst cases: a lower-is-better metric is dominated only when the
candidate's upper bound is no larger than the other's lower bound, and a higher-is-better
metric uses the inverse rule. At least one metric must be strictly separated. Incomplete,
unsupported, failed, or overlapping evidence cannot create a superiority claim.

## Reproduction

```python
from modelsurgeon.evaluation import build_default_physical_pareto_study

study = build_default_physical_pareto_study()
print(study.study_id)
```

Replace cells with measured results through `build_physical_pareto_cell`, then pass the
complete matrix to `build_physical_pareto_study` to derive the frontier. The default record
is intentionally a negative evidence record until the licensed models, held-out corpus,
and bounded runtimes are available.

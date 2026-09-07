# Autonomous optimize orchestrator

`modelsurgeon optimize --execute` is the bounded execution boundary for the
v2 autonomous workflow. It composes trusted runtime adapters with the existing
capability/candidate planning, search resume, worker, campaign-promotion, and
artifact safety contracts. The orchestrator does not load a model, select
tensors, or invent measurements itself.

## Invocation

```text
modelsurgeon optimize \
  --model HuggingFaceTB/SmolLM2-135M \
  --revision <immutable-revision> \
  --preset balanced \
  --hardware-profile cpu-small \
  --execute \
  --state artifacts/optimize/run.json \
  --runtime my_project.runtime:factory \
  --approve plan_review \
  --approve source_model \
  --approve resource_budget \
  --approve artifact_write \
  --json
```

`--runtime` is a trusted `module:factory` adapter implementing
`run_stage(context) -> StageResult`. A runtime must return measured, complete,
constraint-passing evidence and a distinct immutable child artifact before a
result can be promoted. Without a runtime, the built-in preflight adapter
validates the workflow and stops at an explicit `unknown` result; it never
pretends that a model was evaluated.

## State and resume

The state file is a versioned, atomically replaced JSON record. It contains the
deterministic run and stage IDs, the linear DAG dependencies, cursor, attempt
counts, stage evidence, approvals, overrides, alternatives, reasons, and
accepted artifact lineage. Completed stages are skipped on resume. A stage that
was interrupted while running is retried with a new attempt; completed work is
not duplicated. A changed plan or source identity is rejected.

The workflow stages are profile, capability, baseline, candidate generation,
active search, surgery, repair, quantization, deployment benchmark, Pareto
selection, and report. A non-supported stage outcome is retained and terminates
the run. If Pareto evidence is missing, incomplete, fails hard constraints, or
does not provide a feasible candidate, the run completes with an explicit
failure and no accepted artifact.

Approvals are bound to the canonical plan digest, the content-addressed plan
diff, an operator identity, a non-secret operator context, and an explicit
expiry. An approval for an older state is never reused: resume rejects a
material plan change and reports the deterministic diff ID and changed paths.
Overrides are accepted only with a matching `override:<name>` approval and are
passed to the runtime as data, never interpreted as commands.

The persisted stage evidence can be replayed without loading a model. The
replay digest and selected candidate are derived from canonical evidence and
stable candidate tie-breaks, so identical evidence produces the same strategy
decision ID and final candidate. `modelsurgeon optimize --package` emits a
directory containing the plan, run, decision evidence, and an offline-verifiable
Merkle index. A package must be signed with `--package-key-id` and
`--package-key-env`; unavailable external artifacts and non-deterministic
tolerances are retained in the manifest and reduce the claim to bounded
reproducibility.

The schema is `autonomous_optimize_run`, version 2. Source paths and secrets
remain outside the state contract; artifact lineage uses content digests.

# Reproduce a persisted run

`modelsurgeon reproduce RUN_ID` resolves one immutable experiment run from the SQLite metadata store and its content-addressed artifact store. It never guesses missing evidence and never executes the command text stored in an artifact.

## Capture contract

A schema-v2 reproducibility manifest contains the complete canonical resolved configuration, exact original command arguments, full source/model/dataset/tool/seed/hardware/lock evidence, and optional absolute or relative tolerances for phase-qualified metrics such as `baseline:perplexity` or `delta:latency_seconds`.

The configuration is hashed again and must match the experiment input's recorded config digest. Tolerances may reference only measured original metrics. The manifest is linked to exactly one persisted candidate and stored under the `reproducibility_manifest` artifact role.

## Inspect before execution

```bash
uv run modelsurgeon reproduce run_<sha256> \
  --metadata ./artifacts/experiments.sqlite3 \
  --artifacts ./artifacts/store \
  --repository . \
  --lock ./uv.lock \
  --dry-run
```

The canonical JSON plan lists:

- the exact model, dataset, mutation, versions, seeds, and original outcome;
- the complete resolved configuration;
- the exact recorded command argument vector;
- original phase-qualified metrics and declared tolerances; and
- every source revision, worktree, lock, or stable hardware/software mismatch.

Volatile available/free RAM and disk readings are not environment identities. Total capacity, OS, CPU, CUDA devices/drivers, Python, PyTorch, and ModelSurgeon versions are compared.

## Replay through a trusted adapter

Execution requires an explicitly selected local adapter implementing `ReproductionExecutor.execute(plan)`:

```bash
uv run modelsurgeon reproduce run_<sha256> \
  --metadata ./artifacts/experiments.sqlite3 \
  --artifacts ./artifacts/store \
  --lock ./uv.lock \
  --executor my_project.replay:factory \
  --output ./reproduction.json
```

The adapter receives the verified immutable plan and returns a mapping of phase-qualified metric names to finite values. It is trusted code chosen by the operator; command text from the artifact is display/evidence only. Replay is refused if any environment mismatch remains.

The result links the original run, candidate, manifest, and exact command to every comparison. A metric passes when its absolute delta is no larger than the greater of its declared absolute tolerance and `abs(original) * relative_tolerance`. Missing and unexpected metrics fail. Output publication is non-overwriting, and a failed comparison exits nonzero while retaining the canonical result.

## Failure behavior

The command fails explicitly for unknown run IDs, missing or ambiguous manifests, corrupt content-addressed artifacts, unsupported manifest schemas, inconsistent run/model/dataset/config identities, absent measured metrics, malformed tolerances, unsafe environment drift, executor failures, and existing output paths.

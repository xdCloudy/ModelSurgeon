# First Surgeon proof run

Milestone `v0.5 — First Surgeon` separates implementation from the empirical proof. The
implementation can land without a large training run; the proof issue remains open until a real
mutation dataset has been trained and evaluated.

## Generate the proof dataset

The generic campaign orchestration is available through `first-surgeon-proof`. It enumerates
canonical mask candidates, obtains pre-mutation feature partitions before each mutation, executes
each candidate through the existing transactional single-mutation runner, builds canonical mutation
examples, creates a held-out-component split, runs the leakage audit, and writes training-ready
records.

```bash
modelsurgeon first-surgeon-proof \
  --runtime your_runtime_module:create_runtime \
  --output ./proof-data \
  --max-candidates 5000 \
  --seed 42 \
  --split-seed 43
```

The output directory contains:

- `examples.jsonl` — canonical supervised mutation examples;
- `split.json` — grouped component-level train/validation/test assignments;
- `leakage.json` — the mandatory leakage audit;
- `campaign.json` — candidate enumeration and dataset-build provenance.

The command fails rather than publishing a dataset if any split partition is empty or the leakage
audit reports a finding.

### Runtime boundary

A proof runtime uses the normal `SingleMutationExperimentRuntime` contract and adds four pieces of
model-specific evidence through `FirstSurgeonProofRuntime`:

- `component_graph` — the canonical graph used for candidate enumeration;
- `run_id` — the stable campaign run identity;
- `pre_mutation_feature_partitions(candidate)` — static and activation feature partitions captured
  before applying the candidate mutation;
- `experiment_record(candidate, result)` — the canonical experiment metadata/metrics bound to the
  rolled-back single-mutation result.

This deliberately keeps model/framework-specific tensor hooks outside the generic campaign layer.
For a real Hugging Face proof, the runtime must provide genuine PyTorch masking, model-forward,
tokenization/perplexity, and activation-feature adapters; the generic orchestrator does not emulate
those boundaries.

## Required dataset

Use a leakage-safe campaign dataset containing at least several thousand mask examples from a
small model. Each example must contain:

- canonical `pre_mutation_features` with static weight statistics and activation features;
- baseline and post-mutation perplexity observations;
- the exact model identifier, immutable revision, format, and quantization;
- a grouped split manifest whose test components do not occur in train or validation.

Do not create a row-level random split after examples are generated. `first-surgeon-proof` uses the
repository's grouped split machinery so component groups cannot leak across partitions.

## Environment

The base package deliberately does not make LightGBM a mandatory dependency. Install it only in the
environment used for the proof run:

```bash
uv sync --all-extras --locked
uv pip install 'lightgbm>=4,<5'
```

Record the exact LightGBM version in the proof issue together with the ModelSurgeon commit SHA.

## Train delta-perplexity regressor

```bash
modelsurgeon train-surgeon ./proof-data/examples.jsonl \
  --split ./proof-data/split.json \
  --registry ./artifacts/surgeons \
  --target perplexity \
  --baseline lightgbm-regressor \
  --threads 4 \
  --seed 42 \
  --json
```

The command fits preprocessing on train only, uses validation for early stopping, evaluates the
held-out test split only after model selection, and publishes an immutable model bundle containing
the preprocessing schema, target schema, split manifest, metrics, model revisions, quantization,
and run provenance.

## Train safe-mutation classifier

Choose safety limits before reading test results. Example only:

```bash
modelsurgeon train-surgeon ./proof-data/examples.jsonl \
  --split ./proof-data/split.json \
  --registry ./artifacts/surgeons \
  --target safe_mutation \
  --safe-threshold perplexity=0.25 \
  --baseline lightgbm-classifier \
  --threads 4 \
  --seed 42 \
  --top-n 50 \
  --json
```

If another safety metric is part of the policy, add another `--safe-threshold metric=value`. Missing
required observations are masked rather than silently treated as safe or unsafe.

## Baseline comparison

Run the same held-out candidates through the seeded-random, normalized-magnitude, and versioned
hand-crafted ranking baselines in `modelsurgeon.surgeon.ranking`. Record the seed, magnitude
normalization, component aggregation, heuristic version, missing-feature policy, and selection
propensities alongside the learned-model metrics.

The proof issue is complete only when the report contains, at minimum:

- safe classifier ROC-AUC and PR-AUC;
- precision/recall at the declared top-N;
- delta-perplexity MAE and RMSE;
- grouped-bootstrap confidence intervals;
- random and magnitude baseline comparison on the identical held-out candidates;
- train/validation/test group counts and confirmation that held-out components are disjoint;
- dataset/config/model revisions, seeds, CPU thread count, peak memory/VRAM where applicable, and
  elapsed training time.

## Inference smoke test

Use the digest emitted by training:

```bash
modelsurgeon predict-surgeon ./candidate.json \
  --registry ./artifacts/surgeons \
  --bundle sha256:<digest> \
  --json
```

Inference fails closed if the feature schema version differs or a feature required by the trained
preprocessor is absent.

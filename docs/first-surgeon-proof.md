# First Surgeon proof run

Milestone `v0.5 — First Surgeon` separates implementation from the empirical proof. The
implementation can land without a large training run; the proof issue remains open until a real
mutation dataset has been trained and evaluated.

## Generate the proof dataset

### Turnkey Hugging Face MLP-channel proof

For supported Hugging Face causal LMs with the standard gated `gate_proj` / `up_proj` /
`down_proj` MLP layout, use the production PyTorch runtime directly. It masks the selected
intermediate channel at the `down_proj` input, captures pre-mutation channel activations during the
baseline pass, computes channel weight statistics, measures token-weighted causal-LM perplexity,
and always removes the temporary hook after evaluation.

Install the optional HF runtime first:

```bash
uv sync --extra dev --extra hf --locked
```

Prepare a UTF-8 calibration corpus and run several thousand independently addressable MLP-channel
masks. Record the exact ModelSurgeon revision in the campaign provenance:

```bash
modelsurgeon first-surgeon-hf-proof <model-id-or-local-path> ./calibration.txt \
  --output ./proof-data \
  --max-candidates 5000 \
  --sequence-length 256 \
  --max-tokens 4096 \
  --safe-perplexity-delta 0.25 \
  --seed 42 \
  --split-seed 43 \
  --tool-revision "$(git rev-parse HEAD)"
```

For Hub models, the runtime records the resolved immutable model/tokenizer revision returned by
Transformers. Local calibration text is content-addressed with SHA-256. The command refuses model
layouts whose MLP projection dimensions do not match the declared intermediate width.

The proof runtime currently restricts this empirical campaign to **MLP-channel masks**. That is
intentional: a small transformer exposes thousands of independent channel candidates without
pretending that attention-head or physical structural surgery is implemented when it is not.

### Generic runtime entry point

The generic campaign orchestration remains available through `first-surgeon-proof` for another
adapter/runtime. It enumerates canonical mask candidates, obtains pre-mutation feature partitions
before each mutation, executes each candidate through the existing transactional single-mutation
runner, builds canonical mutation examples, creates a held-out-component split, runs the leakage
audit, and writes training-ready records.

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

A generic proof runtime uses the normal `SingleMutationExperimentRuntime` contract and adds four
pieces of model-specific evidence through `FirstSurgeonProofRuntime`:

- `component_graph` — the canonical graph used for candidate enumeration;
- `run_id` — the stable campaign run identity;
- `pre_mutation_feature_partitions(candidate)` — static and activation feature partitions captured
  before applying the candidate mutation;
- `experiment_record(candidate, result)` — the canonical experiment metadata/metrics bound to the
  rolled-back single-mutation result.

This keeps model/framework-specific tensor hooks outside the generic campaign layer. The built-in
Hugging Face runtime supplies those boundaries with real PyTorch hooks and causal-LM forwards; the
generic orchestrator does not emulate them.

## Required dataset

Use a leakage-safe campaign dataset containing at least several thousand mask examples from a
small model. Each example must contain:

- canonical `pre_mutation_features` with static weight statistics and activation features;
- baseline and post-mutation perplexity observations;
- the exact model identifier, immutable revision, format, and quantization;
- a grouped split manifest whose test components do not occur in train or validation.

Do not create a row-level random split after examples are generated. `first-surgeon-proof` and
`first-surgeon-hf-proof` use the repository's grouped split machinery so component groups cannot
leak across partitions.

## Training environment

The base package deliberately does not make LightGBM a mandatory dependency. Install it only in the
environment used for the proof training run:

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
and run provenance. Regression metrics include grouped-bootstrap confidence intervals when test
groups are available.

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
required observations are masked rather than silently treated as safe or unsafe. Classification
metrics include ROC-AUC, PR-AUC, precision/recall at the declared top-N, calibration error, and
grouped-bootstrap confidence intervals where mathematically defined.

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

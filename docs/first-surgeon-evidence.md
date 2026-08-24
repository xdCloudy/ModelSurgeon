# First Surgeon empirical evidence

The `first-surgeon-evidence` command consumes the leakage-audited dataset produced by
`first-surgeon-hf-proof` and runs the complete v0.5 learned-vs-baseline evaluation.

## Install the optional training backend

```bash
uv sync --all-extras --locked
uv pip install 'lightgbm>=4,<5'
```

## Run the evidence stage

```bash
modelsurgeon first-surgeon-evidence ./proof-data/examples.jsonl \
  --split ./proof-data/split.json \
  --registry ./proof-data/registry \
  --output ./proof-data/first-surgeon-evidence.json \
  --safe-perplexity-delta 0.25 \
  --threads 4 \
  --seed 42 \
  --top-n 50 \
  --bootstrap-repetitions 1000
```

The command fails closed if the held-out examples are not fully measured, if train/test safe labels
lack both classes, if the magnitude baseline cannot rank every held-out candidate, or if learned and
baseline methods are not evaluated on the identical held-out candidate set.

The evidence JSON records both immutable LightGBM bundle digests, grouped-bootstrap metrics,
seeded-random and magnitude rankings, hashes of the source dataset and split manifest, source model
and tool revisions, training seed/thread configuration, bounded process-memory telemetry, and
post-publication inference smoke predictions.

Do not mark the empirical proof complete merely because this command is available. Issue #104
requires the command to be run on the real several-thousand-mask dataset and the resulting evidence
attached to the issue.

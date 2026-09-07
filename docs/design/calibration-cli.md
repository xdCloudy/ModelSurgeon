# Calibration CLI contract

`modelsurgeon calibrate PLAN.json --cache CACHE.json` builds one bounded
Hugging Face calibration manifest. The plan is intentionally strict: every
dataset and tokenizer revision, trust rationale, preprocessing configuration,
selection seed, and tokenization bound is explicit and contributes to the
persisted contract.

The command accepts schema version `1` plans with this shape:

```json
{
  "schema_version": 1,
  "dataset": {
    "dataset": "org/data",
    "revision": "<immutable revision>",
    "split": "train",
    "license": "apache-2.0",
    "trust": "trusted",
    "trust_reason": "reviewed source and license",
    "metadata": {}
  },
  "preprocessing": {
    "name": "plain-text",
    "version": "1",
    "configuration": {"normalization": "none"}
  },
  "tokenizer": {
    "tokenizer": "org/tokenizer",
    "revision": "<immutable revision>",
    "configuration": {"trust_remote_code": false}
  },
  "selection": {"seed": 0, "sample_count": 512, "algorithm": "sha256-rank-v1"},
  "text_field": "text",
  "batch_size": 8,
  "max_tokens": 512
}
```

The command streams the pinned dataset and tokenizes only the deterministically
selected samples. A successful cache contains the calibration contract,
sample/content identities, token IDs, token count, and a `sha256:` cache
identity. Cache publication writes and fsyncs a same-directory temporary file
before replacement, so a failed or interrupted run cannot replace a completed
manifest with partial data. Existing caches are validated and reused unless
`--refresh` is supplied.

`--dry-run` validates the plan and reports its revisions and intended cache
without importing Hugging Face dependencies, reading the dataset, creating the
cache directory, or writing any dataset-derived state. The command reports
`token_count: null` and `cache_identity: null` in this mode because those values
depend on the streamed samples.

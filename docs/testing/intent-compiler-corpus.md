# Intent compiler equivalence and refusal corpus

The versioned fixture at `tests/fixtures/intent_compiler_corpus_v1.json` is a
regression boundary for the v2.1 conversational control plane. It stores
canonical `IntentRecord` values rather than provider prose, so the compiler and
policy layers are tested against typed fields, expected `OptimizationSpec`
records, ambiguity outcomes, diagnostic codes, and exact provenance.

Run it with the repository test suite:

```text
uv run pytest tests/test_intent_corpus.py
```

The test exercises the canonical compiler function and the public compiler
entrypoint. If another compiler implementation is added, pass it to
`run_intent_corpus` and keep its name stable. Provider availability is a
separate capability: an unavailable provider must be recorded as a known skip,
not treated as a passing compiler result.

## Adding a fixture

Add a complete canonical intent record under `cases`, with a stable
`case_id`. Use `equivalence_group` for rephrasings that should produce the
same canonical spec. The `expected` object must declare both compiler and
policy outcomes, exact specs (or `null`), sorted diagnostic codes, ambiguity
categories, exact provenance, a case classification, and `retained: true`.

Executable input records may carry an emitted spec, but the compiler still
rebuilds it from typed fields and refuses a mismatch. Non-executable policy
outcomes must have a `null` policy spec. Do not add a threshold, budget, or
operation to an expected spec unless it is present in the typed intent and
supported by the existing objective contract.

Every result is retained, including unsupported, refused, clarification,
inconclusive, and implementation-failure cases. A failed implementation is
reported in the corpus run instead of being dropped or converted into a
success. The runner enforces fixture-size, case-count, request-length,
field-count, ambiguity-count, and implementation-count budgets before replay.

Record the repository revision, corpus revision, exact command, environment,
available implementations, and known skips with any benchmark or release
evidence. This corpus does not claim model/provider quality; it checks the
deterministic typed boundary and its refusal behavior.

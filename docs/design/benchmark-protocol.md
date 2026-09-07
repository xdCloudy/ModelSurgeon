# Equal-budget competitive benchmark protocol

`modelsurgeon.evaluation.benchmark_protocol` is the executable v1.1 freeze for
competitive benchmark planning. It is a preregistration boundary: constructing
the manifest does not run a benchmark and does not convert an unknown result
into a success claim.

## Frozen contract

The protocol composes the existing seven-rung permissive Llama/Qwen/Mistral
ladder with two versioned methods, seven pinned task identities, two output
targets, equal resource limits, eight metric definitions, and three deterministic
seeds (`11`, `23`, and `47`). Every model/method/task combination is present as
one `ProtocolCell` with both an applicability state and a license decision.

Held-out perplexity is currently `supported`. ARC, HellaSwag, Winogrande,
MMLU, GSM8K, and code cells remain `unknown` until the evaluator capability and
contamination gates are satisfied. Those cells are retained in the manifest and
must later become measured, unsupported, failed, or incomplete; omission is not
permitted.

The metric contract records exact units, aggregation, direction, and decision
rules for quality, artifact, runtime, cost, and reliability. The statistical
plan uses paired hierarchical bootstrap units, 2,000 resamples, 95% intervals,
and declares the power and precision limitations before execution. A change to
any model, task, method, budget, seed, metric, or decision rule creates a new
protocol identity and preserves the prior manifest.

## Retained evidence

- Protocol manifest: `docs/research/v1.1-benchmark-protocol-v1.json`
- Contamination and license audit: `docs/research/v1.1-benchmark-contamination-license-audit-v1.json`
- Implementation: `src/modelsurgeon/evaluation/benchmark_protocol.py`
- Contract tests: `tests/test_benchmark_protocol.py`

The checked-in manifest envelope is content-addressed as
`protocol_d0a846ef00f22f2f77cc1dd9f3ae6b517a5019e0ade71df92194c8439f0c908e`.
The canonical full manifest is emitted by `render_benchmark_protocol()` from
the typed object named in `manifest_source`.
The accompanying audit is
`audit_1d263f08775c1d5e309ff3b294525261948a455ce03bf1946e1c1f5a74c4a119`.

The audit explicitly excludes Gemma until its license is approved for this
study. This is a licensing exclusion, not a quality result, and is kept
separate from the permissive ladder.

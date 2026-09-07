# Deterministic intent compiler

`modelsurgeon.conversation.IntentCompiler` is the bounded control-plane
boundary from a canonical `IntentRecord` to an immutable `OptimizationSpec`.
It consumes normalized fields only; it never parses or executes the original
request text. Provider output is untrusted until the fields pass the typed
compiler and the existing `ObjectiveContract` validation.

## Example

An intent record containing these fields:

```text
objective.metric = latency
quality.minimum = 0.98 (ratio)
budget.wall_time = 600 (seconds)
operations.allowed = [evaluate, search]
deployment.targets = [cpu]
```

compiles to the existing latency objective and immutable-source quality hard
constraint. Budgets, allowed operations and deployment targets are carried as
explicit envelope metadata. The compiler does not select tensors or add an
operation that was not present in the record.

## Outcomes and diagnostics

`compile_intent(record)` returns a `CompilationResult` with a canonical
`IntentOutcome` and sorted `CompilerDiagnostic` records. Missing hard
constraints and required ambiguities produce `clarification_required`.
Unknown metrics, operations or deployment capabilities produce `unsupported`.
Non-executable provider records and invalid boundary states produce `refused`.
Non-executable results never carry a spec.

The compiler has finite field and request-length bounds. Repeated requests
with the same canonical fields produce byte-identical spec serialization;
provenance remains attached to the returned intent record, while the spec
itself contains only explicit contract and envelope values.

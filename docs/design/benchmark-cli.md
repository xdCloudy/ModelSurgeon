# Reproducible benchmark CLI

The `modelsurgeon benchmark` command group turns the versioned v1.1 benchmark
protocol into a deterministic matrix that can be inspected before execution.

```text
modelsurgeon benchmark protocol --format json
modelsurgeon benchmark plan --output plan.json
modelsurgeon benchmark run --plan plan.json --state state.json
modelsurgeon benchmark resume --plan plan.json --state state.json
modelsurgeon benchmark audit --plan plan.json --state state.json
modelsurgeon benchmark report --plan plan.json --state state.json --format markdown
```

`plan` is content-addressed by the protocol, executor identity, and every cell.
Supported cells are `pending`; unsupported, infeasible, and unknown protocol
decisions are retained as terminal `unsupported` cells with their reasons.
No model is downloaded and no external method is run during planning.

`run` writes a state record only after a cell has reached a terminal result.
Existing cell results are immutable, so `resume` executes only missing pending
cells. `import` accepts one externally executed JSON result and refuses to
replace an existing cell. Both paths validate the plan and protocol identities
before changing state.

Executors implement `execute(cell) -> mapping` and must return
`{"status":"success", "metrics": {...}}`, or a terminal failure reason.
`SubprocessBenchmarkExecutor` is the bounded JSON-line adapter used for a tiny
real external-method smoke; it never invokes a shell and applies a wall-time
limit. The deterministic fake executor is reserved for unit and offline CLI
coverage.

Exit status is intentionally machine-readable:

- `0`: a valid plan/state and no method or comparison failures;
- `1`: a method failure or failed comparison is retained in state;
- `2`: protocol, plan, state, or import validation failed.

Reports preserve all terminal outcomes. A skipped or failed cell is never
silently removed from the matrix and cannot be counted as a measured success.

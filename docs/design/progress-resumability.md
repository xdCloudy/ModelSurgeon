# Optimize progress and resumability v1

`modelsurgeon.progress.ProgressStore` is the non-interactive progress boundary
for optimize campaigns. It persists one atomically replaced JSON state file per
campaign. The state contains a schema-versioned ordered event log and a
snapshot derived from that log.

## Event and recovery contract

Events cover the download, calibration, search, surgery, repair, evaluation,
and publication stages. Sequence numbers are contiguous; replaying a completed
stage is idempotent and does not add a duplicate event. Restarting the store
loads the last accepted snapshot and preserves deterministic `next_work`.

Pause, cancellation, and resume are explicit events. Cancellation retains the
source artifact digest and last accepted artifact digest; it does not roll back
or overwrite either artifact. Completed campaigns are terminal and resume is a
no-op. State writes use a temporary file followed by atomic replacement.

Each snapshot exposes completed/total work, elapsed time, linear ETA when
observations are sufficient, and an uncertainty explanation. Unknown or
failed outcomes remain explicit rather than being reported as success.

## Diagnostics and CLI

Diagnostic bundles contain the snapshot and ordered events plus an environment
mapping passed through recursive secret and local-path redaction. Existing
diagnostic files are never overwritten. The CLI emits plain JSON without
terminal control sequences:

```text
modelsurgeon progress init run-1 --stages download,search,evaluation --json
modelsurgeon progress event run-1 --stage download --status completed --completed 3 --total 10 --json
modelsurgeon progress pause run-1 --json
modelsurgeon progress resume run-1 --json
modelsurgeon progress diagnostics run-1 --output run-1-diagnostics.json --json
```

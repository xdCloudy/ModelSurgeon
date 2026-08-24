# Stage runtime telemetry

ModelSurgeon records stage-level runtime telemetry as immutable attempts associated with an experiment candidate. The telemetry contract is versioned independently from the older `StageTiming` summary rows so interrupted and resumed work can retain separate evidence without rewriting history.

## Recorded fields

Each `StageTelemetrySnapshot` records:

- wall-clock and process CPU seconds;
- processed token and candidate counts when known;
- derived token/s and candidate/s rates;
- peak process RSS;
- peak CUDA allocated and reserved bytes when a CUDA provider is available;
- process read/write byte deltas when the host exposes process I/O counters;
- a stable hardware-normalization context and `hwctx_...` identity;
- `complete` or `partial` state.

Unavailable host counters remain `null`. CPU-only measurements never fabricate CUDA values.

## Interrupt and resume semantics

`StageTelemetryRecorder.run()` executes the measured operation through the bounded memory telemetry collector. Final telemetry is persisted from the collector's guaranteed finalization path. If the operation raises, including an interrupt-style `BaseException`, the original exception propagates after a `partial` attempt is appended.

A resumed invocation of the same candidate/stage appends the next attempt number. Existing attempts are never overwritten. `ExperimentMetadataStore.list_stage_telemetry()` returns complete history, while `latest_stage_telemetry()` returns the newest attempt for each stage.

## Hardware-normalized comparison

`HardwareNormalizationContext` includes stable comparison fields: OS name, CPU architecture/model/core count, total system memory, and CUDA device name/memory/compute capability. Its canonical record is hashed into a `hwctx_...` identity. Runtime results are directly comparable only when their normalization context IDs match; callers may still implement a broader normalization model explicitly rather than silently comparing unlike hosts.

## Persistence

Experiment database schema v5 adds `experiment_stage_telemetry`. Rows are append-only by `(candidate_id, stage, attempt)` and indexed both by candidate/stage/attempt and hardware context/stage. Existing `experiment_stage_timings` rows remain unchanged for backward compatibility.

# Chat optimize execution adapter

The v2.3 chat slice can submit an executable `SpecPreview` through the stable
ModelSurgeon optimize APIs. The text provider remains a control-plane client:
only the validated, exact `SpecPreview` is accepted by the adapter, and no
provider text, tensor choice, measurement, or artifact claim is used as engine
evidence.

## Boundary

`ChatOptimizeAdapter` performs three bounded operations:

1. `preview()` creates a `preview_plan` read-only tool request and calls
   `build_optimize_plan(..., dry_run=False)`. Planning remains read-only and
   returns a deterministic plan identity, digest, outcome, and evidence ref.
2. `execute()` calls `SpecPreview.confirm()` and then creates an
   `execute_approved_plan` consequential request. The dispatcher validates the
   approval identity, plan digest, plan approval points, budget, and transaction
   before the handler calls `OptimizeOrchestrator`.
3. The handler returns only the orchestrator's canonical run/campaign outcome,
   retained evidence IDs, and an accepted immutable artifact ID when the
   existing promotion gate has accepted one.

The adapter translates only contract fields representable by the stable
`Settings` API. Unsupported metrics, plugin objectives, non-weighted modes,
non-absolute constraint baselines, invalid plan identities, and missing target
settings fail closed as explicit unsupported or failed outcomes.

## Progress, cancellation, and recovery

The trusted runtime wrapper emits one typed start and completion event per
orchestrator stage. These events may be streamed by the CLI as JSON records;
the final turn retains the same bounded events with the canonical tool result.
Cancellation is cooperative and reaches the runtime before a stage begins and
after it returns. Interrupted workflows are persisted by the existing
orchestrator and can be resumed with `--resume`; the adapter treats that as a
new explicit request against the same durable campaign rather than replaying a
previous paused tool response.

Example shape:

```text
modelsurgeon chat ./text-model.gguf \
  --target-model ./target-model \
  --target-revision <immutable-revision> \
  --state ./artifacts/chat-run.json \
  --preview-plan --request "retain quality and reduce latency"
```

Execution additionally requires `--execute`, an explicit `--approval-id`, and
the required stable plan approvals. A missing approval, unsupported input,
failed campaign, interruption, cancellation, or no-artifact result is retained
as a typed negative result and never presented as a successful artifact.

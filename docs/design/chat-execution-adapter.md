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
   existing promotion gate has accepted one. The result's evidence references
   are the IDs retained by the #469 `CampaignStateStore`, not provider output
   or raw stage labels.

The JSON file passed to `state_path` remains the optimizer's stage cursor and
resume record. The adapter also writes a sibling
`<state stem>.campaign.sqlite3` file as the canonical conversational record.
It binds the session, exact spec, plan, source digest, approval, provider
capability context, budgets, lifecycle, outcome, and append-only evidence.
The provider context is limited to the engine-owned identity/capability card;
transcript text, summaries, secrets, and tool payloads are rejected by the
campaign-state boundary.

The approval stored in campaign state is bound to the exact plan ID, plan
digest, material-diff ID, capability scope, operator context, reuse policy, and
expiry. `one_time` approvals cannot authorize a second execution; `reusable`
approvals can authorize bounded resumptions until expiry. The adapter projects
the orchestrator's immutable, redacted approval audit chain into campaign
state, preserving direct/chat parity without treating transcript text as audit
evidence.

The adapter translates only contract fields representable by the stable
`Settings` API. Unsupported metrics, plugin objectives, non-weighted modes,
non-absolute constraint baselines, invalid plan identities, and missing target
settings fail closed as explicit unsupported or failed outcomes.

## Progress, cancellation, and recovery

The trusted runtime wrapper emits one typed start and completion event per
orchestrator stage. These events may be streamed by the CLI as JSON records;
the final turn retains the same bounded events with the canonical tool result.
Cancellation is cooperative and reaches the runtime before a stage begins and
after it returns. The adapter uses the supplied `--state` path as the
WAL-backed canonical campaign store and keeps the stage cursor in a sibling
`.execution.json` file. The campaign store records `created -> running ->
paused/completed/failed/cancelled` transitions with deterministic IDs and
provenance; a stale or expired approval cannot be resumed. `pause()`,
`resume()`, `cancel()`, `reconnect()`, and `restart()` operate on those typed
records and are safe to retry by operation ID.

Interrupted workflows are persisted by the existing orchestrator and can be
resumed with `--resume`; the adapter treats that as a new explicit request
against the same canonical campaign rather than replaying a previous paused
tool response. Completed stage records are read and never evaluated again
after reconnect or process restart. Cancellation retains already committed
evidence and makes the campaign terminal; it never promotes a partial result.

Example shape:

```text
modelsurgeon chat ./text-model.gguf \
  --target-model ./target-model \
  --target-revision <immutable-revision> \
  --state ./artifacts/chat-campaign.sqlite3 \
  --preview-plan --request "retain quality and reduce latency"
```

Execution additionally requires `--execute`, an explicit `--approval-id`, and
the required stable plan approvals. `--approval-expires-at` and repeated
`--approval-reuse code=one_time|reusable` options apply the same lifecycle as
the direct optimize command. A missing, expired, overbroad, or stale approval,
unsupported input,
failed campaign, interruption, cancellation, or no-artifact result is retained
as a typed negative result and never presented as a successful artifact. An
accepted run is `completed/supported` with an immutable artifact; a rejected
candidate is `completed/failed` with a retained `decision: rejected` record;
unsupported plans are `completed/unsupported`; runtime errors are `failed`; and
interrupted runs are `paused/unknown` and resumable. Cancellation before the
consequential boundary creates no campaign or artifact state.

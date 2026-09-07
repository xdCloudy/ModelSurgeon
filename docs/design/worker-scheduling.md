# Resource-aware worker scheduling

The worker scheduler is a bounded in-process control-plane contract for
independent experiment cells. It supports the same typed task protocol for a
local CPU worker, a local multi-GPU worker, and an explicitly registered remote
worker. It is not a cloud orchestration service and does not open a public
listener or carry shell commands.

## Registration and capability boundaries

Each worker registers a versioned `WorkerCapabilityProfile` containing its
transport kind, allowlisted `WorkerOperation` values, CPU/RAM/VRAM/disk
capacity, immutable hardware inventory, revision, and concurrency limit.
Registration requires a caller-held credential; the scheduler stores only its
SHA-256 digest. Re-registering a worker with a different credential is rejected.
Remote transport is an explicit profile value and does not imply network
exposure.

## Tasks and placement

`WorkerTaskSpec` contains a campaign ID, one allowlisted operation,
content-addressed inputs, deterministic resource estimates, priority, seed,
and retry bound. Its task ID is a canonical hash of these fields. Placement
sorts by priority then task ID and workers by worker ID. A task is assigned only
when the worker advertises the operation and the aggregate RAM, VRAM, disk,
CPU, GPU, and concurrency reservations remain within capacity. CPU-only
workers cannot receive VRAM work.

Campaign budgets cap accepted task count and active task count. Re-submitting
the same canonical task is idempotent; changing its content cannot bypass the
quota because the task ID is content-derived.

## Leases, recovery, and publication

Assignments have expiring leases and monotonic heartbeat timestamps. An
expired assignment releases reservations and returns to the queue until its
bounded retry count is exhausted. Stale lease tokens cannot complete a task.
Results require the owning worker ID and, for supported outcomes, a
content-addressed output digest. A completed result can be published again
only if it is byte-for-byte identical; a conflicting duplicate is rejected.
This prevents lost or duplicated workers from double-publishing artifacts or
exceeding campaign budgets.

The scheduler returns explicit supported, unsupported, failed, unknown, queued,
completed, and exhausted states. The existing experiment metadata queue and
stage resource guards remain responsible for durable candidate leases and
stage-local measurements; this module provides the typed worker placement and
control-plane boundary around them.

# Autonomous campaign coordinator

The coordinator is the approval and promotion boundary around the existing
lease-aware `CampaignRunner`, stage resource budgets, OOM recovery, state
machine, transactions, and checkpoint lineage store. It does not duplicate
candidate execution; it prevents an executor result from being promoted unless
the evidence is complete and independently safe.

An `ApprovedCampaignPlan` binds source artifact digest, objective contract,
candidate-space identity, candidate IDs, seed, evaluation budget, and approval
ID. `run_approved` accepts an existing runner through a typed protocol, keeping
leases, heartbeats, resume, and fault isolation in the established runner.

## Promotion gate

Promotion requires a plan member, measured and complete evidence, passed hard
constraints, a committed mutation transaction, a distinct immutable child
artifact, and the unchanged source digest. Predicted, partial, constraint-
failing, uncommitted, corrupt, duplicate, or source-changing results produce
explicit unknown, rejected, or failed decisions and never become accepted
state.

Promotion decisions are content-addressed by plan, candidate, evaluation,
artifact, outcome, and reason. Repeating the same candidate decision is
idempotent. Replanning removes measured rejected/failed candidates, retains
unknown candidates for more evidence, and emits a deterministic next-plan ID.

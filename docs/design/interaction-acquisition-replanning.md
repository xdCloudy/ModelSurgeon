# Interaction-aware acquisition and replanning

`modelsurgeon.active_learning.interaction_replanning` adds a state-bound
acquisition lifecycle on top of the existing explore/exploit policy, schedule,
and resource-budget contracts.

## State and budget boundary

Every candidate records the exact parent state and predictor revision used to
compile it. Planning drops candidates from any other parent state and retains
their IDs in `rejected_stale_candidate_ids`; a stale candidate can therefore
never enter an evaluation selection. Selections are ranked by utility and
safety together with epistemic uncertainty, interaction uncertainty, and
diversity. Cumulative wall, tier, GPU, and disk estimates are checked before a
candidate is selected.

## Outcome lifecycle

An outcome must identify the selected candidate, the exact parent state, a
lineage token, and observed resources. Accepted outcomes can advance to a new
state. Rejected, failed, unsupported, or surprising outcomes trigger a bounded
replan; with no replan budget an unsafe failure produces a terminal rollback.
Large interaction residuals also trigger replanning even when the caller did
not pre-classify the result as surprising.

The prior plan's complete outcome history is copied into the next plan,
including negative interactions. Pending selections are reported as invalidated
when a state changes, so descendants cannot be evaluated from stale state.
Plan IDs are canonical digests of state, generation, selections, and history,
which makes resume and audit checks deterministic.

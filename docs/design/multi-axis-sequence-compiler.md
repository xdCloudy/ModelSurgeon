# Multi-axis sequence compiler

`modelsurgeon.search.multi_axis` compiles already-planned architecture changes
against one exact `DeployableArchitectureState`. It is a pre-mutation contract;
it does not allocate tensors or select an optimization candidate.

Each `ArchitectureMutationRequest` names one changed axis, the source state ID,
the validated physical `MutationPlan`, the complete identity remap, and the
target state. The compiler requires the target mutation order to append the
plan identity, rejects stale source IDs, rejects axes that are unknown or
unsupported, and requires a legal hardware alignment decision for depth,
width, head, and hidden/embedding changes.

The sequence expands remaps across unaffected active components, tracks
invalidated identities, rejects reuse of removed identities, and retains every
step. It reconciles cumulative parameter and storage deltas against the root
and final states before accepting the sequence. Rejection results include the
first step index and the invariant that blocked compilation.

Quantization, placement, and mutation order remain part of the state identity,
so the compiler never treats non-commutative mixed operations as equivalent by
accident.

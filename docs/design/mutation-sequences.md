# Graph-valid mutation sequences v1

Every sequence plan names the exact source-state ID against which its graph-valid
mutation closure was compiled. Extending a different or newer state fails as stale.
The plan's identity remap must cover exactly its affected closure; the sequence adds
explicit retained mappings for unaffected active components and composes the result
back to every root identity.

Active, invalidated, and root component IDs are canonical sets. Removed or renamed
source IDs enter permanent sequence history and no later mapping may reuse them.
Cumulative parameter, FLOP, memory, and storage deltas are exact sums of accepted
plans. Targets outside the active graph and mutations that remove the entire state
fail before execution.

Ordered state IDs always preserve the real mutation order. A separate equivalence ID
sorts mutation IDs only when every step carries the same explicit commutativity proof,
every remap retains its identities, and affected closures are pairwise disjoint. This
allows safe deduplication of proven-independent masks without inferring that
structural removals, renumbering, or overlapping edits commute.

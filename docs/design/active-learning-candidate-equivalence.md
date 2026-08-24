# Active-learning candidate equivalence

Issue #110 identifies mutation equivalence from mutation kind, the sorted order-insensitive
coupling closure, and the complete canonical request parameters. Similar candidates remain
distinct whenever their kind, parameter values, or affected component set differs.

Adapters may explicitly declare equivalent mutation IDs. Those keys are namespaced by the
adapter identity and declaration version, preventing an equivalence rule from leaking
across adapters or revisions. Undeclared candidates always use canonical request semantics.

Deduplication first excludes equivalence classes already completed or actively in flight,
then retains the first candidate in each remaining class in deterministic pool order.
Every exclusion records the candidate, versioned equivalence key, and exact reason. A class
cannot simultaneously be supplied as completed and in flight.

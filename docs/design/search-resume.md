# Atomic search resume v1

Each append-only resume snapshot captures the canonical policy state, deterministic
tie/RNG seed and decision cursor, accepted lineage and Pareto-frontier checkpoint
IDs, evaluation/GPU/disk budget consumption, ordered pending evaluations, and the
evidence-arrival cursor. The frontier must be a subset of accepted lineage, and
policy selections must exactly match reserved evaluation budget.

SQLite stores immutable generation payloads plus a single latest-generation pointer.
A save checks the caller's expected generation, inserts the checksum-protected
snapshot, and advances that pointer in one transaction. A stale worker therefore
cannot overwrite progress made after interruption, process restart, or reboot.

Loading verifies SHA-256 integrity and reconstructs typed state with strict field and
schema checks. The policy ID binds config, seed, objectives, and constraints; the
restored decision cursor and ordered evidence state reproduce the same decisions
when evidence arrives in the same order.

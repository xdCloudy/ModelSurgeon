# Evaluated keep, rollback, and checkpoint lineage v1

Only measured hard-constraint evidence can decide lineage. A passing candidate must
have an applied transaction that still owns its mutable state; it commits into one
content-addressed child checkpoint with exactly one immutable accepted parent and
the complete evaluation record. The accepted checkpoint table is the sole source of
valid search roots.

If any measured constraint fails, the coordinator restores the mutation transaction
before releasing all candidate-scoped temporary artifact leases. It then persists a
rollback decision with evidence and released artifact IDs but creates no checkpoint.
Consequently rejected state and candidate IDs cannot be used as parents or promoted
back into the frontier.

The SQLite lineage persists roots, parent links, state IDs, artifact SHA-256s, and
evaluation evidence. Candidate decisions and candidate state IDs are immutable and
unique. Content-addressed shared artifacts are not blindly deleted; the release
contract is specifically for candidate-scoped leases, allowing the artifact layer to
retain shared blobs safely.

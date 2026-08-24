# Persistent Pareto archive v1

The Pareto archive binds one SQLite file to one immutable canonical objective set.
Candidates, objective estimates and optional confidence bounds, JSON payloads,
frontier membership, and insertion sequence persist across process restarts.

Dominance is deliberately conservative. For every configured objective, a
candidate's worst confidence bound must be no worse than the other candidate's best
bound, and at least one comparison must be strict. Minimize and maximize directions
are honored independently. Point estimates act as zero-width intervals. Overlapping
intervals or a missing objective make two candidates incomparable, so incomplete
evidence remains visible on the frontier instead of being guessed.

Candidate updates may only fill objectives that were previously missing; payloads
and existing evidence are immutable. Each insert/update runs in one transaction and
recomputes all memberships, which restores candidates when new evidence changes a
prior incomplete comparison. Queries are ordered by candidate ID, making insertion,
resume, and tie behavior deterministic.

# Architecture search lifecycle

Architecture candidates move through explicit stages: predicted, reserved,
materialized, evaluated, accepted, rejected, or failed. Reservations charge a
canonical architecture identity and its equivalent proven path exactly once;
the budget retains both identity sets so duplicate structural paths cannot
consume the evaluation budget twice.

Snapshots retain the ordered candidate set, decision cursor, evidence-arrival
cursor, accepted frontier, checkpoint lineage, budget, and every terminal
outcome. SQLite persistence advances a generation atomically and verifies the
content digest on reload. Restarting from a snapshot therefore produces the
same next decision for the same evidence order.

Only passing measured evidence may transition an evaluated state to accepted,
and only accepted candidates may enter the frontier. Predicted-only, unknown,
unsupported, and failed outcomes remain explicit and cannot become parent
checkpoints. The lifecycle is deliberately policy-neutral; beam, evolutionary,
and Bayesian selection belong to downstream policy issues.

The checked-in tests exercise deterministic restart, all important stage
transitions, measured-only promotion, frontier safety, duplicate suppression,
budget accounting, and failure/rejection retention.

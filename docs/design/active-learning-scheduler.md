# Active-learning evaluation scheduler

Issue #114 persists the complete acquisition decision before submitting any evaluation.
The schedule binds the canonical acquisition report to a SHA-256, tool revision, candidate
rank, reason, propensity, and policy score. Resume must present the same acquisition digest;
a changed selection fails rather than silently rerunning policy.

Each entry transitions from selected to scheduled to completed. Scheduling links the
canonical experiment ID, and completion links the resulting dataset example ID while
retaining all selection metadata. Atomic fsynced replacement protects every transition.
After interruption, `pending` returns only selected or scheduled entries from the persisted
batch, so partial execution resumes without reselection or duplicate completed work.
